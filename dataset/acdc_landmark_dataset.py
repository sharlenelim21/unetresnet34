"""
ACDC landmark dataset — mirrors landmark_dataset.py exactly.

Differences from LandmarkDataset:
  - Three folders (images, masks, points) share identical filenames
  - Points are 4-D .nii.gz (H, W, S, 2):
      channel 0 = LM1 binary mask  (superior/anterior RV insertion point)
      channel 1 = LM2 binary mask  (inferior RV insertion point)
  - patient_ids parameter restricts which files are loaded
  - in_channels=1 → MRI only [1,256,256]
    in_channels=2 → MRI + seg mask [2,256,256], mask /3 → [0,1]
"""

import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from utils.heatmap import coords_to_heatmaps


# ── augmentation helpers (copied verbatim from landmark_dataset.py) ────────────

def elastic_deform(image, coords, alpha=20, sigma=5, seed=None):
    rng = np.random.RandomState(seed)
    H, W = image.shape
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = np.clip(x + dx, 0, W - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, H - 1).astype(np.float32)
    image_def = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    new_coords = coords.copy()
    for i, (cx, cy) in enumerate([(coords[0], coords[1]), (coords[2], coords[3])]):
        ix = int(np.clip(round(cx), 0, W - 1))
        iy = int(np.clip(round(cy), 0, H - 1))
        new_coords[i * 2]     = np.clip(cx + dx[iy, ix], 0, W - 1)
        new_coords[i * 2 + 1] = np.clip(cy + dy[iy, ix], 0, H - 1)
    return image_def, new_coords


def random_crop_resize(image, coords, crop_ratio=0.85):
    H, W = image.shape
    ch = int(H * crop_ratio)
    cw = int(W * crop_ratio)
    top  = np.random.randint(0, H - ch + 1)
    left = np.random.randint(0, W - cw + 1)
    cropped = image[top:top+ch, left:left+cw]
    resized = cv2.resize(cropped, (W, H))
    new_coords = coords.copy()
    new_coords[0] = np.clip((coords[0] - left) * W / cw, 0, W - 1)
    new_coords[1] = np.clip((coords[1] - top)  * H / ch, 0, H - 1)
    new_coords[2] = np.clip((coords[2] - left) * W / cw, 0, W - 1)
    new_coords[3] = np.clip((coords[3] - top)  * H / ch, 0, H - 1)
    return resized, new_coords


def random_rotate(image, coords, max_angle=45, M=None):
    """
    Rotate image and coords by a random angle.
    If M is provided, use it directly (allows applying the same rotation
    to a second image, e.g. seg mask, without sampling a new angle).
    Returns (rotated_image, updated_coords, M) so the caller can reuse M.
    """
    H, W = image.shape
    if M is None:
        angle = np.random.uniform(-max_angle, max_angle)
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    new_coords = coords.copy()
    for i in range(2):
        px, py = coords[i * 2], coords[i * 2 + 1]
        nx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ny = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        new_coords[i * 2]     = np.clip(nx, 0, W - 1)
        new_coords[i * 2 + 1] = np.clip(ny, 0, H - 1)
    return rotated, new_coords, M


# ── per-patient normalisation cache (copied verbatim) ─────────────────────────

class _PatientNormCache:
    def __init__(self):
        self._cache = {}

    def get(self, img_path: str, img_array: np.ndarray):
        if img_path not in self._cache:
            mu  = img_array.mean()
            std = img_array.std() + 1e-8
            self._cache[img_path] = (mu, std)
        return self._cache[img_path]


_NORM_CACHE = _PatientNormCache()


# ── dataset ────────────────────────────────────────────────────────────────────

class ACDCLandmarkDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        rvip_dir,
        patient_ids,
        in_channels=1,
        augment=False,
        sigma=8,
        min_landmark_dist=5,
        min_slice_variance=0.01,
    ):
        self.image_dir          = image_dir
        self.mask_dir           = mask_dir
        self.rvip_dir           = rvip_dir
        self.patient_ids        = [p.lower() for p in patient_ids]
        self.in_channels        = in_channels
        self.augment            = augment
        self.sigma              = sigma
        self.min_landmark_dist  = min_landmark_dist
        self.min_slice_variance = min_slice_variance

        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No valid ACDC samples found. Check paths and patient IDs.")

        print(f"OK ACDCLandmarkDataset: {len(self.samples)} valid slices "
              f"(in_channels={in_channels})")

    def set_sigma(self, sigma):
        self.sigma = sigma

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _patient_files(self):
        return sorted(
            fn for fn in os.listdir(self.image_dir)
            if fn.endswith(".nii.gz")
            and any(pid in fn.lower() for pid in self.patient_ids)
        )

    def _build_samples(self):
        samples = []

        for fname in self._patient_files():
            img_path  = os.path.join(self.image_dir, fname)
            mask_path = os.path.join(self.mask_dir,  fname)
            rvip_path = os.path.join(self.rvip_dir,  fname)

            if not os.path.exists(mask_path) or not os.path.exists(rvip_path):
                continue

            try:
                img_vol  = nib.load(img_path).get_fdata().astype(np.float32)
                mask_vol = nib.load(mask_path).get_fdata().astype(np.float32)
                pts_vol  = nib.load(rvip_path).get_fdata()   # (H, W, S, 2)
            except Exception as exc:
                print(f"  [skip] {fname}: {exc}")
                continue

            if img_vol.ndim == 4:
                img_vol  = img_vol[..., 0]
            if mask_vol.ndim == 4:
                mask_vol = mask_vol[..., 0]

            if pts_vol.ndim != 4 or pts_vol.shape[3] != 2:
                print(f"  [skip] {fname}: unexpected points shape {pts_vol.shape}")
                continue

            H, W, n_slices = img_vol.shape

            for z in range(n_slices):
                coords = self._extract_points(pts_vol, z)
                if coords is None:
                    continue

                # Landmark separation filter (in 256-space)
                dist = np.linalg.norm([coords[2] - coords[0], coords[3] - coords[1]]) \
                       * 256 / max(H, W)
                if dist < self.min_landmark_dist:
                    continue

                # Variance filter — same logic as LandmarkDataset
                img_sl = img_vol[:, :, z]
                mu_s   = img_sl.mean()
                std_s  = img_sl.std() + 1e-8
                if ((img_sl - mu_s) / std_s).var() < self.min_slice_variance:
                    continue

                samples.append((fname, z, coords, H, W))

        return samples

    def _extract_points(self, pts_vol, z):
        """
        Extract LM1 and LM2 from pts_vol[:,:,z,:].
        channel 0 = LM1, channel 1 = LM2 (binary masks, value > 0.5).
        Returns (x1, y1, x2, y2) float32 in original voxel space,
        LM1 = superior (smaller y), LM2 = inferior (larger y).
        Returns None if either landmark is absent.
        """
        lm1_mask   = pts_vol[:, :, z, 0]
        lm1_coords = np.argwhere(lm1_mask > 0.5)   # rows: (row=y, col=x)
        if len(lm1_coords) == 0:
            return None

        lm2_mask   = pts_vol[:, :, z, 1]
        lm2_coords = np.argwhere(lm2_mask > 0.5)
        if len(lm2_coords) == 0:
            return None

        lm1_y, lm1_x = lm1_coords.mean(axis=0)
        lm2_y, lm2_x = lm2_coords.mean(axis=0)

        # Enforce LM1 = superior (smaller y = higher in image)
        if lm1_y > lm2_y:
            lm1_x, lm1_y, lm2_x, lm2_y = lm2_x, lm2_y, lm1_x, lm1_y

        return np.array([lm1_x, lm1_y, lm2_x, lm2_y], dtype=np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, z, coords, H_orig, W_orig = self.samples[idx]

        img_path  = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir,  fname)

        img_vol  = nib.load(img_path).get_fdata().astype(np.float32)
        mask_vol = nib.load(mask_path).get_fdata().astype(np.float32)
        if img_vol.ndim  == 4: img_vol  = img_vol[...,  0]
        if mask_vol.ndim == 4: mask_vol = mask_vol[..., 0]

        img_2d  = img_vol[:,  :, z]
        mask_2d = mask_vol[:, :, z]

        # Per-patient normalisation (same as LandmarkDataset)
        mu, std   = _NORM_CACHE.get(img_path, img_vol)
        image_res = cv2.resize(img_2d, (256, 256))
        image_res = (image_res - mu) / std

        # Scale coords to 256×256 space
        x1, y1, x2, y2 = coords
        x1 = x1 * 256 / W_orig;  x2 = x2 * 256 / W_orig
        y1 = y1 * 256 / H_orig;  y2 = y2 * 256 / H_orig
        coords_scaled = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Resize seg mask to 256×256 before augmentation so geometric transforms
        # can be applied to both channels with the same parameters.
        mask_res = cv2.resize(mask_2d, (256, 256),
                              interpolation=cv2.INTER_NEAREST).astype(np.float32)

        # Augmentation — geometric ops applied identically to image AND mask.
        if self.augment:
            # Horizontal flip
            if np.random.rand() < 0.5:
                image_res        = np.fliplr(image_res).copy()
                mask_res         = np.fliplr(mask_res).copy()
                coords_scaled[0] = 255 - coords_scaled[0]
                coords_scaled[2] = 255 - coords_scaled[2]

            # Vertical flip
            if np.random.rand() < 0.5:
                image_res        = np.flipud(image_res).copy()
                mask_res         = np.flipud(mask_res).copy()
                coords_scaled[1] = 255 - coords_scaled[1]
                coords_scaled[3] = 255 - coords_scaled[3]

            # Rotation — sample M once, apply to both image and mask (Issue 6)
            if np.random.rand() < 0.7:
                image_res, coords_scaled, M = random_rotate(
                    image_res, coords_scaled, max_angle=45
                )
                # Apply the exact same rotation matrix to the seg mask
                H_r, W_r = mask_res.shape
                mask_res = cv2.warpAffine(mask_res, M, (W_r, H_r),
                                          flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_REFLECT)

            # Intensity-only augmentations (image only, mask unchanged)
            if np.random.rand() < 0.5:
                image_res = image_res * np.random.uniform(0.8, 1.2) \
                          + np.random.uniform(-0.1, 0.1)
            if np.random.rand() < 0.3:
                image_res = image_res \
                          + np.random.randn(*image_res.shape).astype(np.float32) * 0.05

            # Elastic deform — apply to image only (mask stays with original pixels)
            if np.random.rand() < 0.4:
                image_res, coords_scaled = elastic_deform(
                    image_res, coords_scaled, alpha=15, sigma=4
                )

            # Crop-resize — apply to both
            if np.random.rand() < 0.4:
                cr = np.random.uniform(0.80, 0.95)
                image_res, coords_scaled = random_crop_resize(
                    image_res, coords_scaled, crop_ratio=cr
                )
                H_m, W_m = mask_res.shape
                ch = int(H_m * cr);  cw = int(W_m * cr)
                top  = np.random.randint(0, H_m - ch + 1)
                left = np.random.randint(0, W_m - cw + 1)
                mask_res = cv2.resize(mask_res[top:top+ch, left:left+cw],
                                      (W_m, H_m), interpolation=cv2.INTER_NEAREST)

            # Re-enforce LM1=superior after ALL geometric augmentations (Issue 5)
            if coords_scaled[1] > coords_scaled[3]:
                coords_scaled = np.array([
                    coords_scaled[2], coords_scaled[3],
                    coords_scaled[0], coords_scaled[1],
                ], dtype=np.float32)

        coords_scaled = np.clip(coords_scaled, 0, 255)

        # Build image tensor
        if self.in_channels == 2:
            mask_norm = (mask_res / 3.0).astype(np.float32)
            image_out = np.stack([image_res.astype(np.float32), mask_norm], axis=0)
        else:
            image_out = np.expand_dims(image_res.astype(np.float32), axis=0)

        heatmaps = coords_to_heatmaps(coords_scaled, (256, 256), sigma=self.sigma)

        return (
            torch.tensor(image_out,     dtype=torch.float32),
            torch.tensor(heatmaps,      dtype=torch.float32),
            torch.tensor(coords_scaled, dtype=torch.float32),
        )
