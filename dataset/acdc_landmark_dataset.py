"""
ACDC 2-Channel Landmark Dataset
=================================
Loads cardiac MRI slices from the ACDC dataset with:
  - Channel 1: MRI image (normalised)
  - Channel 2: Segmentation mask (LV/RV/Myo, normalised 0-1)
  - Ground truth: RV insertion point centroids from .nrrd RVIP labels

Folder structure expected:
    image_dir/   patient001_frame01.nii.gz  ...
    mask_dir/    patient001_frame01_gt.nii.gz  ...   (seg mask, labels 0/1/2/3)
    rvip_dir/    patient001_frame01_rvip.nrrd  ...   (RVIP labels 0/1/2)

Label conventions:
    Seg mask  : 0=background, 1=RV, 2=Myocardium, 3=LV
    RVIP mask : 0=background, 1=upper insertion point, 2=lower insertion point
"""

import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import nrrd

from utils.heatmap import coords_to_heatmaps

# ── augmentation helpers (reused from landmark_dataset.py) ───────────────────

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


def enforce_superior_ordering(coords):
    """
    Ensure LM1 (coords[0:2]) is always the superior point (smaller y index).

    Any augmentation that flips or rotates the image can cause LM1 and LM2
    to swap their vertical positions. Calling this after all augmentations
    guarantees the channel assignment is consistent with training convention:
      channel 0 heatmap → superior (upper) RV insertion point
      channel 1 heatmap → inferior (lower) RV insertion point
    """
    x1, y1, x2, y2 = coords
    if y1 > y2:   # LM1 has drifted below LM2 — swap
        return np.array([x2, y2, x1, y1], dtype=np.float32)
    return coords


def random_rotate(image, coords, max_angle=45):
    H, W = image.shape
    angle = np.random.uniform(-max_angle, max_angle)
    cx_img, cy_img = W / 2.0, H / 2.0
    M = cv2.getRotationMatrix2D((cx_img, cy_img), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    new_coords = coords.copy()
    for i in range(2):
        px, py = coords[i * 2], coords[i * 2 + 1]
        nx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ny = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        new_coords[i * 2]     = np.clip(nx, 0, W - 1)
        new_coords[i * 2 + 1] = np.clip(ny, 0, H - 1)
    return rotated, new_coords


# ── dataset ───────────────────────────────────────────────────────────────────

class ACDCLandmarkDataset(Dataset):
    """
    2-channel ACDC landmark dataset.

    Returns (image_2ch, heatmaps, coords) where:
        image_2ch : [2, 256, 256] float32 tensor
                    channel 0 = normalised MRI
                    channel 1 = normalised seg mask (0-1 range)
        heatmaps  : [2, 256, 256] float32 Gaussian heatmaps
        coords    : [4] float32 (x1, y1, x2, y2) in 256-space
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        rvip_dir,
        slice_axis=2,
        augment=False,
        sigma=8,
        min_landmark_dist=5,
        min_slice_variance=0.01,
    ):
        self.image_dir          = image_dir
        self.mask_dir           = mask_dir
        self.rvip_dir           = rvip_dir
        self.slice_axis         = slice_axis
        self.augment            = augment
        self.sigma              = sigma
        self.min_landmark_dist  = min_landmark_dist
        self.min_slice_variance = min_slice_variance

        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No valid ACDC samples found. Check your data paths.")

        print(f"OK ACDCLandmarkDataset: {len(self.samples)} valid slices "
              f"(axis={slice_axis}, augment={augment})")

    def set_sigma(self, sigma):
        self.sigma = sigma

    def _build_samples(self):
        samples = []
        files   = sorted([f for f in os.listdir(self.image_dir)
                          if f.endswith(".nii.gz") and "_gt" not in f])

        for fname in files:
            # Derive matching file names
            base     = fname.replace(".nii.gz", "")   # patient001_frame01
            mask_f   = base + "_gt.nii.gz"
            rvip_f   = base + "_rvip.nrrd"

            img_path  = os.path.join(self.image_dir, fname)
            mask_path = os.path.join(self.mask_dir,  mask_f)
            rvip_path = os.path.join(self.rvip_dir,  rvip_f)

            if not os.path.exists(mask_path) or not os.path.exists(rvip_path):
                continue

            img       = nib.load(img_path).get_fdata().astype(np.float32)
            seg       = np.round(nib.load(mask_path).get_fdata()).astype(np.int32)
            rvip, _   = nrrd.read(rvip_path)

            # Use minimum slice count across all three volumes
            n_slices = min(img.shape[self.slice_axis],
                           seg.shape[self.slice_axis],
                           rvip.shape[self.slice_axis])

            for i in range(n_slices):
                rvip_2d = np.take(rvip, i, axis=self.slice_axis)

                # Skip slices without both RVIP labels
                if not (np.any(rvip_2d == 1) and np.any(rvip_2d == 2)):
                    continue

                # Extract landmark coordinates
                coords = self._extract_points(rvip_2d)
                if coords is None:
                    continue

                # Check minimum landmark distance in 256-space
                H_2d, W_2d = rvip_2d.shape
                dist = np.linalg.norm(
                    [coords[2] - coords[0], coords[3] - coords[1]]
                ) * 256 / max(H_2d, W_2d)
                if dist < self.min_landmark_dist:
                    continue

                # Check slice variance (skip blank slices)
                img_2d = np.take(img, i, axis=self.slice_axis)
                if img_2d.var() < self.min_slice_variance:
                    continue

                samples.append((fname, base, i, coords, H_2d, W_2d))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, base, slice_idx, coords, H_orig, W_orig = self.samples[idx]

        # ── load MRI slice ────────────────────────────────────────────────────
        img_path = os.path.join(self.image_dir, fname)
        img      = nib.load(img_path).get_fdata().astype(np.float32)
        img_2d   = np.take(img, slice_idx, axis=self.slice_axis)

        # Resize and normalise MRI
        img_resized = cv2.resize(img_2d, (256, 256))
        mu  = img_resized.mean()
        std = img_resized.std() + 1e-8
        img_resized = (img_resized - mu) / std

        # ── load seg mask slice ───────────────────────────────────────────────
        mask_path = os.path.join(self.mask_dir, base + "_gt.nii.gz")
        seg       = np.round(nib.load(mask_path).get_fdata()).astype(np.float32)
        seg_2d    = np.take(seg, slice_idx, axis=self.slice_axis)

        # Resize seg mask (nearest neighbour to preserve label values)
        seg_resized = cv2.resize(seg_2d, (256, 256),
                                 interpolation=cv2.INTER_NEAREST).astype(np.float32)

        # Normalise seg to 0-1 range (labels are 0,1,2,3 → divide by 3)
        seg_max = seg_resized.max()
        if seg_max > 0:
            seg_resized = seg_resized / seg_max

        # ── scale landmark coords to 256-space ───────────────────────────────
        x1, y1, x2, y2 = coords
        x1 = x1 * 256 / W_orig;  x2 = x2 * 256 / W_orig
        y1 = y1 * 256 / H_orig;  y2 = y2 * 256 / H_orig
        coords_scaled = np.array([x1, y1, x2, y2], dtype=np.float32)

        # ── augmentation (applied to MRI only, seg mask gets same transform) ─
        if self.augment:
            # Horizontal flip
            if np.random.rand() < 0.5:
                img_resized     = np.fliplr(img_resized).copy()
                seg_resized     = np.fliplr(seg_resized).copy()
                coords_scaled[0] = 255 - coords_scaled[0]
                coords_scaled[2] = 255 - coords_scaled[2]

            # Vertical flip
            if np.random.rand() < 0.5:
                img_resized     = np.flipud(img_resized).copy()
                seg_resized     = np.flipud(seg_resized).copy()
                coords_scaled[1] = 255 - coords_scaled[1]
                coords_scaled[3] = 255 - coords_scaled[3]

            # Rotation (apply same transform to both channels)
            if np.random.rand() < 0.7:
                # Generate one angle and apply to BOTH image and seg mask
                angle = np.random.uniform(-45, 45)
                H, W  = img_resized.shape
                M     = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)

                # Rotate image (bilinear) and update coords
                img_resized = cv2.warpAffine(img_resized, M, (W, H),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT)
                for i in range(2):
                    px, py = coords_scaled[i*2], coords_scaled[i*2+1]
                    coords_scaled[i*2]   = np.clip(M[0,0]*px + M[0,1]*py + M[0,2], 0, W-1)
                    coords_scaled[i*2+1] = np.clip(M[1,0]*px + M[1,1]*py + M[1,2], 0, H-1)

                # Rotate seg mask with SAME matrix (nearest neighbour, no interpolation)
                seg_resized = cv2.warpAffine(seg_resized, M, (W, H),
                                             flags=cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Intensity jitter (MRI only — don't jitter the seg mask)
            if np.random.rand() < 0.5:
                img_resized = img_resized * np.random.uniform(0.8, 1.2) \
                            + np.random.uniform(-0.1, 0.1)

            # Gaussian noise (MRI only)
            if np.random.rand() < 0.3:
                img_resized = img_resized \
                            + np.random.randn(*img_resized.shape).astype(np.float32) * 0.05

            # Elastic deform (MRI only — seg mask not deformed to avoid label artifacts)
            if np.random.rand() < 0.4:
                img_resized, coords_scaled = elastic_deform(
                    img_resized, coords_scaled, alpha=15, sigma=4
                )

            # Random crop/resize
            if np.random.rand() < 0.4:
                img_resized, coords_scaled = random_crop_resize(
                    img_resized, coords_scaled,
                    crop_ratio=np.random.uniform(0.80, 0.95)
                )

        # ── re-enforce superior/inferior ordering after all augmentations ───────
        # Flips and rotations can swap LM1/LM2 vertical positions.
        # This guarantees channel 0 = upper RVIP, channel 1 = lower RVIP.
        coords_scaled = enforce_superior_ordering(coords_scaled)
        coords_scaled = np.clip(coords_scaled, 0, 255)

        # ── stack into 2-channel input ────────────────────────────────────────
        image_2ch = np.stack([img_resized, seg_resized], axis=0)  # (2, 256, 256)

        # ── generate heatmaps ─────────────────────────────────────────────────
        heatmaps = coords_to_heatmaps(coords_scaled, (256, 256), sigma=self.sigma)

        return (
            torch.tensor(image_2ch,    dtype=torch.float32),
            torch.tensor(heatmaps,     dtype=torch.float32),
            torch.tensor(coords_scaled, dtype=torch.float32),
        )

    def _extract_points(self, rvip_mask):
        """
        Extract two RV insertion point landmarks from the RVIP mask.

        Label 1 = upper (anterior) RV insertion point
        Label 2 = lower (inferior) RV insertion point

        Uses centroid of each annotation cluster.
        LM1 = superior (smaller row index), LM2 = inferior.
        Returns (x1, y1, x2, y2) where x=col, y=row.
        """
        pts_1 = np.argwhere(rvip_mask == 1)
        pts_2 = np.argwhere(rvip_mask == 2)

        if len(pts_1) == 0 or len(pts_2) == 0:
            return None

        p1 = pts_1.mean(axis=0)   # (row, col)
        p2 = pts_2.mean(axis=0)

        # Enforce consistent ordering: LM1 = superior (smaller row index)
        if p1[0] > p2[0]:
            p1, p2 = p2, p1

        return np.array([p1[1], p1[0], p2[1], p2[0]], dtype=np.float32)