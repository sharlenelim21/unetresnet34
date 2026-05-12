"""
ACDCLandmarkDataset — using pre-computed heatmap point files
=============================================================
Loads cardiac MRI slices from the ACDC dataset.

Folder structure:
    image_dir/   patient001_frame01.nii.gz  ...   (MRI volumes)
    mask_dir/    patient001_frame01.nii.gz  ...   (seg masks 0/1/2/3)
    point_dir/   patient001_frame01.nii.gz  ...   (heatmap volumes H×W×S×2)

Point file format:
    4D volume (H, W, n_slices, 2)
    channel 0 → Gaussian heatmap for LM1 (upper/anterior RVIP)
    channel 1 → Gaussian heatmap for LM2 (lower/inferior RVIP)
    Peak of each channel = landmark coordinate

Seg mask labels:
    0=background, 1=RV, 2=Myocardium, 3=LV

in_channels=1 → [1, 256, 256] MRI only
in_channels=2 → [2, 256, 256] MRI + seg mask
               seg channel zeroed when RV absent (Fix 2)
"""

import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from utils.heatmap import coords_to_heatmaps


# ── helpers ───────────────────────────────────────────────────────────────────

def enforce_superior_ordering(coords):
    x1, y1, x2, y2 = coords
    if y1 > y2:
        return np.array([x2, y2, x1, y1], dtype=np.float32)
    return coords


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
    ch, cw = int(H * crop_ratio), int(W * crop_ratio)
    top  = np.random.randint(0, H - ch + 1)
    left = np.random.randint(0, W - cw + 1)
    resized = cv2.resize(image[top:top+ch, left:left+cw], (W, H))
    new_coords = coords.copy()
    new_coords[0] = np.clip((coords[0] - left) * W / cw, 0, W - 1)
    new_coords[1] = np.clip((coords[1] - top)  * H / ch, 0, H - 1)
    new_coords[2] = np.clip((coords[2] - left) * W / cw, 0, W - 1)
    new_coords[3] = np.clip((coords[3] - top)  * H / ch, 0, H - 1)
    return resized, new_coords


# ── dataset ───────────────────────────────────────────────────────────────────

class ACDCLandmarkDataset(Dataset):
    """
    ACDC landmark dataset using pre-computed heatmap point files.
    Supports both 1-channel (MRI only) and 2-channel (MRI + seg mask) input.
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        point_dir,
        in_channels=1,
        slice_axis=2,
        augment=False,
        sigma=8,
        min_landmark_dist=5,
        min_slice_variance=0.01,
    ):
        self.image_dir          = image_dir
        self.mask_dir           = mask_dir
        self.point_dir          = point_dir
        self.in_channels        = in_channels
        self.slice_axis         = slice_axis
        self.augment            = augment
        self.sigma              = sigma
        self.min_landmark_dist  = min_landmark_dist
        self.min_slice_variance = min_slice_variance

        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No valid ACDC samples found. Check your data paths.")

        ch_str = "MRI only" if in_channels == 1 else "MRI + seg mask"
        print(f"OK ACDCLandmarkDataset: {len(self.samples)} valid slices "
              f"({ch_str}, augment={augment})")

    def set_sigma(self, sigma):
        self.sigma = sigma

    def _extract_coords(self, hm_slice):
        """
        Extract (x1,y1,x2,y2) from a heatmap slice (H, W, 2).
        Peak of channel 0 = LM1, peak of channel 1 = LM2.
        Returns None if either channel has no signal.
        """
        hm1 = hm_slice[:, :, 0]
        hm2 = hm_slice[:, :, 1]

        if hm1.max() < 0.1 or hm2.max() < 0.1:
            return None

        r1, c1 = np.unravel_index(hm1.argmax(), hm1.shape)
        r2, c2 = np.unravel_index(hm2.argmax(), hm2.shape)

        p1 = np.array([float(r1), float(c1)])
        p2 = np.array([float(r2), float(c2)])

        # LM1 = superior (smaller row index)
        if p1[0] > p2[0]:
            p1, p2 = p2, p1

        return np.array([p1[1], p1[0], p2[1], p2[0]], dtype=np.float32)

    def _build_samples(self):
        samples = []
        files = sorted([f for f in os.listdir(self.image_dir)
                        if f.endswith(".nii.gz")])

        for fname in files:
            img_path   = os.path.join(self.image_dir, fname)
            mask_path  = os.path.join(self.mask_dir,  fname)
            point_path = os.path.join(self.point_dir, fname)

            if not os.path.exists(mask_path) or not os.path.exists(point_path):
                continue

            img    = nib.load(img_path).get_fdata().astype(np.float32)
            points = nib.load(point_path).get_fdata().astype(np.float32)
            # points: (H, W, n_slices, 2)

            n_slices = min(img.shape[self.slice_axis], points.shape[2])

            for i in range(n_slices):
                hm_slice = points[:, :, i, :]   # (H, W, 2)

                if hm_slice[:, :, 0].max() < 0.1 or hm_slice[:, :, 1].max() < 0.1:
                    continue

                coords = self._extract_coords(hm_slice)
                if coords is None:
                    continue

                H_2d, W_2d = hm_slice.shape[:2]
                dist = np.linalg.norm(
                    [coords[2] - coords[0], coords[3] - coords[1]]
                ) * 256 / max(H_2d, W_2d)
                if dist < self.min_landmark_dist:
                    continue

                img_2d = np.take(img, i, axis=self.slice_axis)
                if img_2d.var() < self.min_slice_variance:
                    continue

                samples.append((fname, i, coords, H_2d, W_2d))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, slice_idx, coords, H_orig, W_orig = self.samples[idx]

        # ── MRI ───────────────────────────────────────────────────────────────
        img    = nib.load(os.path.join(self.image_dir, fname)).get_fdata().astype(np.float32)
        img_2d = np.take(img, slice_idx, axis=self.slice_axis)
        img_r  = cv2.resize(img_2d, (256, 256))
        mu, std = img_r.mean(), img_r.std() + 1e-8
        img_r  = (img_r - mu) / std

        # ── seg mask ──────────────────────────────────────────────────────────
        seg    = np.round(nib.load(os.path.join(self.mask_dir, fname)).get_fdata()).astype(np.float32)
        seg_2d = np.take(seg, slice_idx, axis=self.slice_axis)
        seg_r  = cv2.resize(seg_2d, (256, 256),
                            interpolation=cv2.INTER_NEAREST).astype(np.float32)

        # Fix 2 — blank seg channel when RV (label 1) is absent
        if not np.any(np.round(seg_r) == 1):
            seg_r = np.zeros_like(seg_r)
        else:
            seg_max = seg_r.max()
            if seg_max > 0:
                seg_r = seg_r / seg_max

        # ── scale coords to 256-space ─────────────────────────────────────────
        cs = np.array([
            coords[0] * 256 / W_orig,
            coords[1] * 256 / H_orig,
            coords[2] * 256 / W_orig,
            coords[3] * 256 / H_orig,
        ], dtype=np.float32)

        # ── augmentation ──────────────────────────────────────────────────────
        if self.augment:
            if np.random.rand() < 0.5:
                img_r = np.fliplr(img_r).copy()
                seg_r = np.fliplr(seg_r).copy()
                cs[0] = 255 - cs[0];  cs[2] = 255 - cs[2]

            if np.random.rand() < 0.5:
                img_r = np.flipud(img_r).copy()
                seg_r = np.flipud(seg_r).copy()
                cs[1] = 255 - cs[1];  cs[3] = 255 - cs[3]

            if np.random.rand() < 0.7:
                angle = np.random.uniform(-45, 45)
                H, W  = img_r.shape
                M     = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
                img_r = cv2.warpAffine(img_r, M, (W, H),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
                for i in range(2):
                    px, py = cs[i*2], cs[i*2+1]
                    cs[i*2]   = np.clip(M[0,0]*px + M[0,1]*py + M[0,2], 0, W-1)
                    cs[i*2+1] = np.clip(M[1,0]*px + M[1,1]*py + M[1,2], 0, H-1)
                seg_r = cv2.warpAffine(seg_r, M, (W, H),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            if np.random.rand() < 0.5:
                img_r = img_r * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1)

            if np.random.rand() < 0.3:
                img_r = img_r + np.random.randn(*img_r.shape).astype(np.float32) * 0.05

            if np.random.rand() < 0.4:
                img_r, cs = elastic_deform(img_r, cs, alpha=15, sigma=4)

            if np.random.rand() < 0.4:
                img_r, cs = random_crop_resize(img_r, cs,
                                               crop_ratio=np.random.uniform(0.80, 0.95))

        cs = enforce_superior_ordering(cs)
        cs = np.clip(cs, 0, 255)

        # ── stack channels ────────────────────────────────────────────────────
        if self.in_channels == 1:
            image_out = np.expand_dims(img_r, axis=0)
        else:
            image_out = np.stack([img_r, seg_r], axis=0)

        heatmaps = coords_to_heatmaps(cs, (256, 256), sigma=self.sigma)

        return (
            torch.tensor(image_out, dtype=torch.float32),
            torch.tensor(heatmaps,  dtype=torch.float32),
            torch.tensor(cs,        dtype=torch.float32),
        )