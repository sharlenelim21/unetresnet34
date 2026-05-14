"""
ACDC landmark dataset — loads MRI slices with RV insertion point annotations.

Each __getitem__ returns one 2-D slice that has both LM1 (anterior/superior
RV insertion point) and LM2 (inferior insertion point, larger y).
"""

import os
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom as ndimage_zoom, rotate as ndimage_rotate
import nibabel as nib

from utils.heatmap import coords_to_heatmaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hist_standardise(img: np.ndarray) -> np.ndarray:
    """Clip [p1, p99] then z-score normalise. Returns float32."""
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    mu, sd = img.mean(), img.std()
    if sd < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mu) / sd).astype(np.float32)


def _resize2d(arr: np.ndarray, target_h: int, target_w: int, order: int) -> np.ndarray:
    h, w = arr.shape
    return ndimage_zoom(arr, (target_h / h, target_w / w), order=order).astype(np.float32)


def _scale_coord(xy, h_orig, w_orig, H, W):
    """Scale (x, y) from original voxel space to resized 256×256 space."""
    return (xy[0] * W / w_orig, xy[1] * H / h_orig)


# ---------------------------------------------------------------------------
# NIfTI landmark loader
# ---------------------------------------------------------------------------

def _load_rvip_nifti(path: str) -> dict:
    """
    Load a .nii.gz landmark label mask and return
        { z_index: (lm1_xy, lm2_xy) }
    where lm1_xy and lm2_xy are (float x, float y) tuples,
    and lm2 always has the larger y (inferior).

    The volume has the same (H, W, Z) layout as the MRI and mask volumes.
    Pixel value 1 = LM1, pixel value 2 = LM2, 0 = background.
    The centroid of each label on each slice is used as the coordinate.
    """
    vol = nib.load(path).get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]

    H, W, Z = vol.shape
    coords: dict = {}

    for z in range(Z):
        sl = vol[:, :, z]
        entry = {}
        for label in (1, 2):
            ys, xs = np.where(sl == label)
            if len(xs):
                entry[label] = (float(xs.mean()), float(ys.mean()))
        if 1 in entry and 2 in entry:
            lm1, lm2 = entry[1], entry[2]
            if lm1[1] > lm2[1]:    # enforce lm2 = inferior (larger y)
                lm1, lm2 = lm2, lm1
            coords[z] = (lm1, lm2)

    return coords


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_image(directory: str, pid: str, frame: str) -> str | None:
    """
    Find the MRI .nii.gz file for patient `pid` and frame ED/ES.
    Handles naming styles:
      patient001_frame01.nii.gz  (ED=frame01, ES=frameXX)
      patient001_ED.nii.gz
    """
    tag = frame.upper()
    best = []
    for fn in os.listdir(directory):
        fn_up = fn.upper()
        if pid.upper() not in fn_up:
            continue
        if not fn.endswith(".nii.gz"):
            continue
        if "_GT" in fn_up:
            continue
        # Frame matching
        if tag in fn_up:
            best.append(fn)
        elif tag == "ED" and "FRAME01" in fn_up and "FRAME01" in fn_up:
            best.append(fn)
        elif tag == "ES" and "FRAME" in fn_up and "FRAME01" not in fn_up:
            best.append(fn)
    if best:
        return os.path.join(directory, sorted(best)[0])
    # Fallback: any non-gt file with this pid
    for fn in sorted(os.listdir(directory)):
        if pid.upper() in fn.upper() and fn.endswith(".nii.gz") and "_GT" not in fn.upper():
            return os.path.join(directory, fn)
    return None


def _find_mask(directory: str, pid: str, frame: str) -> str | None:
    tag = frame.upper()
    best = []
    for fn in os.listdir(directory):
        fn_up = fn.upper()
        if pid.upper() not in fn_up:
            continue
        if not fn.endswith(".nii.gz"):
            continue
        if "_GT" not in fn_up:
            continue
        if tag in fn_up:
            best.append(fn)
    if best:
        return os.path.join(directory, sorted(best)[0])
    for fn in sorted(os.listdir(directory)):
        if pid.upper() in fn.upper() and fn.endswith(".nii.gz") and "_GT" in fn.upper():
            return os.path.join(directory, fn)
    return None


def _find_rvip(directory: str, pid: str, frame: str) -> str | None:
    """Find the landmark .nii.gz file for patient `pid` and frame ED/ES."""
    tag = frame.upper()
    for fn in sorted(os.listdir(directory)):
        fn_up = fn.upper()
        if pid.upper() not in fn_up:
            continue
        if not fn.endswith(".nii.gz"):
            continue
        if tag in fn_up:
            return os.path.join(directory, fn)
    # Fallback: first .nii.gz with this pid
    for fn in sorted(os.listdir(directory)):
        if pid.upper() in fn.upper() and fn.endswith(".nii.gz"):
            return os.path.join(directory, fn)
    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ACDCLandmarkDataset(Dataset):
    """
    Args:
        image_dir:   directory of .nii.gz MRI volumes
        mask_dir:    directory of _gt.nii.gz segmentation masks
        rvip_dir:    directory of landmark .nii.gz label masks (label 1=LM1, 2=LM2)
        patient_ids: list of strings like ["patient001", ...]
        in_channels: 1 = MRI only, 2 = MRI + seg mask
        augment:     whether to apply data augmentation
        sigma:       Gaussian heatmap sigma (can be updated via set_sigma)
        target_size: (H, W) to resize all slices to
    """

    H = 256
    W = 256

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        rvip_dir: str,
        patient_ids: list,
        in_channels: int = 1,
        augment: bool = False,
        sigma: float = 8.0,
        target_size: tuple = (256, 256),
    ):
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.rvip_dir    = rvip_dir
        self.in_channels = in_channels
        self.augment     = augment
        self.sigma       = sigma
        self.H, self.W   = target_size

        self.slices: list[dict] = []
        self._build_index(patient_ids)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _build_index(self, patient_ids: list):
        n_missing = n_lowvar = n_close = 0

        for pid in patient_ids:
            for frame in ("ED", "ES"):
                img_path  = _find_image(self.image_dir, pid, frame)
                mask_path = _find_mask(self.mask_dir,  pid, frame)
                rvip_path = _find_rvip(self.rvip_dir,  pid, frame)

                if img_path is None or mask_path is None or rvip_path is None:
                    continue

                try:
                    img_vol  = nib.load(img_path).get_fdata(dtype=np.float32)
                    mask_vol = nib.load(mask_path).get_fdata(dtype=np.float32)
                    lm_map   = _load_rvip_nifti(rvip_path)
                except Exception as exc:
                    print(f"  [skip] {pid}/{frame}: {exc}")
                    continue

                # Squeeze time dimension if 4-D
                if img_vol.ndim == 4:
                    img_vol = img_vol[..., 0]
                if mask_vol.ndim == 4:
                    mask_vol = mask_vol[..., 0]

                n_slices = img_vol.shape[2]

                for z in range(n_slices):
                    if z not in lm_map:
                        n_missing += 1
                        continue

                    lm1, lm2 = lm_map[z]          # (x,y) in original voxel space
                    img_sl   = img_vol[:, :, z]

                    # Variance filter
                    if float(img_sl.var()) < 0.01:
                        n_lowvar += 1
                        continue

                    # Landmark separation filter — compute in 256×256 space
                    h_orig, w_orig = img_sl.shape
                    lm1s = _scale_coord(lm1, h_orig, w_orig, self.H, self.W)
                    lm2s = _scale_coord(lm2, h_orig, w_orig, self.H, self.W)
                    sep  = math.hypot(lm1s[0] - lm2s[0], lm1s[1] - lm2s[1])
                    if sep < 10.0:
                        n_close += 1
                        continue

                    self.slices.append({
                        "img_path":  img_path,
                        "mask_path": mask_path,
                        "z":         z,
                        "lm1":       lm1,
                        "lm2":       lm2,
                        "h_orig":    h_orig,
                        "w_orig":    w_orig,
                    })

        print(
            f"[ACDCLandmarkDataset] {len(self.slices)} slices "
            f"| skipped: {n_missing} missing, {n_lowvar} low-var, {n_close} close"
        )

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment(self, img, mask, lm1, lm2):
        """
        Apply geometric then intensity augmentation.
        img, mask: float32 numpy (H, W)
        lm1, lm2: (x, y) in 256×256 space
        Returns same types.
        """
        H, W = img.shape

        # Horizontal flip
        if random.random() < 0.5:
            img  = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            lm1  = (W - 1 - lm1[0], lm1[1])
            lm2  = (W - 1 - lm2[0], lm2[1])

        # Vertical flip
        if random.random() < 0.5:
            img  = img[::-1, :].copy()
            mask = mask[::-1, :].copy()
            lm1  = (lm1[0], H - 1 - lm1[1])
            lm2  = (lm2[0], H - 1 - lm2[1])

        # Rotation ±30°
        if random.random() < 0.5:
            angle   = random.uniform(-30.0, 30.0)
            img     = ndimage_rotate(img,  angle, reshape=False, order=1, mode="nearest")
            mask    = ndimage_rotate(mask, angle, reshape=False, order=0, mode="nearest")
            cx, cy  = W / 2.0, H / 2.0
            rad     = math.radians(-angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)

            def _rot(x, y):
                dx, dy = x - cx, y - cy
                nx = cos_a * dx - sin_a * dy + cx
                ny = sin_a * dx + cos_a * dy + cy
                return (float(np.clip(nx, 0, W - 1)), float(np.clip(ny, 0, H - 1)))

            lm1 = _rot(*lm1)
            lm2 = _rot(*lm2)

        # Intensity augmentation (image only)
        if random.random() < 0.4:       # Gamma
            gamma = random.uniform(0.7, 1.4)
            lo, hi = img.min(), img.max()
            rng = hi - lo
            if rng > 1e-6:
                img = ((img - lo) / rng) ** gamma * rng + lo

        if random.random() < 0.4:       # Gaussian noise
            img = img + np.random.normal(0, random.uniform(0.01, 0.05), img.shape).astype(np.float32)

        if random.random() < 0.4:       # Brightness shift
            img = img + random.uniform(-0.2, 0.2)

        # Re-standardise after intensity augmentation
        img = _hist_standardise(img)

        # Re-enforce LM2 = inferior (larger y) after flips
        if lm1[1] > lm2[1]:
            lm1, lm2 = lm2, lm1

        return img, mask, lm1, lm2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_sigma(self, sigma: float):
        self.sigma = sigma

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx: int):
        rec = self.slices[idx]

        # Load volumes (re-load each call; could cache if RAM allows)
        img_vol  = nib.load(rec["img_path"]).get_fdata(dtype=np.float32)
        mask_vol = nib.load(rec["mask_path"]).get_fdata(dtype=np.float32)
        if img_vol.ndim  == 4: img_vol  = img_vol[...,  0]
        if mask_vol.ndim == 4: mask_vol = mask_vol[..., 0]

        z = rec["z"]
        img_sl  = img_vol[:, :, z]
        mask_sl = mask_vol[:, :, z]
        h_orig, w_orig = rec["h_orig"], rec["w_orig"]

        # Resize to 256×256
        img_sl  = _resize2d(img_sl,  self.H, self.W, order=1)
        mask_sl = _resize2d(mask_sl, self.H, self.W, order=0)

        # Standardise
        img_sl = _hist_standardise(img_sl)

        # Scale coords to 256×256 space
        lm1 = _scale_coord(rec["lm1"], h_orig, w_orig, self.H, self.W)
        lm2 = _scale_coord(rec["lm2"], h_orig, w_orig, self.H, self.W)

        # Augmentation
        if self.augment:
            img_sl, mask_sl, lm1, lm2 = self._augment(img_sl, mask_sl, lm1, lm2)

        # Clamp to image bounds
        lm1 = (float(np.clip(lm1[0], 0, self.W - 1)), float(np.clip(lm1[1], 0, self.H - 1)))
        lm2 = (float(np.clip(lm2[0], 0, self.W - 1)), float(np.clip(lm2[1], 0, self.H - 1)))

        # Build image tensor
        img_t = torch.from_numpy(img_sl).unsqueeze(0)          # [1, H, W]
        if self.in_channels == 2:
            mask_t = torch.from_numpy(mask_sl / 3.0).unsqueeze(0)  # [1, H, W]
            image_out = torch.cat([img_t, mask_t], dim=0)           # [2, H, W]
        else:
            image_out = img_t                                        # [1, H, W]

        # Coords tensor [4]: x1, y1, x2, y2
        coords = torch.tensor([lm1[0], lm1[1], lm2[0], lm2[1]], dtype=torch.float32)

        # Heatmaps [2, H, W]  — note: coords_to_heatmaps takes image_shape=(H,W)
        heatmaps = coords_to_heatmaps(
            coords.numpy(),
            image_shape=(self.H, self.W),
            sigma=self.sigma,
        )
        heatmaps = torch.from_numpy(heatmaps)   # [2, H, W]

        return image_out, heatmaps, coords
