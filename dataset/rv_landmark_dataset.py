"""
RVLandmarkDataset
-----------------
Dataset for rv_landmark training data with **combined-heatmap GT**.

GT format (train_gt/*.nii.gz):
    single-channel volume with two Gaussian blobs per annotated slice.
    Connected-component analysis -> two centroids -> (LM1, LM2) sorted by y.

Optional 2-channel input: MRI stacked with multi-class seg [0,1,2,3]
from train_seg_multi/, normalised to [0,1] by max, with the "Fix 2"
fallback (zero the seg channel when RV label 1 is absent).

Returns (image[C,256,256], heatmap[2,256,256], coords[4]) per sample.

Geometric augmentations apply the SAME transform to image and seg
(seg uses INTER_NEAREST to keep labels categorical).
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import cv2
from scipy.ndimage import label, center_of_mass

from utils.heatmap import coords_to_heatmaps
from dataset.landmark_dataset import enforce_superior_ordering, _PatientNormCache


MODEL_INPUT_SIZE = 256
_NORM_CACHE = _PatientNormCache()


# ── 2-channel-aware geometric augmentations ───────────────────────────────────

def _aug_rotate(img, seg, coords, max_angle=30):
    H, W = img.shape
    angle = np.random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    img_out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    seg_out = None
    if seg is not None:
        seg_out = cv2.warpAffine(seg, M, (W, H), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    new_coords = coords.copy()
    for i in range(2):
        px, py = coords[2 * i], coords[2 * i + 1]
        nx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ny = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        new_coords[2 * i]     = np.clip(nx, 0, W - 1)
        new_coords[2 * i + 1] = np.clip(ny, 0, H - 1)
    return img_out, seg_out, new_coords


def _aug_crop_resize(img, seg, coords, crop_ratio=0.85):
    H, W = img.shape
    ch = int(H * crop_ratio)
    cw = int(W * crop_ratio)
    top  = np.random.randint(0, H - ch + 1)
    left = np.random.randint(0, W - cw + 1)
    img_c = img[top:top + ch, left:left + cw]
    img_out = cv2.resize(img_c, (W, H), interpolation=cv2.INTER_LINEAR)
    seg_out = None
    if seg is not None:
        seg_c = seg[top:top + ch, left:left + cw]
        seg_out = cv2.resize(seg_c, (W, H), interpolation=cv2.INTER_NEAREST)
    new_coords = coords.copy()
    new_coords[0] = np.clip((coords[0] - left) * W / cw, 0, W - 1)
    new_coords[1] = np.clip((coords[1] - top)  * H / ch, 0, H - 1)
    new_coords[2] = np.clip((coords[2] - left) * W / cw, 0, W - 1)
    new_coords[3] = np.clip((coords[3] - top)  * H / ch, 0, H - 1)
    return img_out, seg_out, new_coords


def _aug_elastic(img, seg, coords, alpha=15, sigma=4):
    H, W = img.shape
    dx = cv2.GaussianBlur(np.random.uniform(-1, 1, (H, W)).astype(np.float32),
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(np.random.uniform(-1, 1, (H, W)).astype(np.float32),
                          (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = np.clip(x + dx, 0, W - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, H - 1).astype(np.float32)
    img_out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    seg_out = None
    if seg is not None:
        seg_out = cv2.remap(seg, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    new_coords = coords.copy()
    for i in range(2):
        cx, cy = coords[2 * i], coords[2 * i + 1]
        ix = int(np.clip(round(cx), 0, W - 1))
        iy = int(np.clip(round(cy), 0, H - 1))
        new_coords[2 * i]     = np.clip(cx + dx[iy, ix], 0, W - 1)
        new_coords[2 * i + 1] = np.clip(cy + dy[iy, ix], 0, H - 1)
    return img_out, seg_out, new_coords


# ── blob extraction ───────────────────────────────────────────────────────────

def _extract_two_blobs(gt_slice, thresh_frac=0.1, min_area=3):
    """
    Find the two Gaussian-blob centroids in a combined-heatmap slice.
    Returns coords [x1, y1, x2, y2] sorted by y (smaller y = LM1), or None.
    """
    mx = gt_slice.max()
    if mx <= 0:
        return None
    bw = gt_slice > (thresh_frac * mx)
    lbl, n = label(bw)
    if n < 2:
        return None
    comps = []
    for k in range(1, n + 1):
        area = int((lbl == k).sum())
        if area < min_area:
            continue
        cy, cx = center_of_mass(gt_slice, lbl, k)
        comps.append((float(cx), float(cy), area))
    if len(comps) < 2:
        return None
    comps.sort(key=lambda t: -t[2])     # take 2 largest by area
    (x1, y1, _), (x2, y2, _) = comps[0], comps[1]
    if y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ── patient-level split ───────────────────────────────────────────────────────

def split_volumes(image_dir, val_frac=0.2, seed=42):
    """
    Patient-level deterministic 80/20 split of .nii.gz filenames in image_dir.
    Returns (train_files, val_files).
    """
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".nii.gz"))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(files))
    n_val = int(round(len(files) * val_frac))
    val_idx   = set(perm[:n_val].tolist())
    train = [files[i] for i in range(len(files)) if i not in val_idx]
    val   = [files[i] for i in range(len(files)) if i in val_idx]
    return train, val


# ── dataset ───────────────────────────────────────────────────────────────────

class RVLandmarkDataset(Dataset):
    """
    Args:
        image_dir       : path to train_images/ or test_images/
        gt_dir          : path to train_gt/      (combined heatmap)
        seg_dir         : path to train_seg_multi/  (optional; required if in_channels=2)
        in_channels     : 1 or 2
        slice_axis      : axis along which to slice (default 2)
        augment         : enable training augmentations
        sigma           : Gaussian heatmap sigma (set per-epoch via set_sigma)
        min_landmark_dist : reject slices where the two landmarks are closer than
                            this many pixels (measured at 256x256 scale)
        min_slice_variance: reject very flat slices
        volume_whitelist  : optional list of filenames; only those volumes are used
                            (use with split_volumes for patient-level train/val split)
        thresh_frac, min_area : connected-component thresholds for blob extraction
    """

    def __init__(
        self,
        image_dir,
        gt_dir,
        seg_dir=None,
        in_channels=2,
        slice_axis=2,
        augment=False,
        sigma=6,
        min_landmark_dist=20,
        min_slice_variance=0.01,
        volume_whitelist=None,
        thresh_frac=0.1,
        min_area=3,
    ):
        assert in_channels in (1, 2)
        if in_channels == 2 and seg_dir is None:
            raise ValueError("seg_dir is required when in_channels=2")

        self.image_dir = image_dir
        self.gt_dir    = gt_dir
        self.seg_dir   = seg_dir
        self.in_channels = in_channels
        self.slice_axis  = slice_axis
        self.augment     = augment
        self.sigma       = sigma
        self.min_landmark_dist  = min_landmark_dist
        self.min_slice_variance = min_slice_variance
        self.thresh_frac = thresh_frac
        self.min_area    = min_area

        if volume_whitelist is not None:
            self._whitelist = set(volume_whitelist)
        else:
            self._whitelist = None

        self.samples, stats = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found.")

        print(f"OK RVLandmarkDataset: {len(self.samples)} slices  "
              f"(annotated={stats['annotated']}, dropped_blobs={stats['drop_blobs']}, "
              f"dropped_dist={stats['drop_dist']}, dropped_var={stats['drop_var']}) "
              f"| in_channels={in_channels}")

    def set_sigma(self, sigma):
        self.sigma = sigma

    def _build_samples(self):
        samples = []
        stats = {"annotated": 0, "drop_blobs": 0, "drop_dist": 0, "drop_var": 0}

        files = sorted(f for f in os.listdir(self.image_dir) if f.endswith(".nii.gz"))
        for fname in files:
            if self._whitelist is not None and fname not in self._whitelist:
                continue

            img_path = os.path.join(self.image_dir, fname)
            gt_path  = os.path.join(self.gt_dir,    fname)
            if not os.path.exists(gt_path):
                continue
            if self.in_channels == 2:
                seg_path = os.path.join(self.seg_dir, fname)
                if not os.path.exists(seg_path):
                    continue

            img = nib.load(img_path).get_fdata()
            gt  = nib.load(gt_path).get_fdata()
            if img.shape[:3] != gt.shape[:3]:
                continue

            n_slices = min(img.shape[self.slice_axis], gt.shape[self.slice_axis])
            for i in range(n_slices):
                gt_2d = np.take(gt, i, axis=self.slice_axis)
                if gt_2d.max() <= 0:
                    continue
                stats["annotated"] += 1

                coords_orig = _extract_two_blobs(
                    gt_2d, thresh_frac=self.thresh_frac, min_area=self.min_area
                )
                if coords_orig is None:
                    stats["drop_blobs"] += 1
                    continue

                H_2d, W_2d = gt_2d.shape
                # rescale to 256-space to measure distance consistently
                dx = (coords_orig[2] - coords_orig[0]) * MODEL_INPUT_SIZE / max(W_2d, 1)
                dy = (coords_orig[3] - coords_orig[1]) * MODEL_INPUT_SIZE / max(H_2d, 1)
                dist = float(np.hypot(dx, dy))
                if dist < self.min_landmark_dist:
                    stats["drop_dist"] += 1
                    continue

                img_2d = np.take(img, i, axis=self.slice_axis).astype(np.float32)
                mu_s, std_s = img_2d.mean(), img_2d.std() + 1e-8
                if ((img_2d - mu_s) / std_s).var() < self.min_slice_variance:
                    stats["drop_var"] += 1
                    continue

                samples.append((fname, i, coords_orig, H_2d, W_2d))

        return samples, stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, slice_idx, coords_orig, H_orig, W_orig = self.samples[idx]

        img_path = os.path.join(self.image_dir, fname)
        img_vol  = nib.load(img_path).get_fdata().astype(np.float32)
        mu, std  = _NORM_CACHE.get(img_path, img_vol)

        img_2d = np.take(img_vol, slice_idx, axis=self.slice_axis)
        img_r  = cv2.resize(img_2d, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        img_r  = (img_r - mu) / std

        seg_r = None
        if self.in_channels == 2:
            seg_path = os.path.join(self.seg_dir, fname)
            seg_vol  = nib.load(seg_path).get_fdata().astype(np.float32)
            seg_2d   = np.take(seg_vol, slice_idx, axis=self.slice_axis)
            seg_2d   = np.round(seg_2d)
            seg_r    = cv2.resize(seg_2d, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                   interpolation=cv2.INTER_NEAREST)

        # scale coords to 256-space
        x1, y1, x2, y2 = coords_orig
        coords = np.array([
            x1 * MODEL_INPUT_SIZE / W_orig,
            y1 * MODEL_INPUT_SIZE / H_orig,
            x2 * MODEL_INPUT_SIZE / W_orig,
            y2 * MODEL_INPUT_SIZE / H_orig,
        ], dtype=np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                img_r = np.fliplr(img_r).copy()
                if seg_r is not None:
                    seg_r = np.fliplr(seg_r).copy()
                coords[0] = (MODEL_INPUT_SIZE - 1) - coords[0]
                coords[2] = (MODEL_INPUT_SIZE - 1) - coords[2]
            if np.random.rand() < 0.5:
                img_r = np.flipud(img_r).copy()
                if seg_r is not None:
                    seg_r = np.flipud(seg_r).copy()
                coords[1] = (MODEL_INPUT_SIZE - 1) - coords[1]
                coords[3] = (MODEL_INPUT_SIZE - 1) - coords[3]
            if np.random.rand() < 0.7:
                img_r, seg_r, coords = _aug_rotate(img_r, seg_r, coords, max_angle=30)
            if np.random.rand() < 0.5:
                img_r = img_r * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1)
            if np.random.rand() < 0.3:
                img_r = img_r + np.random.randn(*img_r.shape).astype(np.float32) * 0.05
            if np.random.rand() < 0.4:
                img_r, seg_r, coords = _aug_elastic(img_r, seg_r, coords, alpha=15, sigma=4)
            if np.random.rand() < 0.4:
                img_r, seg_r, coords = _aug_crop_resize(
                    img_r, seg_r, coords, crop_ratio=np.random.uniform(0.80, 0.95)
                )

        coords = enforce_superior_ordering(coords)
        coords = np.clip(coords, 0, MODEL_INPUT_SIZE - 1)

        # build input tensor
        if self.in_channels == 1:
            x = img_r[None, ...]
        else:
            # Fix 2: zero seg channel if RV (label 1) absent in this slice
            if seg_r is None or not np.any(np.round(seg_r) == 1):
                seg_norm = np.zeros_like(img_r, dtype=np.float32)
            else:
                smax = seg_r.max()
                seg_norm = (seg_r / smax).astype(np.float32) if smax > 0 else np.zeros_like(img_r)
            x = np.stack([img_r.astype(np.float32), seg_norm], axis=0)

        heatmaps = coords_to_heatmaps(coords, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                       sigma=self.sigma)

        return (
            torch.tensor(x,        dtype=torch.float32),
            torch.tensor(heatmaps, dtype=torch.float32),
            torch.tensor(coords,   dtype=torch.float32),
        )
