"""
pseudo_label_rv.py — Generate pseudo-labels from model predictions
===================================================================
Runs TTA inference on rv_landmark TRAIN images and saves confident
predictions as NIfTI GT files in the same format as train_gt/.

GT format written:
    Shape: (H, W, n_slices)
    Label 1 = LM1 pixel location (single pixel, rounded)
    Label 2 = LM2 pixel location (single pixel, rounded)
    Label 0 = background

Confidence filters (all must pass for a slice to be accepted):
    hm_max(LM1) > threshold
    hm_max(LM2) > threshold
    Euclidean distance(LM1, LM2) > min_dist  (in original pixel space)
    LM1 y < LM2 y  (superior ordering)

Usage:
    python pseudo_label_rv.py \\
        --checkpoint rv-checkpoints/.../best_model.pth \\
        --in-channels 2 \\
        --instance-norm \\
        --threshold 0.7 \\
        --min-dist 10
"""

import argparse
import json
import os

import cv2
import nibabel as nib
import numpy as np
import torch

from models.unet_resnet34 import UNetResNet34
from utils.postprocess import gaussian_subpixel_argmax

MODEL_INPUT_SIZE = 256

IMAGE_DIR = "data/rv_landmark/train_images"
SEG_DIR   = "data/rv_landmark/train_seg_multi"


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint, in_channels, device, use_instance_norm=False,
               use_group_norm=False):
    model = UNetResNet34(
        in_channels=in_channels, num_classes=2,
        dropout=0.0, pretrained=False, cardiac_pretrained=False,
        use_instance_norm=use_instance_norm,
        use_group_norm=use_group_norm,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ── preprocessing (mirrors inference_rv.py exactly) ───────────────────────────

def preprocess(img_2d, seg_2d, in_channels):
    img_r = cv2.resize(img_2d.astype(np.float32),
                       (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    mu, std = img_r.mean(), img_r.std() + 1e-8
    img_r   = (img_r - mu) / std

    if in_channels == 1:
        return torch.tensor(img_r, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    if seg_2d is None:
        seg_r = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.float32)
    else:
        seg_r = cv2.resize(seg_2d.astype(np.float32),
                           (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                           interpolation=cv2.INTER_NEAREST)
        if not np.any(np.round(seg_r) == 1):
            seg_r = np.zeros_like(seg_r)
        else:
            seg_max = seg_r.max()
            if seg_max > 0:
                seg_r = seg_r / seg_max

    two_ch = np.stack([img_r, seg_r], axis=0)
    return torch.tensor(two_ch, dtype=torch.float32).unsqueeze(0)


# ── TTA (4-variant, mirrors inference_rv.py) ──────────────────────────────────

@torch.no_grad()
def tta_predict(model, tensor, device):
    """
    Returns (avg_heatmap, max_hm1, max_hm2).
    avg_heatmap : [1, 2, 256, 256] — mean of 4 sigmoid outputs, used for coords.
    max_hm1/2   : peak of channel 0/1 taken as max-over-variants BEFORE averaging,
                  so the confidence score reflects the sharpest individual prediction
                  rather than the spatially-smeared average (averaging shifts mass
                  away from the peak, making the averaged peak artificially low).
    """
    x = tensor.to(device)
    variants = []
    peak1_list, peak2_list = [], []
    for hf in [False, True]:
        for vf in [False, True]:
            t = x.clone()
            if hf: t = torch.flip(t, [3])
            if vf: t = torch.flip(t, [2])
            p = torch.sigmoid(model(t))
            if vf: p = torch.flip(p, [2])
            if hf: p = torch.flip(p, [3])
            variants.append(p)
            # record per-variant peak (already in unflipped space)
            p_np = p[0].cpu().numpy()   # [2, 256, 256]
            peak1_list.append(float(p_np[0].max()))
            peak2_list.append(float(p_np[1].max()))
    avg_hm = torch.stack(variants).mean(0)   # [1, 2, 256, 256]
    return avg_hm, max(peak1_list), max(peak2_list)


# ── per-slice inference + confidence check ────────────────────────────────────

def predict_slice(img_2d, seg_2d, model, in_channels, device, use_tta):
    """
    Returns (coords_orig, hm_max1, hm_max2).
    coords_orig : (x1,y1,x2,y2) in original pixel space.
    hm_max1/2   : post-sigmoid peak for each landmark channel, range [0,1].
    """
    H_orig, W_orig = img_2d.shape
    tensor = preprocess(img_2d, seg_2d, in_channels)

    if use_tta:
        hm256, hm_max1, hm_max2 = tta_predict(model, tensor, device)
    else:
        with torch.no_grad():
            hm256 = torch.sigmoid(model(tensor.to(device)))
        hm_np   = hm256[0].cpu().numpy()   # [2, 256, 256]
        hm_max1 = float(hm_np[0].max())
        hm_max2 = float(hm_np[1].max())

    coords_256 = gaussian_subpixel_argmax(hm256, window=7)[0].cpu().numpy()

    coords_orig = np.array([
        coords_256[0] * W_orig / MODEL_INPUT_SIZE,
        coords_256[1] * H_orig / MODEL_INPUT_SIZE,
        coords_256[2] * W_orig / MODEL_INPUT_SIZE,
        coords_256[3] * H_orig / MODEL_INPUT_SIZE,
    ], dtype=np.float32)

    return coords_orig, hm_max1, hm_max2


def passes_confidence(coords, hm_max1, hm_max2, threshold, min_dist):
    """All four confidence gates must pass."""
    if hm_max1 <= threshold:
        return False, f"hm1_max={hm_max1:.3f} <= {threshold}"
    if hm_max2 <= threshold:
        return False, f"hm2_max={hm_max2:.3f} <= {threshold}"
    x1, y1, x2, y2 = coords
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist <= min_dist:
        return False, f"dist={dist:.1f}px <= {min_dist}"
    if y1 >= y2:
        return False, f"ordering fail: y1={y1:.1f} >= y2={y2:.1f}"
    return True, "ok"


# ── NIfTI writer ──────────────────────────────────────────────────────────────

def save_pseudo_gt(accepted_slices, H, W, n_slices_vol, out_path, ref_affine):
    """
    accepted_slices: dict {slice_idx: coords_orig (x1,y1,x2,y2)}
    Writes a (H, W, n_slices_vol) int16 volume: label 1 = LM1, label 2 = LM2.
    """
    vol = np.zeros((H, W, n_slices_vol), dtype=np.int16)
    for sl_idx, coords in accepted_slices.items():
        x1, y1, x2, y2 = coords
        r1, c1 = int(round(y1)), int(round(x1))
        r2, c2 = int(round(y2)), int(round(x2))
        r1 = int(np.clip(r1, 0, H - 1))
        c1 = int(np.clip(c1, 0, W - 1))
        r2 = int(np.clip(r2, 0, H - 1))
        c2 = int(np.clip(c2, 0, W - 1))
        vol[r1, c1, sl_idx] = 1
        vol[r2, c2, sl_idx] = 2

    img_nib = nib.Nifti1Image(vol, affine=ref_affine)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    nib.save(img_nib, out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for rv_landmark train set"
    )
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--in-channels",   type=int,   default=2, choices=[1, 2])
    parser.add_argument("--instance-norm", action="store_true")
    parser.add_argument("--group-norm",    action="store_true",
                        help="Use GroupNorm instead of BatchNorm")
    parser.add_argument("--threshold",     type=float, default=0.7,
                        help="Min heatmap peak confidence for both landmarks")
    parser.add_argument("--min-dist",      type=float, default=10.0,
                        help="Min Euclidean distance (px, original space) between LM1 and LM2")
    parser.add_argument("--out-dir",       default="data/rv_landmark/pseudo_gt")
    parser.add_argument("--no-tta",        action="store_true")
    args = parser.parse_args()
    if args.instance_norm and args.group_norm:
        raise ValueError("Cannot use both --instance-norm and --group-norm")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = not args.no_tta

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Channels   : {args.in_channels}")
    print(f"Threshold  : {args.threshold}  min_dist: {args.min_dist}px")
    print(f"TTA        : {use_tta}")
    print(f"Output dir : {args.out_dir}")
    print()

    model = load_model(args.checkpoint, args.in_channels, device,
                       use_instance_norm=args.instance_norm,
                       use_group_norm=args.group_norm)

    seg_dir = SEG_DIR if args.in_channels == 2 else None

    files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".nii.gz"))
    if not files:
        raise RuntimeError(f"No .nii.gz files found in {IMAGE_DIR}")

    total_slices   = 0
    total_accepted = 0
    per_file_log   = {}

    for fname in files:
        img_path = os.path.join(IMAGE_DIR, fname)
        img_vol  = nib.load(img_path)
        img_arr  = img_vol.get_fdata().astype(np.float32)   # (H, W, S)
        affine   = img_vol.affine
        H, W, n_slices = img_arr.shape

        seg_arr = None
        if seg_dir is not None:
            seg_path = os.path.join(seg_dir, fname)
            if os.path.exists(seg_path):
                seg_arr = np.round(
                    nib.load(seg_path).get_fdata().astype(np.float32)
                )

        accepted_slices = {}
        rejected_slices = []

        for sl in range(n_slices):
            img_2d = img_arr[:, :, sl]
            seg_2d = seg_arr[:, :, sl] if seg_arr is not None else None

            # Skip blank slices (same variance filter as RVLandmarkDataset)
            mu_s, std_s = img_2d.mean(), img_2d.std() + 1e-8
            if ((img_2d - mu_s) / std_s).var() < 0.01:
                rejected_slices.append(sl)
                total_slices += 1
                continue

            coords, hm_max1, hm_max2 = predict_slice(
                img_2d, seg_2d, model, args.in_channels, device, use_tta
            )

            ok, reason = passes_confidence(
                coords, hm_max1, hm_max2, args.threshold, args.min_dist
            )
            total_slices += 1

            if ok:
                accepted_slices[sl] = coords
                total_accepted += 1
            else:
                rejected_slices.append({"slice": sl, "reason": reason})

        n_acc = len(accepted_slices)
        n_rej = len(rejected_slices)
        print(f"  {fname}: {n_acc}/{n_slices} slices accepted")

        per_file_log[fname] = {
            "accepted": sorted(accepted_slices.keys()),
            "rejected": sorted(rejected_slices, key=lambda r: r["slice"]
                               if isinstance(r, dict) else r),
        }

        if n_acc > 0:
            out_path = os.path.join(args.out_dir, fname)
            save_pseudo_gt(accepted_slices, H, W, n_slices, out_path, affine)

    acceptance_rate = total_accepted / max(total_slices, 1)
    total_rejected  = total_slices - total_accepted

    print()
    print(f"Total accepted : {total_accepted}/{total_slices} slices "
          f"({acceptance_rate:.1%})")
    print(f"Total rejected : {total_rejected} slices")
    print(f"Pseudo GT written to: {args.out_dir}/")

    if total_accepted < 50:
        print()
        print("WARNING: fewer than 50 slices accepted.")
        print("Consider re-running with --threshold 0.6")

    log = {
        "checkpoint":       args.checkpoint,
        "threshold":        args.threshold,
        "min_dist":         args.min_dist,
        "use_tta":          use_tta,
        "in_channels":      args.in_channels,
        "total_slices":     total_slices,
        "accepted_slices":  total_accepted,
        "acceptance_rate":  round(acceptance_rate, 4),
        "per_file":         per_file_log,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "pseudo_label_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log written to : {log_path}")


if __name__ == "__main__":
    main()
