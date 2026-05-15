"""
inference_rv.py — Evaluate on rv_landmark test set
====================================================
Works for both 1-channel and 2-channel models.
For 2-channel model without seg masks: second channel is automatically
zeroed (Fix 2 fallback) — model uses MRI-only mode for those slices.

rv_landmark folder structure:
    data/rv_landmark/test_images/    DET0000301.nii.gz ...  (MRI volumes)
    data/rv_landmark/test_gt/        DET0000301.nii.gz ...  (GT masks, label 1 and 2)
    data/rv_landmark/test_seg/       DET0000301.nii.gz ...  (optional seg masks)

GT mask format:
    Shape: (H, W, n_slices)
    Label 0 = background
    Label 1 = upper/anterior RV insertion point  (1 pixel)
    Label 2 = lower/inferior RV insertion point  (1 pixel)

Usage — 1-channel model:
    python inference_rv.py --checkpoint checkpoints/acdc_1ch_.../best_p2.pth --in-channels 1 --eval

Usage — 2-channel model without seg masks:
    python inference_rv.py --checkpoint checkpoints/acdc_2ch_.../best_p2.pth --in-channels 2 --eval

Usage — 2-channel model WITH seg masks:
    python inference_rv.py --checkpoint checkpoints/acdc_2ch_.../best_p2.pth --in-channels 2
                           --seg-dir data/rv_landmark/test_seg --eval

Usage — single image:
    python inference_rv.py --checkpoint checkpoints/acdc_1ch_.../best_p2.pth --in-channels 1
                           --image data/rv_landmark/test_images/DET0000301.nii.gz --auto
"""

import argparse
import json
import os
import numpy as np
import torch
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

from models.unet_resnet34 import UNetResNet34
from utils.postprocess import gaussian_subpixel_argmax

MODEL_INPUT_SIZE = 256

# ── default test paths ────────────────────────────────────────────────────────
TEST_IMAGE_DIR = "data/rv_landmark/test_images"
TEST_GT_DIR    = "data/rv_landmark/test_gt"      # testing_nifti_GT folder


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint, in_channels, device):
    model = UNetResNet34(
        in_channels=in_channels, num_classes=2,
        dropout=0.0, pretrained=False, cardiac_pretrained=False
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded  : {checkpoint}")
    print(f"Device  : {device}")
    print(f"Channels: {in_channels}")
    return model


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_2d, seg_2d, in_channels):
    """
    Prepare input tensor. seg_2d can be None or zeros for 1-channel model.
    Fix 2: zeros the seg channel when RV (label 1) is absent.
    """
    img_r = cv2.resize(img_2d.astype(np.float32), (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    mu, std = img_r.mean(), img_r.std() + 1e-8
    img_r   = (img_r - mu) / std

    if in_channels == 1:
        tensor = torch.tensor(img_r, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        if seg_2d is None:
            seg_r = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.float32)
        else:
            seg_r = cv2.resize(seg_2d.astype(np.float32),
                               (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                               interpolation=cv2.INTER_NEAREST)
            # Fix 2 — blank when no RV
            if not np.any(np.round(seg_r) == 1):
                seg_r = np.zeros_like(seg_r)
            else:
                seg_max = seg_r.max()
                if seg_max > 0:
                    seg_r = seg_r / seg_max

        two_ch = np.stack([img_r, seg_r], axis=0)
        tensor = torch.tensor(two_ch, dtype=torch.float32).unsqueeze(0)

    return tensor


# ── TTA ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_predict(model, tensor, device):
    x = tensor.to(device)
    variants = []
    for hf in [False, True]:
        for vf in [False, True]:
            t = x.clone()
            if hf: t = torch.flip(t, [3])
            if vf: t = torch.flip(t, [2])
            p = torch.sigmoid(model(t))
            if vf: p = torch.flip(p, [2])
            if hf: p = torch.flip(p, [3])
            variants.append(p)
    return torch.stack(variants).mean(0)


# ── GT extraction from testing_nifti_GT mask ─────────────────────────────────

def extract_gt_coords(gt_vol, slice_idx):
    """
    Extract GT (x1,y1,x2,y2) from testing_nifti_GT mask volume.

    Format: (H, W, n_slices) with single-pixel annotations
        Label 1 = upper/anterior RV insertion point
        Label 2 = lower/inferior RV insertion point

    Returns coords in original pixel space, or None if no annotation.
    """
    if slice_idx >= gt_vol.shape[2]:
        return None

    slc  = gt_vol[:, :, slice_idx]
    pts1 = np.argwhere(slc == 1)
    pts2 = np.argwhere(slc == 2)

    if len(pts1) == 0 or len(pts2) == 0:
        return None

    # Single pixel — centroid is just the pixel itself
    p1 = pts1.mean(axis=0)   # (row, col)
    p2 = pts2.mean(axis=0)

    # LM1 = superior (smaller row index)
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    return np.array([p1[1], p1[0], p2[1], p2[0]], dtype=np.float32)


# ── landmark ordering helpers ─────────────────────────────────────────────────

def enforce_ordering(pred):
    x1, y1, x2, y2 = pred
    if y1 > y2:
        return np.array([x2, y2, x1, y1], dtype=np.float32)
    return pred


def match_landmarks(pred, gt):
    e_normal  = np.linalg.norm(pred[:2]-gt[:2]) + np.linalg.norm(pred[2:]-gt[2:])
    e_swapped = np.linalg.norm(pred[2:]-gt[:2]) + np.linalg.norm(pred[:2]-gt[2:])
    if e_swapped < e_normal:
        return np.array([pred[2], pred[3], pred[0], pred[1]], dtype=np.float32)
    return pred


# ── core inference ────────────────────────────────────────────────────────────

def predict(img_2d, seg_2d, model, in_channels, device, use_tta=True):
    H_orig, W_orig = img_2d.shape
    tensor = preprocess(img_2d, seg_2d, in_channels)

    if use_tta:
        hm256 = tta_predict(model, tensor, device)
    else:
        with torch.no_grad():
            hm256 = torch.sigmoid(model(tensor.to(device)))

    coords_256 = gaussian_subpixel_argmax(hm256, window=7)[0].cpu().numpy()

    coords_orig = np.array([
        coords_256[0] * W_orig / MODEL_INPUT_SIZE,
        coords_256[1] * H_orig / MODEL_INPUT_SIZE,
        coords_256[2] * W_orig / MODEL_INPUT_SIZE,
        coords_256[3] * H_orig / MODEL_INPUT_SIZE,
    ], dtype=np.float32)

    hm_np = hm256[0].cpu().numpy()
    heatmap = np.stack([
        cv2.resize(hm_np[c], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        for c in range(2)
    ])

    return coords_orig, heatmap


def compute_mre(pred, gt):
    e1 = np.linalg.norm(pred[:2] - gt[:2])
    e2 = np.linalg.norm(pred[2:] - gt[2:])
    return 0.5 * (e1 + e2), e1, e2


# ── visualisation ─────────────────────────────────────────────────────────────

def visualize(img_2d, seg_2d, coords, heatmap, gt_coords,
              slice_idx, mre_info, save_path, in_channels):
    has_gt  = gt_coords is not None
    has_seg = seg_2d is not None and in_channels == 2

    n_panels = 4 if has_seg else 3
    fig = plt.figure(figsize=(5 * n_panels, 5))
    gs  = gridspec.GridSpec(1, n_panels)

    title = f"slice {slice_idx}"
    if mre_info:
        mre, e1, e2 = mre_info
        title += f"  |  MRE={mre:.2f}px  (LM1={e1:.1f}  LM2={e2:.1f}px)"

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_2d, cmap="gray", aspect="auto")
    if has_gt:
        ax0.scatter([gt_coords[0], gt_coords[2]], [gt_coords[1], gt_coords[3]],
                    c="lime", s=60, zorder=4, edgecolors="black",
                    linewidths=0.6, label="GT")
        ax0.plot([gt_coords[0], gt_coords[2]], [gt_coords[1], gt_coords[3]],
                 color="lime", lw=0.8, alpha=0.5, linestyle="--")
    ax0.scatter([coords[0]], [coords[1]], c="red", s=60, zorder=5,
                edgecolors="white", linewidths=0.8, label="Pred LM1")
    ax0.scatter([coords[2]], [coords[3]], c="deepskyblue", s=60, zorder=5,
                edgecolors="white", linewidths=0.8, label="Pred LM2")
    if has_gt:
        ax0.plot([coords[0], gt_coords[0]], [coords[1], gt_coords[1]],
                 color="yellow", lw=1.0, alpha=0.8, zorder=4)
        ax0.plot([coords[2], gt_coords[2]], [coords[3], gt_coords[3]],
                 color="yellow", lw=1.0, alpha=0.8, zorder=4)
    ax0.set_title(title, fontsize=8)
    ax0.legend(loc="lower right", fontsize=6)
    ax0.axis("off")

    panel = 1
    if has_seg:
        ax_seg = fig.add_subplot(gs[panel])
        ax_seg.imshow(img_2d, cmap="gray", aspect="auto")
        ax_seg.imshow(seg_2d, alpha=0.35, cmap="nipy_spectral", vmin=0, vmax=3)
        if has_gt:
            ax_seg.scatter([gt_coords[0], gt_coords[2]],
                           [gt_coords[1], gt_coords[3]],
                           c="lime", s=60, zorder=4, edgecolors="black", linewidths=0.6)
        ax_seg.scatter([coords[0]], [coords[1]], c="red", s=60, zorder=5,
                       edgecolors="white", linewidths=0.8)
        ax_seg.scatter([coords[2]], [coords[3]], c="deepskyblue", s=60, zorder=5,
                       edgecolors="white", linewidths=0.8)
        ax_seg.set_title("MRI + Seg overlay", fontsize=8)
        ax_seg.axis("off")
        panel += 1

    ax1 = fig.add_subplot(gs[panel])
    ax1.imshow(img_2d, cmap="gray", aspect="auto")
    ax1.imshow(heatmap[0], cmap="hot", alpha=0.5,
               vmin=0, vmax=max(heatmap[0].max(), 1e-6))
    ax1.scatter([coords[0]], [coords[1]], c="red", s=60, zorder=4,
                edgecolors="white", linewidths=0.8)
    if has_gt:
        ax1.scatter([gt_coords[0]], [gt_coords[1]], c="lime", s=60,
                    zorder=4, edgecolors="black", linewidths=0.6)
    ax1.set_title("LM1 heatmap", fontsize=8)
    ax1.axis("off")
    panel += 1

    ax2 = fig.add_subplot(gs[panel])
    ax2.imshow(img_2d, cmap="gray", aspect="auto")
    ax2.imshow(heatmap[1], cmap="hot", alpha=0.5,
               vmin=0, vmax=max(heatmap[1].max(), 1e-6))
    ax2.scatter([coords[2]], [coords[3]], c="deepskyblue", s=60, zorder=4,
                edgecolors="white", linewidths=0.8)
    if has_gt:
        ax2.scatter([gt_coords[2]], [gt_coords[3]], c="lime", s=60,
                    zorder=4, edgecolors="black", linewidths=0.6)
    ax2.set_title("LM2 heatmap", fontsize=8)
    ax2.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── full test set evaluation ──────────────────────────────────────────────────

def evaluate_test_set(model, in_channels, device, seg_dir,
                      out_dir, use_tta=True, checkpoint_path=None):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(TEST_IMAGE_DIR)
                    if f.endswith(".nii.gz")])

    all_mres, all_e1, all_e2 = [], [], []

    print(f"\n{'-'*75}")
    print(f"Test set   : {TEST_IMAGE_DIR}")
    print(f"Seg masks  : {seg_dir if seg_dir else 'None (blank channel / MRI-only mode)'}")
    print(f"Channels   : {in_channels}")
    print(f"\n{'File':<30} {'Slice':>5}  {'MRE':>8}  {'LM1':>8}  {'LM2':>8}")
    print(f"{'-'*75}")

    for fname in files:
        img_path   = os.path.join(TEST_IMAGE_DIR, fname)
        gt_path = os.path.join(TEST_GT_DIR, fname)
        seg_path   = os.path.join(seg_dir, fname) if seg_dir else None

        if not os.path.exists(gt_path):
            print(f"  SKIP (no points): {fname}")
            continue

        img   = nib.load(img_path).get_fdata().astype(np.float32)
        gt_vol = nib.load(gt_path).get_fdata().astype(np.float32)
        seg   = np.round(nib.load(seg_path).get_fdata()).astype(np.float32) \
                if seg_path and os.path.exists(seg_path) else None

        n_slices = min(img.shape[2], gt_vol.shape[2])

        for i in range(n_slices):
            gt = extract_gt_coords(gt_vol, i)
            if gt is None:
                continue

            img_2d = img[:, :, i]
            seg_2d = seg[:, :, i] if seg is not None else None

            coords, heatmap = predict(img_2d, seg_2d, model,
                                      in_channels, device, use_tta)

            coords = enforce_ordering(coords)
            coords = match_landmarks(coords, gt)

            mre, e1, e2 = compute_mre(coords, gt)
            all_mres.append(mre)
            all_e1.append(e1)
            all_e2.append(e2)

            print(f"  {fname:<28} {i:>5}  {mre:>8.2f}px  {e1:>7.1f}px  {e2:>7.1f}px")

            save_path = os.path.join(out_dir,
                                     f"{fname.replace('.nii.gz','')}_slice{i:03d}.png")
            visualize(img_2d, seg_2d, coords, heatmap, gt,
                      slice_idx=i, mre_info=(mre, e1, e2),
                      save_path=save_path, in_channels=in_channels)

    print(f"{'-'*75}")

    if all_mres:
        arr  = np.array(all_mres)
        arr1 = np.array(all_e1)
        arr2 = np.array(all_e2)

        print(f"\n{'='*75}")
        print(f"  SUMMARY — {len(arr)} annotated slices")
        print(f"  Seg masks : {seg_dir if seg_dir else 'None (blank channel)'}")
        print(f"{'='*75}")

        print(f"\n  Mean Radial Error (MRE — average of LM1 and LM2):")
        print(f"    Mean MRE  : {arr.mean():.2f}px")
        print(f"    Std  MRE  : {arr.std():.2f}px")
        print(f"    P50  MRE  : {np.percentile(arr, 50):.2f}px")
        print(f"    P90  MRE  : {np.percentile(arr, 90):.2f}px")
        print(f"    Max  MRE  : {arr.max():.2f}px")

        print(f"\n  Euclidean Distance per Landmark (px):")
        print(f"    {'Metric':<20} {'LM1 (anterior)':>16} {'LM2 (inferior)':>16} {'Both (mean)':>12}")
        print(f"    {'-'*66}")
        print(f"    {'Mean':.<20} {arr1.mean():>15.2f}px {arr2.mean():>15.2f}px {arr.mean():>11.2f}px")
        print(f"    {'Std':.<20} {arr1.std():>15.2f}px {arr2.std():>15.2f}px {arr.std():>11.2f}px")
        print(f"    {'Median (P50)':.<20} {np.percentile(arr1,50):>15.2f}px {np.percentile(arr2,50):>15.2f}px {np.percentile(arr,50):>11.2f}px")
        print(f"    {'P90':.<20} {np.percentile(arr1,90):>15.2f}px {np.percentile(arr2,90):>15.2f}px {np.percentile(arr,90):>11.2f}px")
        print(f"    {'Max':.<20} {arr1.max():>15.2f}px {arr2.max():>15.2f}px {arr.max():>11.2f}px")

        print(f"\n  Success Detection Rate (SDR) — both landmarks:")
        print(f"    SDR@2px   : {(arr < 2.0).mean():.1%}  ({int((arr < 2.0).sum())}/{len(arr)} slices)")
        print(f"    SDR@3px   : {(arr < 3.0).mean():.1%}  ({int((arr < 3.0).sum())}/{len(arr)} slices)")
        print(f"    SDR@5px   : {(arr < 5.0).mean():.1%}  ({int((arr < 5.0).sum())}/{len(arr)} slices)")
        print(f"    SDR@10px  : {(arr < 10.0).mean():.1%}  ({int((arr < 10.0).sum())}/{len(arr)} slices)")

        print(f"\n  SDR per Landmark:")
        print(f"    {'Threshold':<12} {'LM1':>10} {'LM2':>10}")
        for t in [2.0, 3.0, 5.0, 10.0]:
            s1 = (arr1 < t).mean()
            s2 = (arr2 < t).mean()
            print(f"    SDR@{int(t)}px      {s1:>9.1%} {s2:>10.1%}")

        print(f"\nResults → {out_dir}/")

        results = {
            "checkpoint":   checkpoint_path,
            "in_channels":  in_channels,
            "seg_dir":      seg_dir,
            "use_tta":      use_tta,
            "n_slices":     int(len(arr)),
            "mean_mre":     float(arr.mean()),
            "std_mre":      float(arr.std()),
            "p50_mre":      float(np.percentile(arr, 50)),
            "p90_mre":      float(np.percentile(arr, 90)),
            "max_mre":      float(arr.max()),
            "sdr_2":        float((arr < 2.0).mean()),
            "sdr_3":        float((arr < 3.0).mean()),
            "sdr_5":        float((arr < 5.0).mean()),
            "sdr_10":       float((arr < 10.0).mean()),
            "per_landmark": {
                "lm1": {
                    "mean": float(arr1.mean()), "std": float(arr1.std()),
                    "p50":  float(np.percentile(arr1, 50)),
                    "p90":  float(np.percentile(arr1, 90)),
                    "max":  float(arr1.max()),
                    "sdr_2":  float((arr1 < 2.0).mean()),
                    "sdr_5":  float((arr1 < 5.0).mean()),
                    "sdr_10": float((arr1 < 10.0).mean()),
                },
                "lm2": {
                    "mean": float(arr2.mean()), "std": float(arr2.std()),
                    "p50":  float(np.percentile(arr2, 50)),
                    "p90":  float(np.percentile(arr2, 90)),
                    "max":  float(arr2.max()),
                    "sdr_2":  float((arr2 < 2.0).mean()),
                    "sdr_5":  float((arr2 < 5.0).mean()),
                    "sdr_10": float((arr2 < 10.0).mean()),
                },
            },
        }
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Summary written → {os.path.join(out_dir, 'results.json')}")

    return np.array(all_mres)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate landmark detection on rv_landmark test set"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 2],
                        help="1 = MRI only, 2 = MRI + seg mask")
    parser.add_argument("--seg-dir", default=None,
                        help="Path to seg masks (optional, 2ch model only). "
                             "If not provided, second channel is zeroed.")
    parser.add_argument("--image", default=None,
                        help="Single image path for quick testing")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-select annotated slices for single image mode")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate full rv_landmark test set")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--out", default="inference_rv_results")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(args.checkpoint, args.in_channels, device)
    use_tta = not args.no_tta

    ckpt_run = os.path.basename(os.path.dirname(args.checkpoint))
    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ch_label = f"{args.in_channels}ch"
    out_dir  = os.path.join(args.out, f"{ckpt_run}_{ch_label}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output → {out_dir}")

    if args.eval:
        evaluate_test_set(model, args.in_channels, device,
                          seg_dir=args.seg_dir,
                          out_dir=out_dir, use_tta=use_tta,
                          checkpoint_path=args.checkpoint)

    elif args.image:
        fname      = os.path.basename(args.image)
        base       = fname.replace(".nii.gz", "")
        gt_path = os.path.join(TEST_GT_DIR, fname)
        seg_path   = os.path.join(args.seg_dir, fname) \
                     if args.seg_dir and os.path.exists(
                         os.path.join(args.seg_dir, fname)) else None

        img    = nib.load(args.image).get_fdata().astype(np.float32)
        gt_vol = nib.load(gt_path).get_fdata().astype(np.float32) \
                 if os.path.exists(gt_path) else None
        seg    = np.round(nib.load(seg_path).get_fdata()).astype(np.float32) \
                 if seg_path else None

        print(f"Volume : {img.shape}")
        print(f"GT     : {'found' if gt_vol is not None else 'not found'}")
        print(f"Seg    : {'found' if seg is not None else 'not found — blank channel'}")

        if args.auto and gt_vol is not None:
            # Select slices that have both landmark annotations
            slices = [i for i in range(min(img.shape[2], gt_vol.shape[2]))
                      if np.any(gt_vol[:,:,i] == 1) and np.any(gt_vol[:,:,i] == 2)]
        else:
            variances = [img[:,:,i].var() for i in range(img.shape[2])]
            slices = [int(np.argmax(variances))]

        all_mres = []
        for sl in slices:
            img_2d = img[:,:,sl]
            seg_2d = seg[:,:,sl] if seg is not None else None
            coords, heatmap = predict(img_2d, seg_2d, model,
                                      args.in_channels, device, use_tta)
            coords = enforce_ordering(coords)
            gt = extract_gt_coords(gt_vol, sl) if gt_vol is not None else None
            if gt is not None:
                coords = match_landmarks(coords, gt)
                mre, e1, e2 = compute_mre(coords, gt)
                all_mres.append(mre)
                print(f"  Slice {sl}: MRE={mre:.2f}px  LM1={e1:.1f}px  LM2={e2:.1f}px")
            save_path = os.path.join(out_dir, f"{base}_slice{sl:03d}.png")
            visualize(img_2d, seg_2d, coords, heatmap, gt,
                      sl, (mre,e1,e2) if gt is not None else None,
                      save_path, args.in_channels)

        if len(all_mres) > 1:
            arr = np.array(all_mres)
            print(f"\nMean MRE={arr.mean():.2f}px  SDR@5px={(arr<5).mean():.1%}")
        print(f"\nDone → {out_dir}/")

    else:
        print("Use --eval for full test set or --image for single volume")