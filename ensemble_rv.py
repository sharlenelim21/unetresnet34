"""
ensemble_rv.py — Ensemble inference on rv_landmark test set
============================================================
Loads 3 fine-tuned checkpoints simultaneously and averages their heatmaps
before extracting coordinates via gaussian_subpixel_argmax.

Fixed model configuration:
  Model 1: 1ch fine-tune          (finetune_rv_1ch_...)
  Model 2: InsNorm 2ch fine-tune  (finetune_rv_instnorm_...)
  Model 3: Mixed 2ch fine-tune    (finetune_mixed_2ch_...)

Ensemble:
  hm_ensemble = (hm1 + hm2 + hm3) / 3.0
  coords = gaussian_subpixel_argmax(hm_ensemble, window=7)

Usage:
    python ensemble_rv.py
    python ensemble_rv.py --no-tta
    python ensemble_rv.py --out my_results_dir
"""

import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
import torch
from datetime import datetime

from models.unet_resnet34 import UNetResNet34
from utils.postprocess import gaussian_subpixel_argmax

# ── hardcoded ensemble members ────────────────────────────────────────────────

ENSEMBLE = [
    {
        "tag": "combo_ft",
        "checkpoint": "rv-checkpoints/finetune_rv_instnorm_2026-05-17_17-11-24/best_model.pth",
        "in_channels": 2,
        "instance_norm": True,
    },
    {
        "tag": "mixed_ft",
        "checkpoint": "rv-checkpoints/finetune_mixed_2ch_2026-05-17_18-30-17/best_model.pth",
        "in_channels": 2,
        "instance_norm": True,
    },
    {
        "tag": "filtered_ft",
        "checkpoint": "rv-checkpoints/finetune_rv_2ch_2026-05-17_14-36-24/best_model.pth",
        "in_channels": 2,
        "instance_norm": False,
    },
]

# ── data paths ────────────────────────────────────────────────────────────────

MODEL_INPUT_SIZE = 256
TEST_IMAGE_DIR   = "data/rv_landmark/test_images"
TEST_GT_DIR      = "data/rv_landmark/test_gt"
SEG_DIR          = "data/rv_landmark/test_seg_multi"


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(cfg, device):
    model = UNetResNet34(
        in_channels=cfg["in_channels"], num_classes=2,
        dropout=0.0, pretrained=False, cardiac_pretrained=False,
        use_instance_norm=cfg.get("instance_norm", False),
        use_group_norm=cfg.get("group_norm", False),
    ).to(device)
    state = torch.load(cfg["checkpoint"], map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [{cfg['tag']}] warn: {len(missing)} missing keys")
    if unexpected:
        print(f"  [{cfg['tag']}] warn: {len(unexpected)} unexpected keys")
    model.eval()
    return model


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_2d, seg_2d, in_channels):
    """
    Prepare a [1, C, 256, 256] tensor for one model.
    seg_2d can be None — second channel is zeroed in that case (Fix 2 fallback).
    """
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


# ── TTA for a single model ────────────────────────────────────────────────────

@torch.no_grad()
def tta_predict_single(model, tensor, device):
    """Returns averaged heatmap [1, 2, 256, 256] over 4 TTA variants."""
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
    return torch.stack(variants).mean(0)   # [1, 2, H, W]


# ── GT extraction ─────────────────────────────────────────────────────────────

def extract_gt_coords(gt_vol, slice_idx):
    """
    Extract GT (x1,y1,x2,y2) from testing_nifti_GT mask volume.
    Label 1 = superior, Label 2 = inferior RV insertion point.
    Returns coords in original voxel space, or None if absent.
    """
    if slice_idx >= gt_vol.shape[2]:
        return None
    slc  = gt_vol[:, :, slice_idx]
    pts1 = np.argwhere(slc == 1)
    pts2 = np.argwhere(slc == 2)
    if len(pts1) == 0 or len(pts2) == 0:
        return None
    p1 = pts1.mean(axis=0)   # (row, col)
    p2 = pts2.mean(axis=0)
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


def compute_mre(pred, gt):
    e1 = np.linalg.norm(pred[:2] - gt[:2])
    e2 = np.linalg.norm(pred[2:] - gt[2:])
    return 0.5 * (e1 + e2), e1, e2


# ── ensemble inference for one slice ─────────────────────────────────────────

def ensemble_predict(models_and_cfgs, img_2d, seg_2d, device, use_tta=True):
    """
    Run all models on one slice and return the coordinate decoded from the
    averaged heatmap.

    models_and_cfgs : list of (model, cfg) pairs
    seg_2d          : seg mask for 2ch models; 1ch models receive None
    Returns (coords_orig, hm_ensemble_np) where coords are in original pixel space.
    """
    H_orig, W_orig = img_2d.shape
    hm_sum = None

    for model, cfg in models_and_cfgs:
        # 1ch model never uses seg; 2ch model uses seg_2d (may be None)
        seg_in = seg_2d if cfg["in_channels"] == 2 else None
        tensor = preprocess(img_2d, seg_in, cfg["in_channels"])

        if use_tta:
            hm = tta_predict_single(model, tensor, device)
        else:
            with torch.no_grad():
                hm = torch.sigmoid(model(tensor.to(device)))

        if hm_sum is None:
            hm_sum = hm
        else:
            hm_sum = hm_sum + hm

    hm_ensemble = hm_sum / len(models_and_cfgs)   # [1, 2, 256, 256]

    coords_256 = gaussian_subpixel_argmax(hm_ensemble, window=7)[0].cpu().numpy()
    coords_orig = np.array([
        coords_256[0] * W_orig / MODEL_INPUT_SIZE,
        coords_256[1] * H_orig / MODEL_INPUT_SIZE,
        coords_256[2] * W_orig / MODEL_INPUT_SIZE,
        coords_256[3] * H_orig / MODEL_INPUT_SIZE,
    ], dtype=np.float32)

    hm_np = hm_ensemble[0].cpu().numpy()   # [2, 256, 256]
    heatmap = np.stack([
        cv2.resize(hm_np[c], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        for c in range(2)
    ])

    return coords_orig, heatmap


# ── visualisation ─────────────────────────────────────────────────────────────

def visualize(img_2d, seg_2d, coords, heatmap, gt_coords,
              slice_idx, mre_info, save_path):
    has_gt  = gt_coords is not None
    has_seg = seg_2d is not None

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
    ax0.set_title(f"ensemble  {title}", fontsize=8)
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
    ax1.set_title("LM1 heatmap (ensemble avg)", fontsize=8)
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
    ax2.set_title("LM2 heatmap (ensemble avg)", fontsize=8)
    ax2.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── full test set evaluation ──────────────────────────────────────────────────

def evaluate_test_set(models_and_cfgs, device, out_dir, use_tta=True):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(".nii.gz"))

    all_mres, all_e1, all_e2 = [], [], []

    seg_label = SEG_DIR if os.path.isdir(SEG_DIR) else "None (blank channel)"
    print(f"\n{'-'*75}")
    print(f"Test images: {TEST_IMAGE_DIR}")
    print(f"Seg masks  : {seg_label}")
    print(f"TTA        : {use_tta}")
    print(f"Models     : {[c['tag'] for _, c in models_and_cfgs]}")
    print(f"\n{'File':<30} {'Slice':>5}  {'MRE':>8}  {'LM1':>8}  {'LM2':>8}")
    print(f"{'-'*75}")

    for fname in files:
        img_path = os.path.join(TEST_IMAGE_DIR, fname)
        gt_path  = os.path.join(TEST_GT_DIR, fname)
        seg_path = os.path.join(SEG_DIR, fname)

        if not os.path.exists(gt_path):
            print(f"  SKIP (no GT): {fname}")
            continue

        img    = nib.load(img_path).get_fdata().astype(np.float32)
        gt_vol = nib.load(gt_path).get_fdata().astype(np.float32)
        seg    = None
        if os.path.exists(seg_path):
            seg = np.round(nib.load(seg_path).get_fdata()).astype(np.float32)

        n_slices = min(img.shape[2], gt_vol.shape[2])

        for i in range(n_slices):
            gt = extract_gt_coords(gt_vol, i)
            if gt is None:
                continue

            img_2d = img[:, :, i]
            seg_2d = seg[:, :, i] if seg is not None else None

            coords, heatmap = ensemble_predict(
                models_and_cfgs, img_2d, seg_2d, device, use_tta=use_tta
            )
            coords = enforce_ordering(coords)
            coords = match_landmarks(coords, gt)

            mre, e1, e2 = compute_mre(coords, gt)
            all_mres.append(mre)
            all_e1.append(e1)
            all_e2.append(e2)

            print(f"  {fname:<28} {i:>5}  {mre:>8.2f}px  {e1:>7.1f}px  {e2:>7.1f}px")

            save_path = os.path.join(
                out_dir, f"{fname.replace('.nii.gz','')}_slice{i:03d}.png"
            )
            visualize(img_2d, seg_2d, coords, heatmap, gt,
                      slice_idx=i, mre_info=(mre, e1, e2), save_path=save_path)

    print(f"{'-'*75}")

    if not all_mres:
        print("No annotated slices found.")
        return np.array([])

    arr  = np.array(all_mres)
    arr1 = np.array(all_e1)
    arr2 = np.array(all_e2)

    print(f"\n{'='*75}")
    print(f"  SUMMARY — {len(arr)} annotated slices  [ensemble: 1ch + combo + mixed]")
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
        "ensemble": [cfg["tag"] for _, cfg in models_and_cfgs],
        "checkpoints": [cfg["checkpoint"] for _, cfg in models_and_cfgs],
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

    return arr


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble inference on rv_landmark test set "
                    "(1ch + combo-instnorm + mixed-instnorm)"
    )
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable test-time augmentation")
    parser.add_argument("--out", default="inference_rv_results",
                        help="Root output directory (default: inference_rv_results)")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = not args.no_tta

    print(f"\nDevice  : {device}")
    print(f"TTA     : {use_tta}")
    print(f"\nLoading {len(ENSEMBLE)} models …")

    models_and_cfgs = []
    for cfg in ENSEMBLE:
        print(f"  [{cfg['tag']}] {cfg['checkpoint']}")
        model = load_model(cfg, device)
        models_and_cfgs.append((model, cfg))

    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(args.out, f"ensemble_1ch_combo_mixed_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput  → {out_dir}")

    evaluate_test_set(models_and_cfgs, device, out_dir, use_tta=use_tta)
