"""
inference_acdc.py — 2-Channel Landmark Inference for ACDC
===========================================================
Uses the 2-channel model trained on ACDC (MRI + seg mask).
Evaluates on the ACDC test split with ground truth RVIP labels.

Usage (single volume):
    python inference_acdc.py --checkpoint checkpoints/acdc_2ch_.../best_model.pth
                             --image data/acdc-cleaned/test_split/images/patient081_frame01.nii.gz
                             --auto

Usage (all test volumes):
    python inference_acdc.py --checkpoint checkpoints/acdc_2ch_.../best_model.pth
                             --all

Usage (evaluate full test split):
    python inference_acdc.py --checkpoint checkpoints/acdc_2ch_.../best_model.pth
                             --eval
"""

import argparse
import os
import numpy as np
import torch
import cv2
import nibabel as nib
import nrrd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

from models.unet_resnet34 import UNetResNet34
from utils.postprocess import gaussian_subpixel_argmax

# ── config ────────────────────────────────────────────────────────────────────
MODEL_INPUT_SIZE = 256
IN_CHANNELS      = 2

TEST_IMAGE_DIR = "data/acdc-cleaned/test_split/images"
TEST_MASK_DIR  = "data/acdc-cleaned/test_split/masks"
TEST_RVIP_DIR  = "data/acdc-cleaned/test_split/rvip"


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet34(
        in_channels=IN_CHANNELS,
        num_classes=2,
        dropout=0.0,
        pretrained=False,
        cardiac_pretrained=False
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint : {checkpoint}")
    print(f"Device            : {device}")
    print(f"Input channels    : {IN_CHANNELS} (MRI + seg mask)")
    return model


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_2ch(img_2d, seg_2d):
    """
    Prepare a 2-channel input tensor from MRI slice and seg mask slice.
    Returns [1, 2, 256, 256] tensor.
    """
    # Channel 1 — MRI
    img_r = cv2.resize(img_2d.astype(np.float32), (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    mu    = img_r.mean()
    std   = img_r.std() + 1e-8
    img_r = (img_r - mu) / std

    # Channel 2 — seg mask (nearest neighbour, normalise to 0-1)
    seg_r = cv2.resize(seg_2d.astype(np.float32), (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                       interpolation=cv2.INTER_NEAREST)
    seg_max = seg_r.max()
    if seg_max > 0:
        seg_r = seg_r / seg_max

    # Stack → [1, 2, 256, 256]
    two_ch = np.stack([img_r, seg_r], axis=0)
    return torch.tensor(two_ch, dtype=torch.float32).unsqueeze(0)


# ── TTA predict ───────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_predict(model, tensor, device):
    """4-variant TTA: original + h-flip + v-flip + hv-flip, averaged."""
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


# ── GT extraction ─────────────────────────────────────────────────────────────

def extract_gt_coords(rvip_2d):
    """
    Extract GT landmark coords from RVIP mask slice.
    Label 1 = upper insertion point, Label 2 = lower insertion point.
    Returns (x1, y1, x2, y2) in original pixel space, or None.
    """
    pts_1 = np.argwhere(rvip_2d == 1)
    pts_2 = np.argwhere(rvip_2d == 2)

    if len(pts_1) == 0 or len(pts_2) == 0:
        return None

    p1 = pts_1.mean(axis=0)   # (row, col)
    p2 = pts_2.mean(axis=0)

    # LM1 = superior (smaller row index)
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    return np.array([p1[1], p1[0], p2[1], p2[0]], dtype=np.float32)


# ── core inference ────────────────────────────────────────────────────────────

def predict_landmarks(img_2d, seg_2d, model, device, use_tta=True):
    """
    Predict landmarks on a single slice using 2-channel input.

    img_2d : [H, W] MRI slice
    seg_2d : [H, W] segmentation mask (labels 0/1/2/3)

    Returns:
        coords_orig : [4] (x1,y1,x2,y2) in original pixel space
        heatmap     : [2, H, W] at original resolution
    """
    H_orig, W_orig = img_2d.shape
    tensor = preprocess_2ch(img_2d, seg_2d)

    if use_tta:
        heatmap_256 = tta_predict(model, tensor, device)
    else:
        with torch.no_grad():
            heatmap_256 = torch.sigmoid(model(tensor.to(device)))

    # Subpixel localisation in 256-space
    coords_256 = gaussian_subpixel_argmax(heatmap_256, window=7)[0].cpu().numpy()

    # Scale back to original resolution
    coords_orig = np.array([
        coords_256[0] * W_orig / MODEL_INPUT_SIZE,
        coords_256[1] * H_orig / MODEL_INPUT_SIZE,
        coords_256[2] * W_orig / MODEL_INPUT_SIZE,
        coords_256[3] * H_orig / MODEL_INPUT_SIZE,
    ], dtype=np.float32)

    # Upsample heatmap to original resolution
    hm_np = heatmap_256[0].cpu().numpy()
    heatmap_orig = np.stack([
        cv2.resize(hm_np[c], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        for c in range(hm_np.shape[0])
    ])

    return coords_orig, heatmap_orig


# ── landmark ordering helpers ─────────────────────────────────────────────────

def enforce_ordering(pred_coords):
    """
    Apply the same superior/inferior ordering used during training.
    LM1 = smaller y (higher in image = superior).
    Use this when no GT is available.
    """
    x1, y1, x2, y2 = pred_coords
    if y1 > y2:
        return np.array([x2, y2, x1, y1], dtype=np.float32)
    return pred_coords


def match_landmarks(pred_coords, gt_coords):
    """
    Find the assignment of predicted landmarks to GT that minimises
    total error. Handles the case where LM1/LM2 are swapped.
    Use this when GT is available (evaluation).
    """
    pred_lm1 = pred_coords[:2]
    pred_lm2 = pred_coords[2:]
    gt_lm1   = gt_coords[:2]
    gt_lm2   = gt_coords[2:]

    e_normal  = (np.linalg.norm(pred_lm1 - gt_lm1) +
                 np.linalg.norm(pred_lm2 - gt_lm2))
    e_swapped = (np.linalg.norm(pred_lm2 - gt_lm1) +
                 np.linalg.norm(pred_lm1 - gt_lm2))

    if e_swapped < e_normal:
        return np.array([pred_lm2[0], pred_lm2[1],
                         pred_lm1[0], pred_lm1[1]], dtype=np.float32)
    return pred_coords


# ── MRE helper ────────────────────────────────────────────────────────────────

def compute_mre(pred, gt):
    e1 = np.linalg.norm(pred[:2] - gt[:2])
    e2 = np.linalg.norm(pred[2:] - gt[2:])
    return 0.5 * (e1 + e2), e1, e2


# ── visualisation ─────────────────────────────────────────────────────────────

def visualize_prediction(img_2d, seg_2d, coords, heatmap=None,
                          gt_coords=None, slice_idx=None,
                          mre_info=None, save_path=None, show=False):
    has_heatmap = heatmap is not None
    has_gt      = gt_coords is not None
    n_panels    = 4 if has_heatmap else 2

    fig = plt.figure(figsize=(5 * n_panels, 5))
    gs  = gridspec.GridSpec(1, n_panels, figure=fig)

    slice_label = f"slice {slice_idx}" if slice_idx is not None else ""
    if mre_info is not None:
        mre, e1, e2 = mre_info
        title = f"{slice_label}  |  MRE={mre:.2f}px  (LM1={e1:.1f}px  LM2={e2:.1f}px)"
    else:
        title = slice_label

    # Panel 1 — MRI + landmarks
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_2d, cmap="gray", aspect="auto")
    if has_gt:
        ax0.scatter([gt_coords[0], gt_coords[2]], [gt_coords[1], gt_coords[3]],
                    c="lime", s=60, zorder=4, edgecolors="black",
                    linewidths=0.6, label="GT")
        ax0.plot([gt_coords[0], gt_coords[2]], [gt_coords[1], gt_coords[3]],
                 color="lime", lw=0.8, alpha=0.5, linestyle="--", zorder=3)
    ax0.scatter([coords[0]], [coords[1]], c="red", s=60, zorder=5,
                edgecolors="white", linewidths=0.8, label="Pred LM1")
    ax0.scatter([coords[2]], [coords[3]], c="deepskyblue", s=60, zorder=5,
                edgecolors="white", linewidths=0.8, label="Pred LM2")
    ax0.plot([coords[0], coords[2]], [coords[1], coords[3]],
             "w--", lw=0.8, alpha=0.6, zorder=3)
    if has_gt:
        ax0.plot([coords[0], gt_coords[0]], [coords[1], gt_coords[1]],
                 color="yellow", lw=1.0, alpha=0.8, zorder=4)
        ax0.plot([coords[2], gt_coords[2]], [coords[3], gt_coords[3]],
                 color="yellow", lw=1.0, alpha=0.8, zorder=4)
    ax0.set_title(title, fontsize=8)
    ax0.legend(loc="lower right", fontsize=6, markerscale=0.7)
    ax0.axis("off")

    # Panel 2 — Seg mask overlay
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(img_2d, cmap="gray", aspect="auto")
    ax1.imshow(seg_2d, alpha=0.35, cmap="nipy_spectral", vmin=0, vmax=3)
    if has_gt:
        ax1.scatter([gt_coords[0], gt_coords[2]], [gt_coords[1], gt_coords[3]],
                    c="lime", s=60, zorder=4, edgecolors="black", linewidths=0.6)
    ax1.scatter([coords[0]], [coords[1]], c="red", s=60, zorder=5,
                edgecolors="white", linewidths=0.8)
    ax1.scatter([coords[2]], [coords[3]], c="deepskyblue", s=60, zorder=5,
                edgecolors="white", linewidths=0.8)
    ax1.set_title("MRI + Seg mask overlay", fontsize=8)
    ax1.axis("off")

    # Panel 3 — LM1 heatmap
    if has_heatmap:
        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(img_2d, cmap="gray", aspect="auto")
        ax2.imshow(heatmap[0], cmap="hot", alpha=0.5,
                   vmin=0, vmax=max(heatmap[0].max(), 1e-6))
        ax2.scatter([coords[0]], [coords[1]], c="red", s=60, zorder=4,
                    edgecolors="white", linewidths=0.8)
        if has_gt:
            ax2.scatter([gt_coords[0]], [gt_coords[1]], c="lime", s=60,
                        zorder=4, edgecolors="black", linewidths=0.6)
        ax2.set_title("LM1 heatmap  (red=pred  green=GT)", fontsize=8)
        ax2.axis("off")

        # Panel 4 — LM2 heatmap
        ax3 = fig.add_subplot(gs[3])
        ax3.imshow(img_2d, cmap="gray", aspect="auto")
        ax3.imshow(heatmap[1], cmap="hot", alpha=0.5,
                   vmin=0, vmax=max(heatmap[1].max(), 1e-6))
        ax3.scatter([coords[2]], [coords[3]], c="deepskyblue", s=60, zorder=4,
                    edgecolors="white", linewidths=0.8)
        if has_gt:
            ax3.scatter([gt_coords[2]], [gt_coords[3]], c="lime", s=60,
                        zorder=4, edgecolors="black", linewidths=0.6)
        ax3.set_title("LM2 heatmap  (blue=pred  green=GT)", fontsize=8)
        ax3.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Saved -> {save_path}")
    if show:
        plt.show()
    plt.close()


# ── full test set evaluation ──────────────────────────────────────────────────

def evaluate_test_set(model, device, out_dir="inference_acdc_results", use_tta=True):
    """
    Run inference on all ACDC test split volumes and report aggregate metrics.
    """
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(TEST_IMAGE_DIR)
                    if f.endswith(".nii.gz") and "_gt" not in f])

    all_mres = []
    all_e1   = []
    all_e2   = []

    print(f"\n{'-'*75}")
    print(f"{'File':<35} {'Slice':>5}  {'MRE':>8}  {'LM1 err':>8}  {'LM2 err':>8}")
    print(f"{'-'*75}")

    for fname in files:
        base      = fname.replace(".nii.gz", "")
        mask_f    = base + "_gt.nii.gz"
        rvip_f    = base + "_rvip.nrrd"

        img_path  = os.path.join(TEST_IMAGE_DIR, fname)
        mask_path = os.path.join(TEST_MASK_DIR,  mask_f)
        rvip_path = os.path.join(TEST_RVIP_DIR,  rvip_f)

        if not os.path.exists(mask_path) or not os.path.exists(rvip_path):
            print(f"  SKIP (missing mask or rvip): {fname}")
            continue

        img       = nib.load(img_path).get_fdata().astype(np.float32)
        seg       = np.round(nib.load(mask_path).get_fdata()).astype(np.float32)
        rvip, _   = nrrd.read(rvip_path)

        n_slices = min(img.shape[2], seg.shape[2], rvip.shape[2])

        for i in range(n_slices):
            rvip_2d = rvip[:, :, i]
            gt      = extract_gt_coords(rvip_2d)
            if gt is None:
                continue

            img_2d = img[:, :, i]
            seg_2d = seg[:, :, i]

            coords, heatmap = predict_landmarks(
                img_2d, seg_2d, model, device, use_tta=use_tta
            )

            # Step 1 — enforce superior/inferior ordering (matches training)
            coords = enforce_ordering(coords)

            # Step 2 — optimal GT matching (corrects any remaining swap)
            if gt is not None:
                coords = match_landmarks(coords, gt)

            mre, e1, e2 = compute_mre(coords, gt)
            all_mres.append(mre)
            all_e1.append(e1)
            all_e2.append(e2)

            print(f"  {fname:<33} {i:>5}  {mre:>8.2f}px  {e1:>7.1f}px  {e2:>7.1f}px")

            save_path = os.path.join(out_dir, f"{base}_slice{i:03d}.png")
            visualize_prediction(
                img_2d, seg_2d, coords, heatmap,
                gt_coords=gt, slice_idx=i,
                mre_info=(mre, e1, e2),
                save_path=save_path
            )

    print(f"{'-'*75}")

    if len(all_mres) > 0:
        arr = np.array(all_mres)
        print(f"\nSummary across {len(arr)} annotated slices:")
        print(f"  Mean MRE  : {arr.mean():.2f}px")
        print(f"  Std MRE   : {arr.std():.2f}px")
        print(f"  P50  MRE  : {np.percentile(arr, 50):.2f}px")
        print(f"  P90  MRE  : {np.percentile(arr, 90):.2f}px")
        print(f"  Max  MRE  : {arr.max():.2f}px")
        print(f"  SDR@2px   : {(arr < 2.0).mean():.1%}")
        print(f"  SDR@3px   : {(arr < 3.0).mean():.1%}")
        print(f"  SDR@5px   : {(arr < 5.0).mean():.1%}")
        print(f"  SDR@10px  : {(arr < 10.0).mean():.1%}")
        print(f"\n  Mean LM1 error : {np.mean(all_e1):.2f}px")
        print(f"  Mean LM2 error : {np.mean(all_e2):.2f}px")
        print(f"\nResults saved to: {out_dir}/")

    return arr


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2-channel ACDC landmark inference"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pth from train_acdc.py")
    parser.add_argument("--image", default=None,
                        help="Path to a single .nii.gz image file")
    parser.add_argument("--slice", type=int, default=None,
                        help="Specific slice index")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-select slices with RVIP annotations")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate full test split and report metrics")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable TTA")
    parser.add_argument("--out", default="inference_acdc_results",
                        help="Output directory for result images")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(args.checkpoint, device)
    use_tta = not args.no_tta

    # ── create timestamped output directory ───────────────────────────────────
    # Extract checkpoint run name for context (e.g. acdc_2ch_2026-05-09_20-40-01)
    ckpt_run = os.path.basename(os.path.dirname(args.checkpoint))
    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir  = os.path.join(args.out, f"{ckpt_run}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir : {out_dir}")

    # ── full evaluation mode ──────────────────────────────────────────────────
    if args.eval:
        evaluate_test_set(model, device, out_dir=out_dir, use_tta=use_tta)

    # ── single image mode ─────────────────────────────────────────────────────
    elif args.image is not None:
        fname    = os.path.basename(args.image)
        base     = fname.replace(".nii.gz", "")
        img_dir  = os.path.dirname(args.image)

        # Derive seg mask and rvip paths
        for split in ["test_split", "train_split"]:
            mask_path = args.image.replace("images", "masks").replace(
                fname, base + "_gt.nii.gz")
            rvip_path = args.image.replace("images", "rvip").replace(
                fname, base + "_rvip.nrrd")
            if os.path.exists(mask_path):
                break

        img = nib.load(args.image).get_fdata().astype(np.float32)
        seg = np.round(nib.load(mask_path).get_fdata()).astype(np.float32) \
              if os.path.exists(mask_path) else np.zeros_like(img)
        rvip, _ = nrrd.read(rvip_path) if os.path.exists(rvip_path) else (None, None)

        print(f"\nVolume shape: {img.shape}")
        print(f"Seg mask   : {'found' if os.path.exists(mask_path) else 'NOT FOUND — using blank channel'}")
        print(f"RVIP labels: {'found' if rvip is not None else 'NOT FOUND — no GT comparison'}")

        # Decide which slices to process
        if args.auto and rvip is not None:
            slice_indices = [i for i in range(min(img.shape[2], rvip.shape[2]))
                             if np.any(rvip[:,:,i] == 1) and np.any(rvip[:,:,i] == 2)]
            print(f"Mode: AUTO — {len(slice_indices)} annotated slices: {slice_indices}")
        elif args.slice is not None:
            slice_indices = [args.slice]
            print(f"Mode: single slice {args.slice}")
        else:
            variances     = [img[:,:,i].var() for i in range(img.shape[2])]
            slice_indices = [sorted(range(len(variances)),
                                    key=lambda x: variances[x], reverse=True)[0]]
            print(f"Mode: auto-picked best slice: {slice_indices[0]}")

        print(f"\n{'-'*70}")
        all_mres = []

        for sl in slice_indices:
            img_2d = img[:, :, sl]
            seg_2d = seg[:, :, sl] if seg is not None else np.zeros_like(img_2d)
            coords, heatmap = predict_landmarks(img_2d, seg_2d, model, device, use_tta)

            # Enforce ordering then match to GT if available
            coords = enforce_ordering(coords)

            gt       = None
            mre_info = None
            if rvip is not None and sl < rvip.shape[2]:
                gt = extract_gt_coords(rvip[:, :, sl])
                if gt is not None:
                    coords = match_landmarks(coords, gt)
                    mre, e1, e2 = compute_mre(coords, gt)
                    mre_info    = (mre, e1, e2)
                    all_mres.append(mre)
                    print(f"  Slice {sl:>3}: MRE={mre:.2f}px  "
                          f"LM1={e1:.1f}px  LM2={e2:.1f}px")

            save_path = os.path.join(out_dir, f"{base}_slice{sl:03d}.png")
            visualize_prediction(
                img_2d, seg_2d, coords, heatmap,
                gt_coords=gt, slice_idx=sl,
                mre_info=mre_info, save_path=save_path
            )

        if len(all_mres) > 1:
            arr = np.array(all_mres)
            print(f"\nSummary: Mean MRE={arr.mean():.2f}px  "
                  f"P90={np.percentile(arr,90):.2f}px  "
                  f"SDR@5px={(arr<5).mean():.1%}")

        print(f"\nDone. Results saved to: {out_dir}/")

    else:
        print("Please specify --eval, --image, or use --help")