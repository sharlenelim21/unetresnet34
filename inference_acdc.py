"""
Evaluate a trained UNetResNet34 checkpoint on the ACDC test set (patients 091-100).

Usage:
  python inference_acdc.py --checkpoint PATH [--in-channels {1,2}] [--instance-norm]

Outputs:
  • Per-slice table printed to stdout
  • Summary metrics printed to stdout
  • inference_acdc_results/TIMESTAMP/TIMESTAMP_results.json
  • inference_acdc_results/TIMESTAMP/<patient>_<frame>_slice<zzz>.png  (one per slice)
"""

import os, json, argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset.acdc_landmark_dataset import ACDCLandmarkDataset
from models.unet_resnet34 import UNetResNet34
from utils.postprocess import gaussian_subpixel_argmax
from utils.metrics import (
    compute_mre, compute_mre_per_landmark,
    compute_sdr_multi, compute_per_sample_mre, compute_mre_percentiles,
)

# ─────────────────────────────── config ───────────────────────────────────────
IMAGE_DIR   = "data/acdc/images"
MASK_DIR    = "data/acdc/masks"
RVIP_DIR    = "data/acdc/points"
TEST_IDS    = [f"patient{i:03d}" for i in range(91, 101)]
BATCH_SIZE  = 8
NUM_WORKERS = 2
SIGMA_EVAL  = 1.0
OUT_DIR     = "inference_acdc_results"
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────── TTA ──────────────────────────────────────────

@torch.no_grad()
def tta_predict(model, images, device):
    """4-variant TTA: original, h-flip, v-flip, h+v-flip."""
    images = images.to(device)
    variants = [
        images,
        torch.flip(images, [3]),
        torch.flip(images, [2]),
        torch.flip(images, [2, 3]),
    ]
    hms = []
    for v in variants:
        out = model(v)
        if isinstance(out, (tuple, list)):
            out = out[0]
        hms.append(torch.sigmoid(out))

    hms[1] = torch.flip(hms[1], [3])
    hms[2] = torch.flip(hms[2], [2])
    hms[3] = torch.flip(hms[3], [2, 3])

    avg_hm = torch.stack(hms).mean(0)          # [B, 2, H, W]
    coords = gaussian_subpixel_argmax(avg_hm)   # [B, 4]
    return coords, avg_hm


def enforce_superior(coords: torch.Tensor) -> torch.Tensor:
    """Ensure LM1 (index 0) has smaller y than LM2 (inferior)."""
    c    = coords.clone()
    swap = c[:, 1] > c[:, 3]
    c[swap, 0], c[swap, 2] = coords[swap, 2].clone(), coords[swap, 0].clone()
    c[swap, 1], c[swap, 3] = coords[swap, 3].clone(), coords[swap, 1].clone()
    return c


def match_landmarks(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compare normal and swapped assignment; keep whichever gives lower total error.
    pred, gt: [4] tensors (x1,y1,x2,y2).
    """
    d_normal  = (torch.norm(pred[:2] - gt[:2]) + torch.norm(pred[2:] - gt[2:])).item()
    pred_swap = torch.cat([pred[2:], pred[:2]])
    d_swapped = (torch.norm(pred_swap[:2] - gt[:2]) + torch.norm(pred_swap[2:] - gt[2:])).item()
    return pred_swap if d_swapped < d_normal else pred


# ─────────────────────────────── per-landmark SDR ─────────────────────────────

def sdr_lm(pred_lm, gt_lm, thresh):
    """pred_lm, gt_lm: [N, 2]. Returns percentage."""
    d = torch.norm(pred_lm - gt_lm, dim=1)
    return (d < thresh).float().mean().item() * 100.0


# ─────────────────────────────── visualisation ────────────────────────────────

def _save_slice_png(img_np, pred, gt, e1, e2, mre, out_path):
    """
    Save a single-slice overlay PNG.
    img_np : (256, 256) normalised float array (first MRI channel)
    pred   : (4,) numpy [x1,y1,x2,y2] — predicted coords
    gt     : (4,) numpy [x1,y1,x2,y2] — GT coords
    e1/e2  : per-landmark errors (px)
    mre    : mean of e1, e2
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(img_np, cmap="gray", interpolation="nearest")

    # GT landmarks — green
    ax.plot(gt[0], gt[1], "g+", markersize=10, markeredgewidth=2, label="GT LM1")
    ax.plot(gt[2], gt[3], "gx", markersize=10, markeredgewidth=2, label="GT LM2")

    # Predicted landmarks — red (LM1) and blue (LM2)
    ax.plot(pred[0], pred[1], "r.", markersize=8, label="Pred LM1")
    ax.plot(pred[2], pred[3], "b.", markersize=8, label="Pred LM2")

    ax.set_title(f"MRE={mre:.2f}px  LM1={e1:.2f}  LM2={e2:.2f}", fontsize=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────── main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ACDC checkpoint on test set")
    parser.add_argument("--checkpoint",  required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 2],
                        help="Number of input channels (1=MRI only, 2=MRI+mask)")
    parser.add_argument("--instance-norm", action="store_true",
                        help="Use InstanceNorm in encoder (required for instance-norm checkpoints)")
    parser.add_argument("--group-norm", action="store_true",
                        help="Use GroupNorm instead of BatchNorm")
    args = parser.parse_args()
    if args.instance_norm and args.group_norm:
        raise ValueError("Cannot use both --instance-norm and --group-norm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice       : {device}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"In-channels  : {args.in_channels}")
    print(f"Instance-norm: {args.instance_norm}")

    # ── dataset ───────────────────────────────────────────────────────────────
    test_ds = ACDCLandmarkDataset(
        IMAGE_DIR, MASK_DIR, RVIP_DIR,
        patient_ids=TEST_IDS,
        in_channels=args.in_channels,
        augment=False,
        sigma=SIGMA_EVAL,
    )
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test slices : {len(test_ds)}\n")

    # ── model ─────────────────────────────────────────────────────────────────
    model = UNetResNet34(in_channels=args.in_channels, num_classes=2,
                         pretrained=False,
                         use_instance_norm=args.instance_norm,
                         use_group_norm=args.group_norm).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  warn: {len(missing)} missing keys in checkpoint")
    if unexpected:
        print(f"  warn: {len(unexpected)} unexpected keys in checkpoint")
    model.eval()
    print("Model loaded.\n")

    # ── output directory ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(OUT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # ── inference ─────────────────────────────────────────────────────────────
    all_preds, all_gts = [], []
    sample_mres        = []
    per_slice_records  = []

    slice_meta = test_ds.samples

    # Print table header
    print(f"{'idx':>4}  {'file':<40}  {'sl':>3}  {'MRE':>6}  {'LM1':>6}  {'LM2':>6}")
    print("-" * 72)

    global_i = 0
    # Keep raw images for visualisation (batch may be shuffled=False so order is stable)
    with torch.no_grad():
        for imgs, _, coords in loader:
            pred_coords, _ = tta_predict(model, imgs, device)
            pred_coords    = enforce_superior(pred_coords.cpu())

            # Per-sample landmark matching
            matched = []
            for i in range(pred_coords.size(0)):
                p = match_landmarks(pred_coords[i], coords[i])
                matched.append(p)
            pred_coords = torch.stack(matched)

            per_s = compute_per_sample_mre(pred_coords, coords)
            sample_mres.extend(per_s.tolist())

            for i in range(pred_coords.size(0)):
                p   = pred_coords[i].numpy()
                g   = coords[i].numpy()
                e1  = float(np.linalg.norm(p[:2] - g[:2]))
                e2  = float(np.linalg.norm(p[2:] - g[2:]))
                mre = (e1 + e2) / 2.0

                meta  = slice_meta[global_i]
                fname = meta[0]
                z_idx = meta[1]

                print(f"{global_i:>4}  {fname:<40}  {z_idx:>3}  {mre:>6.2f}  {e1:>6.2f}  {e2:>6.2f}")
                per_slice_records.append({
                    "file":      fname,
                    "slice":     z_idx,
                    "mre":       round(mre, 4),
                    "lm1_error": round(e1, 4),
                    "lm2_error": round(e2, 4),
                })

                # ── save per-slice PNG ─────────────────────────────────────────
                img_np = imgs[i, 0].numpy()   # first channel (MRI), (256,256)
                stem   = os.path.splitext(fname.replace(".nii.gz", ""))[0]
                png_name = f"{stem}_slice{z_idx:03d}.png"
                _save_slice_png(img_np, p, g, e1, e2, mre,
                                os.path.join(run_dir, png_name))

                global_i += 1

            all_preds.extend(pred_coords.numpy())
            all_gts.extend(coords.numpy())

    # ── aggregate metrics ─────────────────────────────────────────────────────
    pred_t = torch.tensor(np.array(all_preds))
    gt_t   = torch.tensor(np.array(all_gts))

    mre               = compute_mre(pred_t, gt_t).item()
    mre_lm1, mre_lm2 = compute_mre_per_landmark(pred_t, gt_t)
    sdr_all           = compute_sdr_multi(pred_t, gt_t, thresholds=(2, 3, 5, 10))
    pct               = compute_mre_percentiles(sample_mres)

    pred_lm1 = pred_t.view(-1, 2, 2)[:, 0]
    pred_lm2 = pred_t.view(-1, 2, 2)[:, 1]
    gt_lm1   = gt_t.view(-1, 2, 2)[:, 0]
    gt_lm2   = gt_t.view(-1, 2, 2)[:, 1]

    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Slices evaluated : {len(all_preds)}")
    print(f"Mean MRE : {mre:.2f}px")
    print(f"Std  MRE : {float(np.std(sample_mres)):.2f}px")
    print(f"P50  MRE : {pct[50]:.2f}px")
    print(f"P90  MRE : {pct[90]:.2f}px")
    print(f"Max  MRE : {pct[100]:.2f}px")
    print(f"LM1  MRE : {mre_lm1:.2f}px")
    print(f"LM2  MRE : {mre_lm2:.2f}px")
    print()
    print("SDR (all landmarks):")
    for t in (2, 3, 5, 10):
        print(f"  SDR@{t:>2}px : {sdr_all[t]*100:.1f}%")
    print()
    print("SDR per landmark:")
    for t in (2, 3, 5, 10):
        print(f"  @{t:>2}px  LM1={sdr_lm(pred_lm1, gt_lm1, t):.1f}%"
              f"   LM2={sdr_lm(pred_lm2, gt_lm2, t):.1f}%")
    print("=" * 50)

    # ── save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(run_dir, f"{timestamp}_results.json")

    results = {
        "checkpoint":  args.checkpoint,
        "in_channels": args.in_channels,
        "n_slices":    len(all_preds),
        # flat summary keys matching training results.json format
        "best_val_p90":  None,       # N/A for inference-only run
        "best_val_mre":  None,
        "best_val_sdr5": None,
        "best_epoch":    None,
        "test_mre":      round(mre, 4),
        "test_mre_lm1":  round(mre_lm1, 4),
        "test_mre_lm2":  round(mre_lm2, 4),
        "test_sdr2":     round(sdr_all[2], 4),
        "test_sdr5":     round(sdr_all[5], 4),
        "test_sdr10":    round(sdr_all[10], 4),
        "test_p50":      round(pct[50], 4),
        "test_p90":      round(pct[90], 4),
        "checkpoint":    args.checkpoint,
        "total_epochs":  None,
        # extended inference-specific fields
        "std_mre":       round(float(np.std(sample_mres)), 4),
        "max_mre":       round(pct[100], 4),
        "sdr_lm1": {str(t): round(sdr_lm(pred_lm1, gt_lm1, t), 2) for t in (2, 3, 5, 10)},
        "sdr_lm2": {str(t): round(sdr_lm(pred_lm2, gt_lm2, t), 2) for t in (2, 3, 5, 10)},
        "per_slice": per_slice_records,
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {out_path}")
    print(f"Slice PNGs   → {run_dir}/")


if __name__ == "__main__":
    main()
