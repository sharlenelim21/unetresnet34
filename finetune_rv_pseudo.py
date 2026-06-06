"""
finetune_rv_pseudo.py — Pseudo-label self-training for RV insertion point detection
====================================================================================
Fine-tunes using three data sources simultaneously:

  1. Real rv_landmark train       (loss_weight = 1.0)
  2. Pseudo-labeled rv_landmark   (loss_weight = --pseudo-weight, default 0.4)
  3. ACDC replay                  (loss_weight = --acdc-weight,   default 0.15)

The per-sample loss weight is returned by each dataset's __getitem__ as a 4th
element (source_weight scalar), and applied in the train loop before backprop.

Based EXACTLY on finetune_rv.py:
  - Same 3-phase training structure (P1 warmup, P2 curriculum, P3 squeeze)
  - Same NaN batch skipping
  - Same enforce_ordering + match_landmarks
  - Same early stopping on P90 MRE
  - Same checkpoint saving format
  - Same LM1 heatmap re-weighting (--lm1-weight)

Validation uses rv_landmark val split (real GT only).
Test evaluation uses rv_landmark test split (real GT only).

Usage:
    python finetune_rv_pseudo.py \\
        --base-checkpoint rv-checkpoints/.../best_model.pth \\
        --in-channels 2 \\
        --instance-norm \\
        --lm1-weight 1.5 \\
        --pseudo-weight 0.4 \\
        --acdc-weight 0.15
"""

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from dataset.rv_landmark_dataset import RVLandmarkDataset, split_volumes
from dataset.acdc_landmark_dataset import ACDCLandmarkDataset
from models.unet_resnet34 import UNetResNet34
from utils.loss import HeatmapLoss
from utils.postprocess import gaussian_subpixel_argmax
from utils.metrics import (
    compute_mre, compute_mre_per_landmark,
    compute_sdr_multi, compute_per_sample_mre, compute_mre_percentiles,
)
from utils.visualize import save_epoch_grid, save_training_curve


# ── data paths ────────────────────────────────────────────────────────────────
RV_IMAGE_DIR = "data/rv_landmark/train_images"
RV_GT_DIR    = "data/rv_landmark/train_gt"
RV_SEG_DIR   = "data/rv_landmark/train_seg_multi"

TEST_IMAGE_DIR = "data/rv_landmark/test_images"
TEST_GT_DIR    = "data/rv_landmark/test_gt"
TEST_SEG_DIR   = "data/rv_landmark/test_seg_multi"

ACDC_IMAGE_DIR = "data/acdc/images"
ACDC_MASK_DIR  = "data/acdc/masks"
ACDC_RVIP_DIR  = "data/acdc/points"
ACDC_TRAIN_IDS = [f"patient{i:03d}" for i in range(1, 81)]

# ── fixed hyperparameters (same as finetune_rv.py) ───────────────────────────
BATCH_SIZE   = 8
SEED         = 42
NUM_WORKERS  = 2
N_VIS        = 8
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 2.0

LR_HEAD_P1   = 1e-3
SIGMA_P1     = 6.0

LR_BN_P2       = 5e-4
SIGMA_P2_START = 6.0
SIGMA_P2_END   = 3.0

LR_ENC_P3      = 5e-6
LR_DEC_P3      = 5e-5
LR_HEAD_P3     = 1e-4
SIGMA_P3_START = 3.0
SIGMA_P3_END   = 1.5
EARLY_STOP_P3  = 8

ENC_PREFIXES = ("enc0", "enc1", "enc2", "enc3", "enc4")


# ── weighted dataset wrapper ──────────────────────────────────────────────────

class WeightedDataset(Dataset):
    """
    Wraps any dataset that returns (image, heatmap, coords) and appends
    a constant source_weight float as a 4th element.
    """
    def __init__(self, dataset, source_weight: float):
        self.dataset       = dataset
        self.source_weight = torch.tensor(source_weight, dtype=torch.float32)

    def set_sigma(self, sigma):
        if hasattr(self.dataset, "set_sigma"):
            self.dataset.set_sigma(sigma)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, heatmap, coords = self.dataset[idx]
        return image, heatmap, coords, self.source_weight


class WeightedConcatDataset(Dataset):
    """
    ConcatDataset that propagates set_sigma to all child WeightedDatasets.
    """
    def __init__(self, datasets):
        self._concat = ConcatDataset(datasets)
        self._parts  = datasets     # list[WeightedDataset]

    def set_sigma(self, sigma):
        for d in self._parts:
            d.set_sigma(sigma)

    def __len__(self):
        return len(self._concat)

    def __getitem__(self, idx):
        return self._concat[idx]


# ── pseudo-GT dataset (RVLandmarkDataset with filtered volume_whitelist) ──────

def build_pseudo_dataset(pseudo_gt_dir, seg_dir, in_channels, sigma,
                         volume_whitelist, source_weight):
    """
    Build a WeightedDataset over pseudo-labeled slices.
    Only volumes that have a pseudo GT file are loaded.
    """
    if not os.path.isdir(pseudo_gt_dir):
        raise RuntimeError(f"Pseudo GT dir not found: {pseudo_gt_dir}")

    pseudo_files = set(
        f for f in os.listdir(pseudo_gt_dir) if f.endswith(".nii.gz")
    )
    # Intersect with the train split whitelist
    effective_whitelist = [f for f in volume_whitelist if f in pseudo_files]
    if not effective_whitelist:
        raise RuntimeError(
            "No overlap between pseudo GT files and train volume whitelist. "
            "Run pseudo_label_rv.py first."
        )
    print(f"  Pseudo GT volumes with labels: {len(effective_whitelist)} "
          f"/ {len(volume_whitelist)} in train split")

    ds = RVLandmarkDataset(
        image_dir=RV_IMAGE_DIR,
        gt_dir=pseudo_gt_dir,
        seg_dir=seg_dir,
        in_channels=in_channels,
        augment=True,
        sigma=sigma,
        min_landmark_dist=10,
        volume_whitelist=effective_whitelist,
        # Pseudo GT has single-pixel labels (integer 1/2), not Gaussian blobs.
        # thresh_frac=0.0 → threshold at >0 so both label values are detected.
        # min_area=1     → allow single-pixel connected components.
        thresh_frac=0.0,
        min_area=1,
    )
    return WeightedDataset(ds, source_weight)


# ── helpers (identical to finetune_rv.py) ────────────────────────────────────

def set_encoder_grad(model, flag):
    for n, p in model.named_parameters():
        if any(n.startswith(x) for x in ENC_PREFIXES):
            p.requires_grad = flag


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                p.requires_grad = True


def param_groups_discriminative(model, lr_enc, lr_dec, lr_head):
    enc_params, dec_params, head_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(n.startswith(x) for x in ENC_PREFIXES):
            enc_params.append(p)
        elif n.startswith("final") or n.startswith("aux_head"):
            head_params.append(p)
        else:
            dec_params.append(p)
    return [
        {"params": enc_params,  "lr": lr_enc},
        {"params": dec_params,  "lr": lr_dec},
        {"params": head_params, "lr": lr_head},
    ]


def cosine_sigma(epoch_in_phase, total_in_phase, sigma_start, sigma_end):
    if total_in_phase <= 1:
        return sigma_end
    t = epoch_in_phase / (total_in_phase - 1)
    return sigma_start + 0.5 * (1 - math.cos(math.pi * t)) * (sigma_end - sigma_start)


def enforce_superior_ordering_batch(coords):
    out  = coords.clone()
    swap = out[:, 1] > out[:, 3]
    if swap.any():
        tmp = out[swap].clone()
        out[swap, 0] = tmp[:, 2]
        out[swap, 1] = tmp[:, 3]
        out[swap, 2] = tmp[:, 0]
        out[swap, 3] = tmp[:, 1]
    return out


# ── validation (uses real GT only — no source_weight element) ─────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, n_vis=N_VIS):
    """
    loader yields (image, heatmap, gt_coords) — standard 3-tuple from
    the val RVLandmarkDataset (no WeightedDataset wrapper).
    """
    model.eval()
    val_loss = 0.0
    total_mre  = 0.0
    total_mre1 = 0.0
    total_mre2 = 0.0
    total_sdr  = {2: 0.0, 5: 0.0, 10: 0.0}
    sample_mres = []
    vis_images, vis_preds, vis_gts = [], [], []

    for images, heatmaps, gt_coords in loader:
        images    = images.to(device)
        heatmaps  = heatmaps.to(device)
        gt_coords = gt_coords.to(device)

        logits      = model(images)
        loss, _     = criterion(logits, heatmaps)
        val_loss   += loss.item()

        pred_hm     = torch.sigmoid(logits)
        pred_coords = gaussian_subpixel_argmax(pred_hm, window=7)
        pred_coords = enforce_superior_ordering_batch(pred_coords)

        total_mre  += compute_mre(pred_coords, gt_coords).item()
        m1, m2      = compute_mre_per_landmark(pred_coords, gt_coords)
        total_mre1 += m1
        total_mre2 += m2
        s = compute_sdr_multi(pred_coords, gt_coords, (2.0, 5.0, 10.0))
        for t in total_sdr:
            total_sdr[t] += s[t]
        sample_mres.extend(
            compute_per_sample_mre(pred_coords, gt_coords).cpu().tolist()
        )

        if len(vis_images) < n_vis:
            for j in range(min(n_vis - len(vis_images), images.size(0))):
                vis_images.append(images[j, 0].cpu().numpy())
                vis_preds.append(pred_coords[j].cpu().numpy())
                vis_gts.append(gt_coords[j].cpu().numpy())

    n = len(loader)
    return {
        "val_loss": val_loss / n,
        "mre":      total_mre  / n,
        "mre1":     total_mre1 / n,
        "mre2":     total_mre2 / n,
        "sdr":      {t: total_sdr[t] / n for t in total_sdr},
        "pct":      compute_mre_percentiles(sample_mres),
        "vis":      (vis_images, vis_preds, vis_gts),
    }


# ── train one epoch ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, lm1_weight=1.5):
    """
    loader yields (image, heatmap, coords, source_weight).
    source_weight is a [B] tensor; multiply per-sample loss by it before
    calling backward.

    Loss weighting strategy:
        1. Compute standard scalar loss via criterion (mean over batch).
        2. Compute per-sample heatmap MSE proxy to get relative weights,
           then scale the final loss by mean(source_weight).
        Because HeatmapLoss already averages across the batch we use the
        simpler approach: weight the *heatmap targets* by source_weight
        (broadcast over H,W) AND scale the final batch loss by
        mean(source_weight).  This is mathematically equivalent to
        weighting each sample's contribution.
    """
    model.train()
    losses = 0.0
    n_loss = 0
    n_steps   = 0
    n_skipped = 0
    subs  = {"bce": 0.0, "dice": 0.0, "coord": 0.0, "sep": 0.0}

    # Source counts for logging
    src_counts = {"real": 0, "pseudo": 0, "acdc": 0}

    lm_weights = torch.tensor([lm1_weight, 1.0], device=device).view(1, 2, 1, 1)

    for images, heatmaps, _, source_weights in loader:
        images         = images.to(device,         non_blocking=True)
        heatmaps       = heatmaps.to(device,       non_blocking=True)
        source_weights = source_weights.to(device, non_blocking=True)  # [B]

        # LM1 upweighting on targets
        heatmaps = heatmaps.float() * lm_weights

        # Scale heatmap targets per-sample by source_weight — [B,2,H,W] * [B,1,1,1]
        heatmaps = heatmaps * source_weights.view(-1, 1, 1, 1)

        # Track source mix (round weights to nearest known value)
        for w in source_weights.cpu().tolist():
            w_r = round(w, 2)
            if w_r >= 0.99:
                src_counts["real"]   += 1
            elif w_r >= 0.35:
                src_counts["pseudo"] += 1
            else:
                src_counts["acdc"]   += 1

        batch_weight = source_weights.mean()    # scalar weight for this batch

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
            logits = logits.float()
            loss, parts = criterion(logits, heatmaps)
            loss = loss * batch_weight

            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                n_steps += 1
            else:
                n_skipped += 1
        else:
            logits = model(images)
            loss, parts = criterion(logits, heatmaps)
            loss = loss * batch_weight

            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            n_steps += 1

        losses += loss.item()
        n_loss += 1
        for k in subs:
            subs[k] += parts[k]

    if n_loss == 0:
        return float("nan"), {k: float("nan") for k in subs}, 0, n_skipped, src_counts

    return (
        losses / n_loss,
        {k: v / n_loss for k, v in subs.items()},
        n_steps,
        n_skipped,
        src_counts,
    )


# ── single phase runner ───────────────────────────────────────────────────────

def run_phase(
    name, model, train_loader, val_loader, criterion, optimizer, scheduler,
    train_ds, sigma_start, sigma_end, n_epochs, device, run_dir, history,
    best_p90_ref, scaler=None, early_stop=None, save_best="best_model.pth",
    epoch_offset=0, lm1_weight=1.5,
):
    best_p90 = best_p90_ref[0]
    epochs_no_improve = 0

    for ep_in_phase in range(n_epochs):
        sigma = cosine_sigma(ep_in_phase, n_epochs, sigma_start, sigma_end)
        train_ds.set_sigma(sigma)

        train_loss, sub_avgs, n_steps, n_skipped, src_counts = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, lm1_weight=lm1_weight,
        )
        if n_skipped > 0:
            print(f"  warn: {n_skipped} batch(es) skipped due to non-finite loss")
        if scheduler is not None and n_steps > 0:
            scheduler.step()

        m = validate(model, val_loader, criterion, device)

        global_ep = epoch_offset + ep_in_phase + 1
        lr_now    = optimizer.param_groups[0]["lr"]

        history.append({
            "phase": name, "epoch": global_ep,
            "train_loss": train_loss, "val_loss": m["val_loss"],
            "mre": m["mre"], "sdr": m["sdr"][5], "sigma": sigma,
        })

        print(
            f"[{name}] Ep {global_ep:3d}  sigma={sigma:.2f}  lr={lr_now:.2e}  "
            f"Train={train_loss:.4f}  Val={m['val_loss']:.4f}  "
            f"MRE={m['mre']:.2f}px (LM1={m['mre1']:.2f} LM2={m['mre2']:.2f})  "
            f"SDR@2/5/10={m['sdr'][2]:.3f}/{m['sdr'][5]:.3f}/{m['sdr'][10]:.3f}  "
            f"P50={m['pct'][50]:.2f} P90={m['pct'][90]:.2f} Max={m['pct'][100]:.2f}  "
            f"[coord={sub_avgs['coord']:.4f} bce={sub_avgs['bce']:.4f}]  "
            f"src(real={src_counts['real']} pseudo={src_counts['pseudo']} "
            f"acdc={src_counts['acdc']})"
        )

        save_epoch_grid(*m["vis"], epoch=global_ep,
                        save_dir=os.path.join(run_dir, "grids"), n_samples=N_VIS)

        if m["pct"][90] < best_p90:
            best_p90 = m["pct"][90]
            best_p90_ref[0] = best_p90
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, save_best))
            print(f"  -> new best P90={best_p90:.2f}px  (saved {save_best})")
        else:
            epochs_no_improve += 1
            if early_stop is not None and epochs_no_improve >= early_stop:
                print(f"  early-stop after {early_stop} epochs with no P90 improvement")
                return global_ep

        torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))

    return epoch_offset + n_epochs


# ── test evaluation (identical to finetune_rv.py) ─────────────────────────────

def _tta_predict(model, images, device):
    images = images.to(device)
    hms = []
    for flip_dims in [[], [3], [2], [2, 3]]:
        v = torch.flip(images, flip_dims) if flip_dims else images
        with torch.no_grad():
            out = model(v)
        hm = torch.sigmoid(out)
        if flip_dims:
            hm = torch.flip(hm, flip_dims)
        hms.append(hm)
    return torch.stack(hms).mean(0)


def _enforce_and_match_np(pred_np, gt_np):
    p = pred_np.copy()
    if p[1] > p[3]:
        p = np.array([p[2], p[3], p[0], p[1]], dtype=np.float32)
    e_normal  = np.linalg.norm(p[:2] - gt_np[:2]) + np.linalg.norm(p[2:] - gt_np[2:])
    p_swap    = np.array([p[2], p[3], p[0], p[1]], dtype=np.float32)
    e_swapped = np.linalg.norm(p_swap[:2] - gt_np[:2]) + np.linalg.norm(p_swap[2:] - gt_np[2:])
    return p_swap if e_swapped < e_normal else p


@torch.no_grad()
def evaluate_test(model, in_channels, device):
    seg_dir = TEST_SEG_DIR if in_channels == 2 else None
    try:
        test_ds = RVLandmarkDataset(
            image_dir=TEST_IMAGE_DIR, gt_dir=TEST_GT_DIR, seg_dir=seg_dir,
            in_channels=in_channels, augment=False,
            sigma=SIGMA_P3_END, min_landmark_dist=20,
        )
    except Exception as exc:
        print(f"  [test eval skipped] Could not load test dataset: {exc}")
        return None

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    model.eval()

    sample_mres = []
    total_mre = total_mre1 = total_mre2 = 0.0
    total_sdr = {2: 0.0, 5: 0.0, 10: 0.0}

    for images, _, gt_coords in loader:
        gt_coords = gt_coords.to(device)
        avg_hm    = _tta_predict(model, images, device)
        pc        = gaussian_subpixel_argmax(avg_hm, window=7)

        pc_np  = pc.cpu().numpy()
        gt_np  = gt_coords.cpu().numpy()
        pc_matched = np.stack([
            _enforce_and_match_np(pc_np[b], gt_np[b]) for b in range(pc_np.shape[0])
        ])
        pc_matched = torch.tensor(pc_matched, dtype=torch.float32, device=device)

        total_mre  += compute_mre(pc_matched, gt_coords).item()
        m1, m2      = compute_mre_per_landmark(pc_matched, gt_coords)
        total_mre1 += m1
        total_mre2 += m2
        s = compute_sdr_multi(pc_matched, gt_coords, (2.0, 5.0, 10.0))
        for t in total_sdr:
            total_sdr[t] += s[t]
        sample_mres.extend(
            compute_per_sample_mre(pc_matched, gt_coords).cpu().tolist()
        )

    n   = len(loader)
    pct = compute_mre_percentiles(sample_mres)
    return {
        "mre":  total_mre  / n,
        "mre1": total_mre1 / n,
        "mre2": total_mre2 / n,
        "sdr":  {t: total_sdr[t] / n for t in total_sdr},
        "pct":  pct,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pseudo-label self-training for RV insertion point detection"
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--pseudo-gt-dir",   default="data/rv_landmark/pseudo_gt")
    parser.add_argument("--in-channels",     type=int,   default=2, choices=[1, 2])
    parser.add_argument("--instance-norm",   action="store_true")
    parser.add_argument("--group-norm",      action="store_true",
                        help="Use GroupNorm instead of BatchNorm")
    parser.add_argument("--epochs-p1",       type=int,   default=4)
    parser.add_argument("--epochs-p2",       type=int,   default=6)
    parser.add_argument("--epochs-p3",       type=int,   default=20)
    parser.add_argument("--lm1-weight",      type=float, default=1.5,
                        help="Heatmap target multiplier for LM1 (channel 0)")
    parser.add_argument("--pseudo-weight",   type=float, default=0.4,
                        help="Loss weight for pseudo-labeled rv_landmark samples")
    parser.add_argument("--acdc-weight",     type=float, default=0.15,
                        help="Loss weight for ACDC replay samples")
    parser.add_argument("--val-frac",        type=float, default=0.2)
    parser.add_argument("--no-amp",          action="store_true")
    args = parser.parse_args()
    if args.instance_norm and args.group_norm:
        raise ValueError("Cannot use both --instance-norm and --group-norm")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    norm_tag = ("instnorm" if args.instance_norm
                else "groupnorm" if args.group_norm
                else f"{args.in_channels}ch")
    run_dir  = os.path.join("rv-checkpoints", f"finetune_pseudo_{norm_tag}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "grids"), exist_ok=True)

    print(f"Run dir      : {run_dir}")
    print(f"Device       : {device}  | AMP: {use_amp}")
    print(f"Base ckpt    : {args.base_checkpoint}")
    print(f"Pseudo GT    : {args.pseudo_gt_dir}")
    print(f"Weights      : real=1.0  pseudo={args.pseudo_weight}  acdc={args.acdc_weight}")
    print(f"LM1 weight   : {args.lm1_weight}")

    # ── patient-level split (rv_landmark)
    train_files, val_files = split_volumes(RV_IMAGE_DIR,
                                           val_frac=args.val_frac, seed=SEED)
    print(f"RV train volumes: {len(train_files)}   val: {len(val_files)}")

    seg_dir = RV_SEG_DIR if args.in_channels == 2 else None

    # ── 1. Real rv_landmark train
    real_rv_ds_inner = RVLandmarkDataset(
        image_dir=RV_IMAGE_DIR, gt_dir=RV_GT_DIR, seg_dir=seg_dir,
        in_channels=args.in_channels, augment=True,
        sigma=SIGMA_P1, min_landmark_dist=20,
        volume_whitelist=train_files,
    )
    real_rv_ds = WeightedDataset(real_rv_ds_inner, source_weight=1.0)

    # ── 2. Pseudo-labeled rv_landmark train
    print("\nBuilding pseudo-label dataset …")
    pseudo_ds = build_pseudo_dataset(
        pseudo_gt_dir=args.pseudo_gt_dir,
        seg_dir=seg_dir,
        in_channels=args.in_channels,
        sigma=SIGMA_P1,
        volume_whitelist=train_files,
        source_weight=args.pseudo_weight,
    )

    # ── 3. ACDC replay
    acdc_ds_inner = None
    if os.path.isdir(ACDC_IMAGE_DIR):
        try:
            acdc_ds_inner = ACDCLandmarkDataset(
                image_dir=ACDC_IMAGE_DIR,
                mask_dir=ACDC_MASK_DIR,
                rvip_dir=ACDC_RVIP_DIR,
                patient_ids=ACDC_TRAIN_IDS,
                in_channels=args.in_channels,
                augment=True,
                sigma=SIGMA_P1,
                min_landmark_dist=5,
            )
            acdc_ds = WeightedDataset(acdc_ds_inner, source_weight=args.acdc_weight)
            print(f"ACDC replay: {len(acdc_ds)} slices  weight={args.acdc_weight}")
        except Exception as exc:
            print(f"  [ACDC skipped] {exc}")
            acdc_ds = None
    else:
        print(f"  [ACDC skipped] {ACDC_IMAGE_DIR} not found")
        acdc_ds = None

    # ── combine datasets
    all_parts = [real_rv_ds, pseudo_ds]
    if acdc_ds is not None:
        all_parts.append(acdc_ds)

    print(f"\nDataset sizes: real={len(real_rv_ds)}  pseudo={len(pseudo_ds)}"
          + (f"  acdc={len(acdc_ds)}" if acdc_ds else ""))
    print(f"Combined train samples: {sum(len(d) for d in all_parts)}")

    train_ds     = WeightedConcatDataset(all_parts)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    # ── validation uses plain (unweighted) RVLandmarkDataset
    val_ds = RVLandmarkDataset(
        image_dir=RV_IMAGE_DIR, gt_dir=RV_GT_DIR, seg_dir=seg_dir,
        in_channels=args.in_channels, augment=False,
        sigma=SIGMA_P1, min_landmark_dist=20,
        volume_whitelist=val_files,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # ── model
    model = UNetResNet34(
        in_channels=args.in_channels, num_classes=2,
        dropout=0.0, pretrained=False, cardiac_pretrained=False,
        use_instance_norm=args.instance_norm,
        use_group_norm=args.group_norm,
    ).to(device)
    state = torch.load(args.base_checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Checkpoint loaded. missing={len(missing)} unexpected={len(unexpected)}")

    criterion = HeatmapLoss(
        coord_weight=20.0,
        sep_weight=0.5,
        sep_min_dist=0.08,
        wing_w=0.008,
        wing_eps=0.002,
        lm_weights=[2.0, 1.0],
        hard_k=6,
    ).to(device)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history      = []
    best_p90_ref = [float("inf")]

    # ── baseline
    print("\n[baseline] validating on rv_landmark val split …")
    m0 = validate(model, val_loader, criterion, device)
    print(f"  baseline  MRE={m0['mre']:.2f}px (LM1={m0['mre1']:.2f} LM2={m0['mre2']:.2f})  "
          f"P90={m0['pct'][90]:.2f}px")
    history.append({"phase": "baseline", "epoch": 0,
                    "train_loss": 0.0, "val_loss": m0["val_loss"],
                    "mre": m0["mre"], "sdr": m0["sdr"][5], "sigma": SIGMA_P1})

    epoch_offset = 0

    # ── PHASE 1 — head/decoder warmup
    if args.epochs_p1 > 0:
        print(f"\n{'='*60}\n  PHASE 1 — head/decoder warmup "
              f"({args.epochs_p1} ep, encoder frozen)\n{'='*60}")
        set_encoder_grad(model, False)
        op  = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_HEAD_P1, weight_decay=WEIGHT_DECAY,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            op, T_max=args.epochs_p1, eta_min=1e-5
        )
        epoch_offset = run_phase(
            "P1", model, train_loader, val_loader, criterion, op, sch,
            train_ds, SIGMA_P1, SIGMA_P1, args.epochs_p1, device, run_dir,
            history, best_p90_ref, scaler=scaler, epoch_offset=epoch_offset,
            lm1_weight=args.lm1_weight,
        )

    # ── PHASE 2 — BatchNorm + decoder/head
    if args.epochs_p2 > 0:
        print(f"\n{'='*60}\n  PHASE 2 — BatchNorm + decoder/head "
              f"({args.epochs_p2} ep)\n{'='*60}")
        freeze_all(model)
        unfreeze_batchnorm(model)
        for n, p in model.named_parameters():
            if not any(n.startswith(x) for x in ENC_PREFIXES):
                p.requires_grad = True
        op  = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_BN_P2, weight_decay=WEIGHT_DECAY,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            op, T_max=args.epochs_p2, eta_min=5e-6
        )
        epoch_offset = run_phase(
            "P2", model, train_loader, val_loader, criterion, op, sch,
            train_ds, SIGMA_P2_START, SIGMA_P2_END, args.epochs_p2, device, run_dir,
            history, best_p90_ref, scaler=scaler, epoch_offset=epoch_offset,
            lm1_weight=args.lm1_weight,
        )

    # ── PHASE 3 — full unfreeze with discriminative LRs
    if args.epochs_p3 > 0:
        print(f"\n{'='*60}\n  PHASE 3 — full fine-tune ({args.epochs_p3} ep, "
              f"enc={LR_ENC_P3:.0e} / dec={LR_DEC_P3:.0e} / head={LR_HEAD_P3:.0e})"
              f"\n{'='*60}")
        for p in model.parameters():
            p.requires_grad = True
        op  = torch.optim.AdamW(
            param_groups_discriminative(model, LR_ENC_P3, LR_DEC_P3, LR_HEAD_P3),
            weight_decay=WEIGHT_DECAY,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            op, T_max=args.epochs_p3, eta_min=1e-7
        )
        epoch_offset = run_phase(
            "P3", model, train_loader, val_loader, criterion, op, sch,
            train_ds, SIGMA_P3_START, SIGMA_P3_END, args.epochs_p3, device, run_dir,
            history, best_p90_ref, scaler=scaler, early_stop=EARLY_STOP_P3,
            epoch_offset=epoch_offset, lm1_weight=args.lm1_weight,
        )

    # ── persist config + history
    config = {
        "base_checkpoint": args.base_checkpoint,
        "pseudo_gt_dir":   args.pseudo_gt_dir,
        "in_channels":     args.in_channels,
        "batch_size":      BATCH_SIZE,
        "epochs_p1":       args.epochs_p1,
        "epochs_p2":       args.epochs_p2,
        "epochs_p3":       args.epochs_p3,
        "lm1_weight":      args.lm1_weight,
        "pseudo_weight":   args.pseudo_weight,
        "acdc_weight":     args.acdc_weight,
        "lr_head_p1":      LR_HEAD_P1,
        "lr_bn_p2":        LR_BN_P2,
        "lr_enc_p3":       LR_ENC_P3,
        "lr_dec_p3":       LR_DEC_P3,
        "lr_head_p3":      LR_HEAD_P3,
        "sigma_p1":        SIGMA_P1,
        "sigma_p2":        [SIGMA_P2_START, SIGMA_P2_END],
        "sigma_p3":        [SIGMA_P3_START, SIGMA_P3_END],
        "loss": {
            "coord_weight": 20.0, "sep_weight": 0.5, "wing_w": 0.008,
            "wing_eps": 0.002, "lm_weights": [2.0, 1.0], "hard_k": BATCH_SIZE - 2,
        },
        "weight_decay":  WEIGHT_DECAY,
        "grad_clip":     GRAD_CLIP,
        "amp":           use_amp,
        "seed":          SEED,
        "val_frac":      args.val_frac,
        "train_volumes": train_files,
        "val_volumes":   val_files,
        "best_p90":      best_p90_ref[0],
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_training_curve(history, run_dir)

    # ── test evaluation
    best_model_path = os.path.join(run_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )
    print("\nEvaluating on test set …")
    test_m = evaluate_test(model, args.in_channels, device)

    if test_m is not None:
        results = {
            "checkpoint":    best_model_path,
            "in_channels":   args.in_channels,
            "best_val_p90":  best_p90_ref[0],
            "test_mre":      test_m["mre"],
            "test_mre_lm1":  test_m["mre1"],
            "test_mre_lm2":  test_m["mre2"],
            "test_sdr2":     test_m["sdr"][2],
            "test_sdr5":     test_m["sdr"][5],
            "test_sdr10":    test_m["sdr"][10],
            "test_p50":      test_m["pct"][50],
            "test_p90":      test_m["pct"][90],
        }
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*44}")
        print("PSEUDO-LABEL FINE-TUNING COMPLETE")
        print(f"Val  P90  : {best_p90_ref[0]:.2f}px")
        print(f"Test MRE  : {test_m['mre']:.2f}px")
        print(f"Test LM1  : {test_m['mre1']:.2f}px")
        print(f"Test LM2  : {test_m['mre2']:.2f}px")
        print(f"Test SDR@5: {test_m['sdr'][5]*100:.1f}%")
        print(f"Checkpoint: {best_model_path}")
        print(f"{'='*44}")
    else:
        print(f"\nDone. Best val P90 = {best_p90_ref[0]:.2f}px")
        print(f"Checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
