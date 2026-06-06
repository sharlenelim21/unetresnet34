"""
finetune_translated.py — Cross-domain fine-tuning with CycleGAN translated data
================================================================================
Trains on a 50/50 mix of:
  - Original ACDC data (patients 001-080)
  - CycleGAN-translated ACDC data (same patients, translated to rv_landmark domain)

Validates on original ACDC val split (patients 081-090).
Final test evaluation on rv_landmark test set.

3-phase curriculum (identical structure to finetune_rv.py):
  Phase 1 — Decoder/head warmup       (encoder frozen)
  Phase 2 — BatchNorm + decoder/head  (conv weights frozen; BN stats adapt)
  Phase 3 — Full unfreeze, discriminative LRs

Usage:
    python finetune_translated.py \
        --base-checkpoint acdc-checkpoints/acdc_1ch_.../best_model.pth \
        --in-channels 1 \
        --epochs-p1 4 --epochs-p2 6 --epochs-p3 15
"""

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import nibabel as nib
from scipy.ndimage import label as scipy_label, center_of_mass

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
IMAGE_DIR = "data/acdc/images"
MASK_DIR  = "data/acdc/masks"
RVIP_DIR  = "data/acdc/points"

TRANS_IMAGE_DIR = "data/acdc_translated/images"
TRANS_MASK_DIR  = "data/acdc_translated/masks"
TRANS_RVIP_DIR  = "data/acdc_translated/points"

TEST_IMAGE_DIR = "data/rv_landmark/test_images"
TEST_GT_DIR    = "data/rv_landmark/test_gt"

TRAIN_IDS = [f"patient{i:03d}" for i in range(1,  81)]
VAL_IDS   = [f"patient{i:03d}" for i in range(81, 91)]

# ── fixed hyperparameters ─────────────────────────────────────────────────────
BATCH_SIZE   = 8
SEED         = 42
NUM_WORKERS  = 2
N_VIS        = 8
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 2.0

# Phase 1 — head/decoder warmup
LR_HEAD_P1   = 1e-3
SIGMA_P1     = 6.0

# Phase 2 — BN + head
LR_BN_P2       = 5e-4
SIGMA_P2_START = 6.0
SIGMA_P2_END   = 3.0

# Phase 3 — full fine-tune
LR_ENC_P3      = 5e-6
LR_DEC_P3      = 5e-5
LR_HEAD_P3     = 1e-4
SIGMA_P3_START = 3.0
SIGMA_P3_END   = 1.5
EARLY_STOP_P3  = 8

ENC_PREFIXES = ("enc0", "enc1", "enc2", "enc3", "enc4")


# ── encoder freeze helpers ────────────────────────────────────────────────────

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


# ── ordering / matching ───────────────────────────────────────────────────────

def enforce_superior_ordering_batch(coords):
    """coords: [B, 4] tensor. Swap so y1 < y2 within each row."""
    out  = coords.clone()
    swap = out[:, 1] > out[:, 3]
    if swap.any():
        tmp = out[swap].clone()
        out[swap, 0] = tmp[:, 2]
        out[swap, 1] = tmp[:, 3]
        out[swap, 2] = tmp[:, 0]
        out[swap, 3] = tmp[:, 1]
    return out


def _enforce_and_match_np(pred_np, gt_np):
    """Enforce superior ordering then pick the lower-error assignment."""
    p = pred_np.copy()
    if p[1] > p[3]:
        p = np.array([p[2], p[3], p[0], p[1]], dtype=np.float32)
    g = gt_np
    e_normal  = np.linalg.norm(p[:2] - g[:2]) + np.linalg.norm(p[2:] - g[2:])
    p_swap    = np.array([p[2], p[3], p[0], p[1]], dtype=np.float32)
    e_swapped = np.linalg.norm(p_swap[:2] - g[:2]) + np.linalg.norm(p_swap[2:] - g[2:])
    return p_swap if e_swapped < e_normal else p


# ── validation (on ACDC val set) ─────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, n_vis=N_VIS):
    model.eval()
    val_loss   = 0.0
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

        logits     = model(images)
        loss, _    = criterion(logits, heatmaps)
        val_loss  += loss.item()

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
        sample_mres.extend(compute_per_sample_mre(pred_coords, gt_coords).cpu().tolist())

        if len(vis_images) < n_vis:
            for j in range(min(n_vis - len(vis_images), images.size(0))):
                vis_images.append(images[j, 0].cpu().numpy())
                vis_preds.append(pred_coords[j].cpu().numpy())
                vis_gts.append(gt_coords[j].cpu().numpy())

    n = len(loader)
    return {
        "val_loss": val_loss / n,
        "mre":      total_mre / n,
        "mre1":     total_mre1 / n,
        "mre2":     total_mre2 / n,
        "sdr":      {t: total_sdr[t] / n for t in total_sdr},
        "pct":      compute_mre_percentiles(sample_mres),
        "vis":      (vis_images, vis_preds, vis_gts),
    }


# ── train one epoch ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    losses    = 0.0
    n_loss    = 0
    n_steps   = 0
    n_skipped = 0
    subs      = {"bce": 0.0, "dice": 0.0, "coord": 0.0, "sep": 0.0}

    for images, heatmaps, _ in loader:
        images   = images.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
            logits       = logits.float()
            loss, parts  = criterion(logits, heatmaps)

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
            logits      = model(images)
            loss, parts = criterion(logits, heatmaps)
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
        return float("nan"), {k: float("nan") for k in subs}, 0, n_skipped

    return losses / n_loss, {k: v / n_loss for k, v in subs.items()}, n_steps, n_skipped


# ── phase runner ──────────────────────────────────────────────────────────────

def run_phase(
    name, model, train_loader, val_loader, criterion, optimizer, scheduler,
    train_ds_orig, train_ds_trans, sigma_start, sigma_end, n_epochs,
    device, run_dir, history, best_p90_ref, scaler=None,
    early_stop=None, save_best="best_model.pth", epoch_offset=0,
):
    best_p90 = best_p90_ref[0]
    no_improve = 0

    for ep_in_phase in range(n_epochs):
        sigma = cosine_sigma(ep_in_phase, n_epochs, sigma_start, sigma_end)
        train_ds_orig.set_sigma(sigma)
        train_ds_trans.set_sigma(sigma)

        train_loss, sub_avgs, n_steps, n_skipped = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler
        )
        if n_skipped > 0:
            print(f"  warn: {n_skipped} batch(es) skipped (non-finite loss / inf grads)")
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
            f"[coord={sub_avgs['coord']:.4f} bce={sub_avgs['bce']:.4f}]"
        )

        if m["pct"][90] < best_p90:
            best_p90 = m["pct"][90]
            best_p90_ref[0] = best_p90
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, save_best))
            save_epoch_grid(*m["vis"], epoch=global_ep,
                            save_dir=os.path.join(run_dir, "grids"), n_samples=N_VIS)
            print(f"  -> new best P90={best_p90:.2f}px  (saved {save_best})")
        else:
            no_improve += 1
            if early_stop is not None and no_improve >= early_stop:
                print(f"  early-stop after {early_stop} epochs with no P90 improvement")
                return global_ep

        torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))

    return epoch_offset + n_epochs


# ── rv_landmark test evaluation ───────────────────────────────────────────────

def _extract_rv_gt_coords(gt_slice):
    """
    Extract (x1,y1,x2,y2) from a rv_landmark GT slice (combined Gaussian heatmap).
    Finds two blob centroids via connected components, sorted by y.
    Returns float32 [4] or None if two blobs cannot be found.
    """
    mx = float(gt_slice.max())
    if mx <= 0:
        return None
    bw = gt_slice > 0.1 * mx
    lbl, n = scipy_label(bw)
    if n < 2:
        return None
    comps = []
    for k in range(1, n + 1):
        area = int((lbl == k).sum())
        if area < 3:
            continue
        cy, cx = center_of_mass(gt_slice, lbl, k)
        comps.append((float(cx), float(cy), area))
    if len(comps) < 2:
        return None
    comps.sort(key=lambda t: -t[2])          # two largest blobs
    (x1, y1, _), (x2, y2, _) = comps[0], comps[1]
    if y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _tta_predict(model, images, device):
    """4-variant TTA: original, hflip, vflip, hflip+vflip."""
    images = images.to(device)
    hms = []
    for flip_dims in [[], [3], [2], [2, 3]]:
        v  = torch.flip(images, flip_dims) if flip_dims else images
        with torch.no_grad():
            out = model(v)
        hm = torch.sigmoid(out)
        if flip_dims:
            hm = torch.flip(hm, flip_dims)
        hms.append(hm)
    return torch.stack(hms).mean(0)


@torch.no_grad()
def evaluate_rv_test(model, in_channels, device):
    """
    Evaluate on the rv_landmark test set.
    GT is loaded from TEST_GT_DIR as a combined-heatmap volume;
    two blob centroids are extracted per slice.
    """
    if not os.path.isdir(TEST_IMAGE_DIR) or not os.path.isdir(TEST_GT_DIR):
        print(f"  [test eval skipped] {TEST_IMAGE_DIR} or {TEST_GT_DIR} not found")
        return None

    test_files = sorted(
        f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(".nii.gz")
    )
    if not test_files:
        print("  [test eval skipped] no .nii.gz files found in test_images/")
        return None

    model.eval()
    all_pred, all_gt, sample_mres = [], [], []
    total_mre = total_mre1 = total_mre2 = 0.0
    total_sdr = {2: 0.0, 5: 0.0, 10: 0.0}
    n_batches = 0

    for fname in test_files:
        img_path = os.path.join(TEST_IMAGE_DIR, fname)
        gt_path  = os.path.join(TEST_GT_DIR,    fname)
        if not os.path.exists(gt_path):
            continue

        img_vol = nib.load(img_path).get_fdata().astype(np.float32)
        gt_vol  = nib.load(gt_path).get_fdata().astype(np.float32)
        if img_vol.ndim == 4:
            img_vol = img_vol[..., 0]
        if gt_vol.ndim == 4:
            gt_vol = gt_vol[..., 0]

        mu  = img_vol.mean()
        std = img_vol.std() + 1e-8
        n_slices = min(img_vol.shape[2], gt_vol.shape[2])

        for z in range(n_slices):
            gt_coords = _extract_rv_gt_coords(gt_vol[:, :, z])
            if gt_coords is None:
                continue

            img_2d = img_vol[:, :, z]
            import cv2
            img_r = cv2.resize(img_2d, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_r = ((img_r - mu) / std).astype(np.float32)

            if in_channels == 2:
                # No seg mask available for test — zero the second channel
                img_t = torch.zeros(1, 2, 256, 256, dtype=torch.float32)
                img_t[0, 0] = torch.from_numpy(img_r)
            else:
                img_t = torch.from_numpy(img_r).unsqueeze(0).unsqueeze(0)

            avg_hm = _tta_predict(model, img_t, device)
            pc     = gaussian_subpixel_argmax(avg_hm, window=7)

            pc_np  = pc[0].cpu().numpy()
            gt_np  = gt_coords
            pc_corr = _enforce_and_match_np(pc_np, gt_np)

            gt_t     = torch.tensor(gt_np,   dtype=torch.float32).unsqueeze(0)
            pc_t     = torch.tensor(pc_corr, dtype=torch.float32).unsqueeze(0)

            total_mre  += compute_mre(pc_t, gt_t).item()
            m1, m2      = compute_mre_per_landmark(pc_t, gt_t)
            total_mre1 += m1
            total_mre2 += m2
            s = compute_sdr_multi(pc_t, gt_t, (2.0, 5.0, 10.0))
            for t in total_sdr:
                total_sdr[t] += s[t]
            sample_mres.extend(compute_per_sample_mre(pc_t, gt_t).tolist())
            n_batches += 1

    if n_batches == 0:
        print("  [test eval] no annotated slices found in test set")
        return None

    pct = compute_mre_percentiles(sample_mres)
    return {
        "mre":  total_mre / n_batches,
        "mre1": total_mre1 / n_batches,
        "mre2": total_mre2 / n_batches,
        "sdr":  {t: total_sdr[t] / n_batches for t in total_sdr},
        "pct":  pct,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("rv-checkpoints",
                           f"finetune_translated_{args.in_channels}ch_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "grids"), exist_ok=True)
    print(f"Run dir  : {run_dir}")
    print(f"Device   : {device}  | AMP: {use_amp}")
    print(f"Base ckpt: {args.base_checkpoint}")
    print(f"Channels : {args.in_channels}")

    # ── datasets ──────────────────────────────────────────────────────────────
    # Original ACDC train
    train_ds_orig = ACDCLandmarkDataset(
        IMAGE_DIR, MASK_DIR, RVIP_DIR,
        patient_ids=TRAIN_IDS,
        in_channels=args.in_channels, augment=True, sigma=SIGMA_P1,
    )
    # CycleGAN-translated ACDC train (same patients)
    train_ds_trans = ACDCLandmarkDataset(
        TRANS_IMAGE_DIR, TRANS_MASK_DIR, TRANS_RVIP_DIR,
        patient_ids=TRAIN_IDS,
        in_channels=args.in_channels, augment=True, sigma=SIGMA_P1,
    )
    # ACDC val (original only)
    val_ds = ACDCLandmarkDataset(
        IMAGE_DIR, MASK_DIR, RVIP_DIR,
        patient_ids=VAL_IDS,
        in_channels=args.in_channels, augment=False, sigma=SIGMA_P1,
    )

    # 50/50 mix via ConcatDataset — DataLoader shuffles across both halves
    mixed_train_ds = ConcatDataset([train_ds_orig, train_ds_trans])
    print(f"Train slices: {len(train_ds_orig)} orig + {len(train_ds_trans)} translated "
          f"= {len(mixed_train_ds)} total")
    print(f"Val   slices: {len(val_ds)}")

    train_loader = DataLoader(
        mixed_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = UNetResNet34(
        in_channels=args.in_channels, num_classes=2,
        dropout=0.0, pretrained=False, cardiac_pretrained=False,
    ).to(device)
    state = torch.load(args.base_checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")

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

    # ── baseline ──────────────────────────────────────────────────────────────
    print("\n[baseline] validating loaded checkpoint on ACDC val split")
    m0 = validate(model, val_loader, criterion, device)
    print(f"  baseline  MRE={m0['mre']:.2f}px (LM1={m0['mre1']:.2f} LM2={m0['mre2']:.2f})  "
          f"SDR@2/5/10={m0['sdr'][2]:.3f}/{m0['sdr'][5]:.3f}/{m0['sdr'][10]:.3f}  "
          f"P90={m0['pct'][90]:.2f}px")
    history.append({"phase": "baseline", "epoch": 0,
                    "train_loss": 0.0, "val_loss": m0["val_loss"],
                    "mre": m0["mre"], "sdr": m0["sdr"][5], "sigma": SIGMA_P1})

    epoch_offset = 0

    # ── PHASE 1 — head/decoder warmup, encoder frozen ─────────────────────────
    if args.epochs_p1 > 0:
        print(f"\n{'='*60}\n  PHASE 1 — warmup ({args.epochs_p1} ep, encoder frozen)\n{'='*60}")
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
            train_ds_orig, train_ds_trans,
            SIGMA_P1, SIGMA_P1, args.epochs_p1,
            device, run_dir, history, best_p90_ref,
            scaler=scaler, epoch_offset=epoch_offset,
        )

    # ── PHASE 2 — BN + decoder/head, encoder conv frozen ─────────────────────
    if args.epochs_p2 > 0:
        print(f"\n{'='*60}\n  PHASE 2 — BN + decoder/head ({args.epochs_p2} ep)\n{'='*60}")
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
            train_ds_orig, train_ds_trans,
            SIGMA_P2_START, SIGMA_P2_END, args.epochs_p2,
            device, run_dir, history, best_p90_ref,
            scaler=scaler, epoch_offset=epoch_offset,
        )

    # ── PHASE 3 — full unfreeze, discriminative LRs ───────────────────────────
    if args.epochs_p3 > 0:
        print(f"\n{'='*60}\n  PHASE 3 — full fine-tune ({args.epochs_p3} ep, "
              f"enc={LR_ENC_P3:.0e}/dec={LR_DEC_P3:.0e}/head={LR_HEAD_P3:.0e})\n{'='*60}")
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
            train_ds_orig, train_ds_trans,
            SIGMA_P3_START, SIGMA_P3_END, args.epochs_p3,
            device, run_dir, history, best_p90_ref,
            scaler=scaler, early_stop=EARLY_STOP_P3,
            epoch_offset=epoch_offset,
        )

    # ── persist config + history + curve ──────────────────────────────────────
    config = {
        "base_checkpoint": args.base_checkpoint,
        "in_channels": args.in_channels,
        "train_orig":  IMAGE_DIR,
        "train_trans": TRANS_IMAGE_DIR,
        "val_ids":     VAL_IDS,
        "epochs_p1": args.epochs_p1,
        "epochs_p2": args.epochs_p2,
        "epochs_p3": args.epochs_p3,
        "lr_head_p1": LR_HEAD_P1,
        "lr_bn_p2":   LR_BN_P2,
        "lr_enc_p3":  LR_ENC_P3,
        "lr_dec_p3":  LR_DEC_P3,
        "lr_head_p3": LR_HEAD_P3,
        "sigma_p1": SIGMA_P1,
        "sigma_p2": [SIGMA_P2_START, SIGMA_P2_END],
        "sigma_p3": [SIGMA_P3_START, SIGMA_P3_END],
        "loss": {"coord_weight": 20.0, "sep_weight": 0.5,
                 "wing_w": 0.008, "wing_eps": 0.002,
                 "lm_weights": [2.0, 1.0], "hard_k": 6},
        "weight_decay": WEIGHT_DECAY,
        "grad_clip":   GRAD_CLIP,
        "amp":         use_amp,
        "seed":        SEED,
        "best_p90":    best_p90_ref[0],
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_training_curve(history, run_dir)

    # ── test evaluation on rv_landmark ────────────────────────────────────────
    best_model_path = os.path.join(run_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )
    print("\nEvaluating on rv_landmark test set …")
    test_m = evaluate_rv_test(model, args.in_channels, device)

    if test_m is not None:
        results = {
            "checkpoint":   best_model_path,
            "in_channels":  args.in_channels,
            "best_val_p90": best_p90_ref[0],
            "test_mre":     test_m["mre"],
            "test_mre_lm1": test_m["mre1"],
            "test_mre_lm2": test_m["mre2"],
            "test_sdr2":    test_m["sdr"][2],
            "test_sdr5":    test_m["sdr"][5],
            "test_sdr10":   test_m["sdr"][10],
            "test_p50":     test_m["pct"][50],
            "test_p90":     test_m["pct"][90],
        }
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*44}")
        print("FINE-TUNING COMPLETE (CycleGAN translated)")
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
    parser = argparse.ArgumentParser(
        description="Fine-tune on 50/50 mix of original + CycleGAN-translated ACDC"
    )
    parser.add_argument("--base-checkpoint", required=True,
                        help="Path to pretrained ACDC checkpoint (.pth)")
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 2])
    parser.add_argument("--epochs-p1",   type=int, default=4)
    parser.add_argument("--epochs-p2",   type=int, default=6)
    parser.add_argument("--epochs-p3",   type=int, default=15)
    parser.add_argument("--no-amp",      action="store_true",
                        help="Disable mixed precision (use if AMP causes NaN)")
    args = parser.parse_args()
    main(args)
