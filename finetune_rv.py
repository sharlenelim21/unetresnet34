"""
finetune_rv.py — Cross-domain fine-tuning ACDC -> rv_landmark
=============================================================
Starts from an ACDC checkpoint and fine-tunes on rv_landmark training
data using a 3-phase curriculum:

  Phase 1 — Decoder/head warmup       (encoder frozen)
  Phase 2 — BatchNorm + decoder/head  (conv weights frozen; BN stats track new domain)
  Phase 3 — Full unfreeze, discriminative LRs (encoder LR very small)

Validation uses a patient-level 80/20 split of the train set.
Test set (data/rv_landmark/test_*) is NEVER touched.

Checkpoint criterion: P90 MRE on validation set.

Usage:
    python finetune_rv.py \
        --base-checkpoint checkpoints/acdc_2ch_2026-05-12_00-48-48/best_model.pth \
        --in-channels 2 \
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
from torch.utils.data import DataLoader

from dataset.rv_landmark_dataset import RVLandmarkDataset, split_volumes
from models.unet_resnet34 import UNetResNet34
from utils.loss import HeatmapLoss
from utils.postprocess import gaussian_subpixel_argmax
from utils.metrics import (
    compute_mre, compute_mre_per_landmark,
    compute_sdr_multi, compute_per_sample_mre, compute_mre_percentiles,
)
from utils.visualize import save_epoch_grid, save_training_curve


# ── data paths ────────────────────────────────────────────────────────────────
IMAGE_DIR = "data/rv_landmark/train_images"
GT_DIR    = "data/rv_landmark/train_gt"
SEG_DIR   = "data/rv_landmark/train_seg_multi"

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
LR_BN_P2     = 5e-4
SIGMA_P2_START = 6.0
SIGMA_P2_END   = 3.0

# Phase 3 — full fine-tune
LR_ENC_P3    = 5e-6
LR_DEC_P3    = 5e-5
LR_HEAD_P3   = 1e-4
SIGMA_P3_START = 3.0
SIGMA_P3_END   = 1.5
EARLY_STOP_P3 = 8

ENC_PREFIXES = ("enc0", "enc1", "enc2", "enc3", "enc4")


# ── helpers ───────────────────────────────────────────────────────────────────

def set_encoder_grad(model, flag):
    for n, p in model.named_parameters():
        if any(n.startswith(x) for x in ENC_PREFIXES):
            p.requires_grad = flag


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_batchnorm(model):
    """Make every BatchNorm2d trainable; leave conv weights frozen."""
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
    """coords: [B, 4] tensor. Swap so y1 < y2 within each row."""
    out = coords.clone()
    swap = out[:, 1] > out[:, 3]
    if swap.any():
        tmp = out[swap].clone()
        out[swap, 0] = tmp[:, 2]
        out[swap, 1] = tmp[:, 3]
        out[swap, 2] = tmp[:, 0]
        out[swap, 3] = tmp[:, 1]
    return out


# ── validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, n_vis=N_VIS):
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
    """
    Train one epoch. Mixed precision is used only for the model forward;
    the loss (which contains a tight Wing loss with very small wing_w/eps)
    is computed in fp32 to avoid log()/divide overflow in fp16.

    Returns (avg_loss, sub_avgs, n_steps) where n_steps counts iterations
    where the optimizer actually stepped. The scheduler should only be
    advanced when n_steps > 0 (avoids the "scheduler before optimizer"
    warning when AMP skips a step due to inf/nan gradients).
    """
    model.train()
    losses = 0.0
    n_loss = 0
    n_steps = 0
    n_skipped = 0
    subs   = {"bce": 0.0, "dice": 0.0, "coord": 0.0, "sep": 0.0}

    for images, heatmaps, _ in loader:
        images   = images.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            # forward in fp16 …
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
            # … but compute loss in fp32 (tight wing_w/wing_eps overflow fp16)
            logits = logits.float()
            loss, parts = criterion(logits, heatmaps)

            if not torch.isfinite(loss):
                n_skipped += 1
                # don't backward / step on NaN/Inf — just skip this batch
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # scaler skips the optimizer step internally if grads contained inf/nan;
            # the only signal is the scale decreasing.
            if scaler.get_scale() >= scale_before:
                n_steps += 1
            else:
                n_skipped += 1
        else:
            logits = model(images)
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

    return (
        losses / n_loss,
        {k: v / n_loss for k, v in subs.items()},
        n_steps,
        n_skipped,
    )


# ── single phase runner ───────────────────────────────────────────────────────

def run_phase(
    name, model, train_loader, val_loader, criterion, optimizer, scheduler,
    train_ds, sigma_start, sigma_end, n_epochs, device, run_dir, history,
    best_p90_ref, scaler=None, early_stop=None, save_best="best_model.pth",
    epoch_offset=0,
):
    best_p90 = best_p90_ref[0]
    epochs_no_improve = 0

    for ep_in_phase in range(n_epochs):
        sigma = cosine_sigma(ep_in_phase, n_epochs, sigma_start, sigma_end)
        train_ds.set_sigma(sigma)

        train_loss, sub_avgs, n_steps, n_skipped = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler
        )
        if n_skipped > 0:
            print(f"  warn: {n_skipped} batch(es) skipped due to non-finite loss / inf grads")
        # only advance the scheduler if the optimizer actually stepped this epoch
        if scheduler is not None and n_steps > 0:
            scheduler.step()

        m = validate(model, val_loader, criterion, device)

        global_ep = epoch_offset + ep_in_phase + 1
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "phase": name, "epoch": global_ep,
            "train_loss": train_loss, "val_loss": m["val_loss"],
            "mre": m["mre"], "sdr": m["sdr"][5], "sigma": sigma,
        })

        print(
            f"[{name}] Ep {global_ep:3d}  sigma={sigma:.2f}  lr={lr_now:.2e}  "
            f"Train={train_loss:.4f}  Val={m['val_loss']:.4f}  "
            f"MRE={m['mre']:.2f}px (LM1={m['mre1']:.2f} LM2={m['mre2']:.2f})  "
            f"SDR@2/5/10 = {m['sdr'][2]:.3f}/{m['sdr'][5]:.3f}/{m['sdr'][10]:.3f}  "
            f"P50={m['pct'][50]:.2f} P90={m['pct'][90]:.2f} Max={m['pct'][100]:.2f}  "
            f"[coord={sub_avgs['coord']:.4f} bce={sub_avgs['bce']:.4f}]"
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True,
                        help="ACDC pretrained checkpoint (.pth) to start from")
    parser.add_argument("--in-channels", type=int, default=2, choices=[1, 2])
    parser.add_argument("--epochs-p1", type=int, default=4)
    parser.add_argument("--epochs-p2", type=int, default=6)
    parser.add_argument("--epochs-p3", type=int, default=15)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (slower but more stable)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("checkpoints",
                           f"finetune_rv_{args.in_channels}ch_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "grids"), exist_ok=True)
    print(f"Run dir : {run_dir}")
    print(f"Device  : {device}  | AMP: {use_amp}")
    print(f"Base ckpt: {args.base_checkpoint}")

    # ── patient-level split
    train_files, val_files = split_volumes(IMAGE_DIR, val_frac=args.val_frac, seed=SEED)
    print(f"Train volumes: {len(train_files)}   Val volumes: {len(val_files)}")

    seg_dir = SEG_DIR if args.in_channels == 2 else None

    train_ds = RVLandmarkDataset(
        image_dir=IMAGE_DIR, gt_dir=GT_DIR, seg_dir=seg_dir,
        in_channels=args.in_channels, augment=True,
        sigma=SIGMA_P1, min_landmark_dist=20,
        volume_whitelist=train_files,
    )
    val_ds = RVLandmarkDataset(
        image_dir=IMAGE_DIR, gt_dir=GT_DIR, seg_dir=seg_dir,
        in_channels=args.in_channels, augment=False,
        sigma=SIGMA_P1, min_landmark_dist=20,
        volume_whitelist=val_files,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── model
    model = UNetResNet34(in_channels=args.in_channels, num_classes=2,
                         dropout=0.0, pretrained=False, cardiac_pretrained=False).to(device)
    state = torch.load(args.base_checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Checkpoint loaded. missing={len(missing)} unexpected={len(unexpected)}")

    criterion = HeatmapLoss(
        coord_weight=20.0,
        sep_weight=0.5,
        sep_min_dist=0.08,
        wing_w=0.008,
        wing_eps=0.002,
        # Fix 3 + Fix 4: per-channel weighted BCE / Dice / Wing.
        # [1.5, 1.0] is a softer ratio than the previous [2.5, 1.0] run —
        # that run improved LM1 (25.92->19.68px) but overcorrected on LM2
        # (10.49->19.48px). 1.5:1 nudges LM1 without starving LM2's gradient.
        channel_weights=[1.5, 1.0],
    ).to(device)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = []
    best_p90_ref = [float("inf")]

    # ── baseline validation before any training
    print("\n[baseline] validating ACDC checkpoint on rv_landmark val split")
    m0 = validate(model, val_loader, criterion, device)
    print(f"  baseline   MRE={m0['mre']:.2f}px (LM1={m0['mre1']:.2f} LM2={m0['mre2']:.2f})  "
          f"SDR@2/5/10 = {m0['sdr'][2]:.3f}/{m0['sdr'][5]:.3f}/{m0['sdr'][10]:.3f}  "
          f"P90={m0['pct'][90]:.2f}px")
    history.append({"phase": "baseline", "epoch": 0,
                    "train_loss": 0.0, "val_loss": m0["val_loss"],
                    "mre": m0["mre"], "sdr": m0["sdr"][5], "sigma": SIGMA_P1})

    epoch_offset = 0

    # ── PHASE 1 — head/decoder warmup
    if args.epochs_p1 > 0:
        print(f"\n{'='*60}\n  PHASE 1 — head/decoder warmup ({args.epochs_p1} ep, encoder frozen)\n{'='*60}")
        set_encoder_grad(model, False)
        op = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=LR_HEAD_P1, weight_decay=WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=args.epochs_p1, eta_min=1e-5)
        epoch_offset = run_phase(
            "P1", model, train_loader, val_loader, criterion, op, sch,
            train_ds, SIGMA_P1, SIGMA_P1, args.epochs_p1, device, run_dir,
            history, best_p90_ref, scaler=scaler, epoch_offset=epoch_offset,
        )

    # ── PHASE 2 — BatchNorm + decoder/head
    if args.epochs_p2 > 0:
        print(f"\n{'='*60}\n  PHASE 2 — BatchNorm + decoder/head ({args.epochs_p2} ep)\n{'='*60}")
        freeze_all(model)
        unfreeze_batchnorm(model)
        # also unfreeze decoder + heads (encoder convs stay frozen)
        for n, p in model.named_parameters():
            if not any(n.startswith(x) for x in ENC_PREFIXES):
                p.requires_grad = True
        op = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=LR_BN_P2, weight_decay=WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=args.epochs_p2, eta_min=5e-6)
        epoch_offset = run_phase(
            "P2", model, train_loader, val_loader, criterion, op, sch,
            train_ds, SIGMA_P2_START, SIGMA_P2_END, args.epochs_p2, device, run_dir,
            history, best_p90_ref, scaler=scaler, epoch_offset=epoch_offset,
        )

    # ── PHASE 3 — full unfreeze with discriminative LRs
    if args.epochs_p3 > 0:
        print(f"\n{'='*60}\n  PHASE 3 — full fine-tune ({args.epochs_p3} ep, "
              f"enc={LR_ENC_P3:.0e} / dec={LR_DEC_P3:.0e} / head={LR_HEAD_P3:.0e})\n{'='*60}")
        for p in model.parameters():
            p.requires_grad = True
        op = torch.optim.AdamW(
            param_groups_discriminative(model, LR_ENC_P3, LR_DEC_P3, LR_HEAD_P3),
            weight_decay=WEIGHT_DECAY,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=args.epochs_p3, eta_min=1e-7)
        epoch_offset = run_phase(
            "P3", model, train_loader, val_loader, criterion, op, sch,
            train_ds, SIGMA_P3_START, SIGMA_P3_END, args.epochs_p3, device, run_dir,
            history, best_p90_ref, scaler=scaler, early_stop=EARLY_STOP_P3,
            epoch_offset=epoch_offset,
        )

    # ── persist config + history + curve
    config = {
        "base_checkpoint": args.base_checkpoint,
        "in_channels": args.in_channels,
        "batch_size": BATCH_SIZE,
        "epochs_p1": args.epochs_p1,
        "epochs_p2": args.epochs_p2,
        "epochs_p3": args.epochs_p3,
        "lr_head_p1": LR_HEAD_P1,
        "lr_bn_p2":  LR_BN_P2,
        "lr_enc_p3": LR_ENC_P3,
        "lr_dec_p3": LR_DEC_P3,
        "lr_head_p3": LR_HEAD_P3,
        "sigma_p1": SIGMA_P1,
        "sigma_p2": [SIGMA_P2_START, SIGMA_P2_END],
        "sigma_p3": [SIGMA_P3_START, SIGMA_P3_END],
        "loss": {"coord_weight": 20.0, "sep_weight": 0.5, "wing_w": 0.008,
                 "wing_eps": 0.002, "lm_weights": [2.0, 1.0], "hard_k": BATCH_SIZE - 2},
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "amp": use_amp,
        "seed": SEED,
        "val_frac": args.val_frac,
        "train_volumes": train_files,
        "val_volumes":   val_files,
        "best_p90": best_p90_ref[0],
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_training_curve(history, run_dir)

    print(f"\nDone. Best P90 = {best_p90_ref[0]:.2f}px")
    print(f"Best checkpoint: {os.path.join(run_dir, 'best_model.pth')}")
    print(f"Evaluate on test set with:")
    print(f"  python inference_rv.py --checkpoint {os.path.join(run_dir, 'best_model.pth')} "
          f"--in-channels {args.in_channels}"
          + (f" --seg-dir data/rv_landmark/test_seg_multi" if args.in_channels == 2 else "")
          + " --eval")


if __name__ == "__main__":
    main()
