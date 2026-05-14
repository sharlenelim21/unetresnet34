"""
finetune_translated.py
======================
Fine-tunes the 2ch ACDC checkpoint on a MIXED dataset:
  - Original ACDC slices  (1075 slices, real scanner)
  - Translated ACDC slices (1075 slices, GAN-translated to rv_landmark style)

Both halves share the same RVIP annotations and seg masks.
The translated half teaches the model to handle rv_landmark scanner texture
while the original half prevents catastrophic forgetting of ACDC features.

A consistency loss encourages the model to produce similar heatmaps for a
real ACDC slice and its translated counterpart, acting as a soft invariance
constraint across the domain gap.

3-phase curriculum (mirrors finetune_rv.py structure):
  P1 — encoder frozen, head/decoder warmup, no consistency loss
  P2 — BN + decoder/head, encoder convs frozen, consistency lambda=0.3
  P3 — full unfreeze, discriminative LRs, consistency lambda=0.5

Validation: original ACDC 20% patient split (same as original training).
Checkpoint criterion: val P90 MRE.

Usage:
  python finetune_translated.py
  python finetune_translated.py --epochs-p1 6 --epochs-p2 10 --epochs-p3 20

After this runs:
  python finetune_rv.py \\
    --base-checkpoint checkpoints/finetune_translated_2ch_.../best_model.pth \\
    --in-channels 2 --epochs-p1 4 --epochs-p2 6 --epochs-p3 15
"""

import json
import math
import os
from datetime import datetime

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset

from dataset.acdc_landmark_dataset import ACDCLandmarkDataset
from models.unet_resnet34 import UNetResNet34
from utils.loss import HeatmapLoss
from utils.postprocess import gaussian_subpixel_argmax
from utils.metrics import (
    compute_mre, compute_mre_per_landmark,
    compute_sdr_multi, compute_per_sample_mre, compute_mre_percentiles,
)
from utils.visualize import save_epoch_grid, save_training_curve


# ── paths ─────────────────────────────────────────────────────────────────────
BASE_CHECKPOINT  = "checkpoints/acdc_2ch_2026-05-12_00-48-48/best_model.pth"

ACDC_IMG_DIR     = "data/acdc/images"
ACDC_MASK_DIR    = "data/acdc/masks"
ACDC_POINT_DIR   = "data/acdc/points"

TRANS_IMG_DIR    = "data/acdc_translated/images"
TRANS_MASK_DIR   = "data/acdc_translated/masks"
TRANS_POINT_DIR  = "data/acdc_translated/points"

# ── hyperparameters ───────────────────────────────────────────────────────────
IN_CHANNELS  = 2
BATCH_SIZE   = 8
SEED         = 42
NUM_WORKERS  = 2
N_VIS        = 8
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 2.0
VAL_SPLIT    = 0.2

# Phase 1
P1_EPOCHS   = 6
LR_P1       = 5e-4
SIGMA_P1    = 6.0
LAMBDA_C_P1 = 0.0   # no consistency yet — head not stable

# Phase 2
P2_EPOCHS        = 10
LR_P2            = 2e-4
SIGMA_P2_START   = 6.0
SIGMA_P2_END     = 3.0
LAMBDA_C_P2      = 0.3

# Phase 3
P3_EPOCHS        = 20
LR_ENC_P3        = 5e-6
LR_DEC_P3        = 5e-5
LR_HEAD_P3       = 1e-4
SIGMA_P3_START   = 3.0
SIGMA_P3_END     = 1.5
LAMBDA_C_P3      = 0.5
EARLY_STOP_P3    = 10

ENC_PREFIXES = ("enc0", "enc1", "enc2", "enc3", "enc4")


# ── paired dataset ────────────────────────────────────────────────────────────

class PairedTranslatedDataset(Dataset):
    """
    Wraps an ACDCLandmarkDataset built on original images.
    Each __getitem__ also loads the corresponding translated slice
    (same filename, same slice index, from TRANS_IMG_DIR).

    Returns:
      img_orig  [2, 256, 256]  — original ACDC MRI + seg
      img_trans [2, 256, 256]  — translated MRI + same seg
      heatmaps  [2, 256, 256]  — GT heatmaps (shared)
      coords    [4]            — GT coords (shared)

    If the translated volume file is missing, img_trans falls back to
    img_orig so training degrades gracefully rather than crashing.
    """

    def __init__(self, acdc_ds, trans_img_dir, trans_mask_dir):
        self.ds            = acdc_ds
        self.trans_img_dir = trans_img_dir
        self.trans_mask_dir = trans_mask_dir
        self._missing_warned = set()

    def set_sigma(self, sigma):
        self.ds.set_sigma(sigma)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        import nibabel as nib
        import cv2

        img_orig, heatmaps, coords = self.ds[idx]

        fname, slice_idx, _, H_orig, W_orig = self.ds.samples[idx]

        # ── load translated MRI slice ──────────────────────────────────────
        trans_img_path  = os.path.join(self.trans_img_dir,  fname)
        trans_mask_path = os.path.join(self.trans_mask_dir, fname)

        if not os.path.exists(trans_img_path):
            if fname not in self._missing_warned:
                print(f"  warn: translated file missing, using original: {fname}")
                self._missing_warned.add(fname)
            return img_orig, img_orig, heatmaps, coords

        try:
            t_img = nib.load(trans_img_path).get_fdata().astype(np.float32)
            t_sl  = np.take(t_img, slice_idx, axis=self.ds.slice_axis)
            t_sl  = cv2.resize(t_sl, (256, 256), interpolation=cv2.INTER_LINEAR)
            mu, std = t_sl.mean(), t_sl.std() + 1e-8
            t_sl  = (t_sl - mu) / std

            if os.path.exists(trans_mask_path):
                t_seg = np.round(
                    nib.load(trans_mask_path).get_fdata().astype(np.float32)
                )
                t_seg_sl = np.take(t_seg, slice_idx, axis=self.ds.slice_axis)
                t_seg_r  = cv2.resize(t_seg_sl, (256, 256),
                                      interpolation=cv2.INTER_NEAREST)
                if not np.any(np.round(t_seg_r) == 1):
                    t_seg_r = np.zeros_like(t_seg_r)
                else:
                    smax = t_seg_r.max()
                    t_seg_r = (t_seg_r / smax).astype(np.float32) if smax > 0 else np.zeros_like(t_seg_r)
            else:
                # seg is unchanged — reuse channel 1 from img_orig
                t_seg_r = img_orig[1].numpy()

            img_trans = torch.tensor(
                np.stack([t_sl.astype(np.float32), t_seg_r.astype(np.float32)], axis=0),
                dtype=torch.float32,
            )
        except Exception as e:
            if fname not in self._missing_warned:
                print(f"  warn: error loading translated {fname}: {e} — using original")
                self._missing_warned.add(fname)
            img_trans = img_orig

        return img_orig, img_trans, heatmaps, coords


# ── model helpers ─────────────────────────────────────────────────────────────

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


def param_groups_discriminative(model):
    enc, dec, head = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(n.startswith(x) for x in ENC_PREFIXES):
            enc.append(p)
        elif n.startswith("final") or n.startswith("aux_head"):
            head.append(p)
        else:
            dec.append(p)
    return [
        {"params": enc,  "lr": LR_ENC_P3},
        {"params": dec,  "lr": LR_DEC_P3},
        {"params": head, "lr": LR_HEAD_P3},
    ]


def cosine_sigma(ep, total, s_start, s_end):
    if total <= 1:
        return s_end
    t = ep / (total - 1)
    return s_start + 0.5 * (1 - math.cos(math.pi * t)) * (s_end - s_start)


def enforce_superior_ordering_batch(coords):
    out = coords.clone()
    swap = out[:, 1] > out[:, 3]
    if swap.any():
        tmp = out[swap].clone()
        out[swap, 0] = tmp[:, 2]; out[swap, 1] = tmp[:, 3]
        out[swap, 2] = tmp[:, 0]; out[swap, 3] = tmp[:, 1]
    return out


# ── validation (ACDC only, no consistency) ────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = total_mre = total_mre1 = total_mre2 = 0.0
    total_sdr = {2: 0.0, 5: 0.0, 10: 0.0}
    sample_mres = []
    vis_imgs, vis_preds, vis_gts = [], [], []

    for images, heatmaps, gt_coords in loader:
        images    = images.to(device)
        heatmaps  = heatmaps.to(device)
        gt_coords = gt_coords.to(device)

        logits = model(images)
        loss, _ = criterion(logits, heatmaps)
        val_loss += loss.item()

        pred_hm     = torch.sigmoid(logits)
        pred_coords = gaussian_subpixel_argmax(pred_hm, window=7)
        pred_coords = enforce_superior_ordering_batch(pred_coords)

        total_mre += compute_mre(pred_coords, gt_coords).item()
        m1, m2     = compute_mre_per_landmark(pred_coords, gt_coords)
        total_mre1 += m1; total_mre2 += m2
        s = compute_sdr_multi(pred_coords, gt_coords, (2.0, 5.0, 10.0))
        for t in total_sdr:
            total_sdr[t] += s[t]
        sample_mres.extend(compute_per_sample_mre(pred_coords, gt_coords).cpu().tolist())

        if len(vis_imgs) < N_VIS:
            for j in range(min(N_VIS - len(vis_imgs), images.size(0))):
                vis_imgs.append(images[j, 0].cpu().numpy())
                vis_preds.append(pred_coords[j].cpu().numpy())
                vis_gts.append(gt_coords[j].cpu().numpy())

    n = len(loader)
    return {
        "val_loss": val_loss / n,
        "mre":  total_mre / n,
        "mre1": total_mre1 / n,
        "mre2": total_mre2 / n,
        "sdr":  {t: total_sdr[t] / n for t in total_sdr},
        "pct":  compute_mre_percentiles(sample_mres),
        "vis":  (vis_imgs, vis_preds, vis_gts),
    }


# ── train one epoch (paired loader) ──────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    lambda_c, scaler=None):
    """
    loader yields (img_orig, img_trans, heatmaps, coords).
    Heatmap loss is computed on img_orig (so base performance stays grounded).
    Consistency loss = MSE(sigmoid(model(img_orig)), sigmoid(model(img_trans))).
    """
    model.train()
    total_loss = total_hm = total_cons = 0.0
    n_batches = 0

    for img_orig, img_trans, heatmaps, _ in loader:
        img_orig  = img_orig.to(device,  non_blocking=True)
        img_trans = img_trans.to(device, non_blocking=True)
        heatmaps  = heatmaps.to(device,  non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits_orig = model(img_orig)
                if lambda_c > 0:
                    logits_trans = model(img_trans)
            # loss in fp32 to avoid overflow in Wing loss
            logits_orig = logits_orig.float()
            hm_loss, _ = criterion(logits_orig, heatmaps)

            if lambda_c > 0:
                logits_trans = logits_trans.float()
                cons_loss = F.mse_loss(
                    torch.sigmoid(logits_orig),
                    torch.sigmoid(logits_trans),
                )
                loss = hm_loss + lambda_c * cons_loss
            else:
                cons_loss = torch.tensor(0.0)
                loss = hm_loss

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                continue   # skip was triggered
        else:
            logits_orig = model(img_orig)
            hm_loss, _ = criterion(logits_orig, heatmaps)

            if lambda_c > 0:
                logits_trans = model(img_trans)
                cons_loss = F.mse_loss(
                    torch.sigmoid(logits_orig),
                    torch.sigmoid(logits_trans),
                )
                loss = hm_loss + lambda_c * cons_loss
            else:
                cons_loss = torch.tensor(0.0)
                loss = hm_loss

            if not torch.isfinite(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()
        total_hm   += hm_loss.item()
        total_cons += cons_loss.item()
        n_batches  += 1

    if n_batches == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (total_loss / n_batches,
            total_hm   / n_batches,
            total_cons / n_batches,
            n_batches)


# ── phase runner ──────────────────────────────────────────────────────────────

def run_phase(name, model, train_loader, val_loader, criterion, optimizer,
              scheduler, paired_ds, sigma_start, sigma_end, n_epochs,
              lambda_c, device, run_dir, history, best_p90_ref,
              scaler=None, early_stop=None, epoch_offset=0):

    best_p90 = best_p90_ref[0]
    no_improve = 0

    for ep_in in range(n_epochs):
        sigma = cosine_sigma(ep_in, n_epochs, sigma_start, sigma_end)
        paired_ds.set_sigma(sigma)

        train_loss, hm_loss, cons_loss, n_steps = train_one_epoch(
            model, train_loader, optimizer, criterion, device, lambda_c, scaler
        )
        if n_steps > 0 and scheduler is not None:
            scheduler.step()

        m = validate(model, val_loader, criterion, device)
        global_ep = epoch_offset + ep_in + 1
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "phase": name, "epoch": global_ep,
            "train_loss": train_loss, "hm_loss": hm_loss,
            "cons_loss": cons_loss, "val_loss": m["val_loss"],
            "mre": m["mre"], "sdr": m["sdr"][5], "sigma": sigma,
        })

        print(
            f"[{name}] Ep {global_ep:3d}  σ={sigma:.2f}  lr={lr_now:.2e}  "
            f"Train={train_loss:.4f} (hm={hm_loss:.4f} cons={cons_loss:.4f})  "
            f"Val={m['val_loss']:.4f}  "
            f"MRE={m['mre']:.2f}px (LM1={m['mre1']:.2f} LM2={m['mre2']:.2f})  "
            f"SDR@5={m['sdr'][5]:.3f}  "
            f"P90={m['pct'][90]:.2f}px"
        )

        save_epoch_grid(*m["vis"], epoch=global_ep,
                        save_dir=os.path.join(run_dir, "grids"), n_samples=N_VIS)

        if m["pct"][90] < best_p90:
            best_p90 = m["pct"][90]
            best_p90_ref[0] = best_p90
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            print(f"  -> best P90={best_p90:.2f}px  saved best_model.pth")
        else:
            no_improve += 1
            if early_stop is not None and no_improve >= early_stop:
                print(f"  early-stop ({early_stop} epochs no improvement)")
                return epoch_offset + ep_in + 1

        torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))

    return epoch_offset + n_epochs


# ── module-level paired dataset (must be top-level for multiprocessing pickle) ─

class SubsetPaired(PairedTranslatedDataset):
    """
    PairedTranslatedDataset over a patient-level subset of indices.

    Must be defined at module level (not inside main()) so Python's
    multiprocessing spawn can pickle it for DataLoader workers.
    """

    def __init__(self, acdc_ds, indices, trans_img_dir, trans_mask_dir):
        self.ds              = acdc_ds
        self._indices        = indices
        self.trans_img_dir   = trans_img_dir
        self.trans_mask_dir  = trans_mask_dir
        self._missing_warned = set()

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        import nibabel as nib
        import cv2 as _cv2

        real_idx = self._indices[idx]
        img_orig, heatmaps, coords = self.ds[real_idx]

        fname, slice_idx, _, H_orig, W_orig = self.ds.samples[real_idx]
        trans_img_path  = os.path.join(self.trans_img_dir,  fname)
        trans_mask_path = os.path.join(self.trans_mask_dir, fname)

        if not os.path.exists(trans_img_path):
            if fname not in self._missing_warned:
                print(f"  warn: translated file missing, using original: {fname}")
                self._missing_warned.add(fname)
            return img_orig, img_orig, heatmaps, coords

        try:
            t_img = nib.load(trans_img_path).get_fdata().astype(np.float32)
            t_sl  = np.take(t_img, slice_idx, axis=self.ds.slice_axis)
            t_sl  = _cv2.resize(t_sl, (256, 256), interpolation=_cv2.INTER_LINEAR)
            mu, std = t_sl.mean(), t_sl.std() + 1e-8
            t_sl  = (t_sl - mu) / std

            if os.path.exists(trans_mask_path):
                t_seg    = np.round(nib.load(trans_mask_path).get_fdata().astype(np.float32))
                t_seg_sl = np.take(t_seg, slice_idx, axis=self.ds.slice_axis)
                t_seg_r  = _cv2.resize(t_seg_sl, (256, 256),
                                       interpolation=_cv2.INTER_NEAREST)
                if not np.any(np.round(t_seg_r) == 1):
                    t_seg_r = np.zeros_like(t_seg_r)
                else:
                    smax = t_seg_r.max()
                    t_seg_r = (t_seg_r / smax).astype(np.float32) if smax > 0 \
                              else np.zeros_like(t_seg_r)
            else:
                t_seg_r = img_orig[1].numpy()

            img_trans = torch.tensor(
                np.stack([t_sl.astype(np.float32), t_seg_r.astype(np.float32)], axis=0),
                dtype=torch.float32,
            )
        except Exception as e:
            if fname not in self._missing_warned:
                print(f"  warn: error loading translated {fname}: {e}")
                self._missing_warned.add(fname)
            img_trans = img_orig

        return img_orig, img_trans, heatmaps, coords


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("checkpoints", f"finetune_translated_2ch_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "grids"), exist_ok=True)
    print(f"Run dir : {run_dir}")
    print(f"Device  : {device}   AMP: {use_amp}")

    # ── check translated data exists
    if not os.path.isdir(TRANS_IMG_DIR):
        raise FileNotFoundError(
            f"Translated image dir not found: {TRANS_IMG_DIR}\n"
            f"Run generate_translated_dataset.py first."
        )
    n_trans = len([f for f in os.listdir(TRANS_IMG_DIR) if f.endswith(".nii.gz")])
    n_orig  = len([f for f in os.listdir(ACDC_IMG_DIR)  if f.endswith(".nii.gz")])
    print(f"ACDC volumes    : {n_orig}")
    print(f"Translated vols : {n_trans}")
    if n_trans < n_orig * 0.5:
        print(f"  warning: only {n_trans}/{n_orig} translated files found — "
              f"missing files will fall back to originals")

    # ── build ACDC original dataset (used for both paired train and val)
    probe_ds = ACDCLandmarkDataset(
        ACDC_IMG_DIR, ACDC_MASK_DIR, ACDC_POINT_DIR,
        in_channels=IN_CHANNELS, augment=False, sigma=SIGMA_P1,
    )
    n_total = len(probe_ds)
    del probe_ds

    # patient-level 80/20 split: use same seed as original training
    rng = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(n_total, generator=rng).tolist()
    n_val     = int(n_total * VAL_SPLIT)
    train_idx = idx[n_val:]
    val_idx   = idx[:n_val]
    print(f"ACDC slices — total={n_total}  train={len(train_idx)}  val={len(val_idx)}")

    # ── paired train dataset
    acdc_train_ds = ACDCLandmarkDataset(
        ACDC_IMG_DIR, ACDC_MASK_DIR, ACDC_POINT_DIR,
        in_channels=IN_CHANNELS, augment=True, sigma=SIGMA_P1,
    )
    # subset to training indices only
    acdc_train_sub = Subset(acdc_train_ds, train_idx)

    paired_train_ds = SubsetPaired(
        acdc_train_ds, train_idx, TRANS_IMG_DIR, TRANS_MASK_DIR
    )
    print(f"Paired train slices: {len(paired_train_ds)}")

    # ── val dataset — original ACDC only, plain loader (no pairing needed)
    acdc_val_ds = ACDCLandmarkDataset(
        ACDC_IMG_DIR, ACDC_MASK_DIR, ACDC_POINT_DIR,
        in_channels=IN_CHANNELS, augment=False, sigma=SIGMA_P1,
    )
    val_subset = Subset(acdc_val_ds, val_idx)

    train_loader = DataLoader(
        paired_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── model
    model = UNetResNet34(
        in_channels=IN_CHANNELS, num_classes=2,
        dropout=0.0, pretrained=False, cardiac_pretrained=False,
    ).to(device)
    state = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {BASE_CHECKPOINT}  missing={len(missing)} unexpected={len(unexpected)}")

    criterion = HeatmapLoss(
        coord_weight=20.0,
        sep_weight=0.5,
        sep_min_dist=0.08,
        wing_w=0.008,
        wing_eps=0.002,
        lm_weights=[2.0, 1.0],
        hard_k=BATCH_SIZE - 2,
    ).to(device)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    history     = []
    best_p90_ref = [float("inf")]

    # baseline val before any training
    print("\n[baseline] ACDC checkpoint on ACDC val split")
    m0 = validate(model, val_loader, criterion, device)
    print(f"  MRE={m0['mre']:.2f}px  SDR@5={m0['sdr'][5]:.3f}  P90={m0['pct'][90]:.2f}px")
    history.append({"phase": "baseline", "epoch": 0,
                    "train_loss": 0.0, "hm_loss": 0.0, "cons_loss": 0.0,
                    "val_loss": m0["val_loss"], "mre": m0["mre"],
                    "sdr": m0["sdr"][5], "sigma": SIGMA_P1})

    epoch_offset = 0

    # ── PHASE 1 — head/decoder warmup, encoder frozen
    if args.epochs_p1 > 0:
        print(f"\n{'='*60}\n  PHASE 1 — head/decoder warmup "
              f"({args.epochs_p1} ep, encoder frozen, λ_c=0)\n{'='*60}")
        set_encoder_grad(model, False)
        op = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_P1, weight_decay=WEIGHT_DECAY,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            op, T_max=args.epochs_p1, eta_min=1e-5)
        epoch_offset = run_phase(
            "P1", model, train_loader, val_loader, criterion, op, sch,
            paired_train_ds, SIGMA_P1, SIGMA_P1, args.epochs_p1,
            LAMBDA_C_P1, device, run_dir, history, best_p90_ref,
            scaler=scaler, epoch_offset=epoch_offset,
        )

    # ── PHASE 2 — BN adaptation, encoder convs frozen
    if args.epochs_p2 > 0:
        print(f"\n{'='*60}\n  PHASE 2 — BN adaptation "
              f"({args.epochs_p2} ep, λ_c={LAMBDA_C_P2})\n{'='*60}")
        freeze_all(model)
        unfreeze_batchnorm(model)
        for n, p in model.named_parameters():
            if not any(n.startswith(x) for x in ENC_PREFIXES):
                p.requires_grad = True
        op = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_P2, weight_decay=WEIGHT_DECAY,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            op, T_max=args.epochs_p2, eta_min=5e-6)
        epoch_offset = run_phase(
            "P2", model, train_loader, val_loader, criterion, op, sch,
            paired_train_ds, SIGMA_P2_START, SIGMA_P2_END, args.epochs_p2,
            LAMBDA_C_P2, device, run_dir, history, best_p90_ref,
            scaler=scaler, epoch_offset=epoch_offset,
        )

    # ── PHASE 3 — full unfreeze, discriminative LRs
    if args.epochs_p3 > 0:
        print(f"\n{'='*60}\n  PHASE 3 — full fine-tune "
              f"({args.epochs_p3} ep, λ_c={LAMBDA_C_P3})\n{'='*60}")
        for p in model.parameters():
            p.requires_grad = True
        op = torch.optim.AdamW(
            param_groups_discriminative(model),
            weight_decay=WEIGHT_DECAY,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            op, T_max=args.epochs_p3, eta_min=1e-7)
        epoch_offset = run_phase(
            "P3", model, train_loader, val_loader, criterion, op, sch,
            paired_train_ds, SIGMA_P3_START, SIGMA_P3_END, args.epochs_p3,
            LAMBDA_C_P3, device, run_dir, history, best_p90_ref,
            scaler=scaler, early_stop=EARLY_STOP_P3, epoch_offset=epoch_offset,
        )

    # ── save config + history + curve
    config = {
        "base_checkpoint": BASE_CHECKPOINT,
        "in_channels": IN_CHANNELS,
        "epochs_p1": args.epochs_p1,
        "epochs_p2": args.epochs_p2,
        "epochs_p3": args.epochs_p3,
        "lambda_c_p1": LAMBDA_C_P1,
        "lambda_c_p2": LAMBDA_C_P2,
        "lambda_c_p3": LAMBDA_C_P3,
        "lr_p1": LR_P1, "lr_p2": LR_P2,
        "lr_enc_p3": LR_ENC_P3, "lr_dec_p3": LR_DEC_P3, "lr_head_p3": LR_HEAD_P3,
        "sigma_p1": SIGMA_P1,
        "sigma_p2": [SIGMA_P2_START, SIGMA_P2_END],
        "sigma_p3": [SIGMA_P3_START, SIGMA_P3_END],
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "best_p90": best_p90_ref[0],
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # save_training_curve expects history entries with train_loss/val_loss/mre/sdr/sigma
    save_training_curve(history, run_dir)

    best_ckpt = os.path.join(run_dir, "best_model.pth")
    print(f"\n{'='*44}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best P90 MRE  : {best_p90_ref[0]:.2f}px")
    print(f"  Total epochs  : {epoch_offset}")
    print(f"  Checkpoint    : {best_ckpt}")
    print(f"{'='*44}")
    print(f"\nNext — fine-tune on real rv_landmark data:")
    print(f"  python finetune_rv.py \\")
    print(f"    --base-checkpoint {best_ckpt} \\")
    print(f"    --in-channels 2 \\")
    print(f"    --epochs-p1 4 --epochs-p2 6 --epochs-p3 15")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-p1", type=int, default=P1_EPOCHS)
    parser.add_argument("--epochs-p2", type=int, default=P2_EPOCHS)
    parser.add_argument("--epochs-p3", type=int, default=P3_EPOCHS)
    parser.add_argument("--no-amp",    action="store_true",
                        help="Disable mixed precision (use if AMP causes NaN)")
    args = parser.parse_args()
    main(args)
