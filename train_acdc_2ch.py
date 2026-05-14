"""
Train UNetResNet34 on ACDC — 2-channel (MRI + segmentation mask).
3-phase curriculum: warmup → curriculum (sigma decay + mixup) → precision.

Identical to train_acdc_1ch.py in every way except IN_CHANNELS=2 and RUN_TAG.
Purpose: direct 1-ch vs 2-ch comparison.

Checkpoints saved to: acdc-checkpoints/acdc_2ch_TIMESTAMP/
"""

# ── only these two lines differ between train_acdc_1ch.py and train_acdc_2ch.py ──
IN_CHANNELS = 2
RUN_TAG     = "acdc_2ch"
# ─────────────────────────────────────────────────────────────────────────────────

import os, json, math, random, time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from dataset.acdc_landmark_dataset import ACDCLandmarkDataset
from models.unet_resnet34 import UNetResNet34
from utils.loss import HeatmapLoss
from utils.postprocess import gaussian_subpixel_argmax
from utils.metrics import (
    compute_mre, compute_mre_per_landmark,
    compute_sdr_multi, compute_per_sample_mre, compute_mre_percentiles,
)
from utils.visualize import save_epoch_grid, save_training_curve

# ─────────────────────────────── config ───────────────────────────────────────
IMAGE_DIR = "data/acdc/images"
MASK_DIR  = "data/acdc/masks"
RVIP_DIR  = "data/acdc/points"    # landmark .nii.gz label masks (1=LM1, 2=LM2)

TRAIN_IDS = [f"patient{i:03d}" for i in range(1,  81)]
VAL_IDS   = [f"patient{i:03d}" for i in range(81, 91)]
TEST_IDS  = [f"patient{i:03d}" for i in range(91, 101)]

BATCH_SIZE   = 8
NUM_WORKERS  = 2
SEED         = 42
GRAD_CLIP    = 2.0
WEIGHT_DECAY = 1e-4
N_VIS        = 8

P1_EPOCHS    = 6
P2_EPOCHS    = 50
P3_EPOCHS    = 50

SIGMA_P1     = 12.0
SIGMA_P2_END = 2.0
SIGMA_P3_END = 1.0

LOSS_KWARGS = dict(
    coord_weight = 20.0,
    sep_weight   = 0.5,
    sep_min_dist = 0.08,
    wing_w       = 0.008,
    wing_eps     = 0.002,
    lm_weights   = [2.0, 1.0],
    hard_k       = 6,
)
# ──────────────────────────────────────────────────────────────────────────────


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────── encoder freeze / unfreeze ────────────────────────────
_ENC  = {"enc0", "enc1", "enc2", "enc3", "enc4"}
_HEAD = {"final", "aux_head1", "aux_head2"}


def freeze_encoder(model):
    for name, p in model.named_parameters():
        if name.split(".")[0] in _ENC:
            p.requires_grad_(False)


def unfreeze_encoder_bn(model):
    for name, mod in model.named_modules():
        if name.split(".")[0] in _ENC and isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p in mod.parameters():
                p.requires_grad_(True)


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad_(True)


def p3_param_groups(model):
    enc, dec, head = [], [], []
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        if top in _ENC:
            enc.append(p)
        elif top in _HEAD or "final" in name:
            head.append(p)
        else:
            dec.append(p)
    return [
        {"params": enc,  "lr": 5e-6},
        {"params": dec,  "lr": 5e-5},
        {"params": head, "lr": 1e-4},
    ]


# ──────────────────────────── schedules ───────────────────────────────────────

def cosine_anneal(start, end, step, total):
    frac = step / max(total - 1, 1)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * frac))


# ──────────────────────────────── TTA ─────────────────────────────────────────

@torch.no_grad()
def tta_predict(model, images, device):
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

    avg_hm = torch.stack(hms).mean(0)
    coords = gaussian_subpixel_argmax(avg_hm)
    return coords, avg_hm


def enforce_superior(coords: torch.Tensor) -> torch.Tensor:
    c    = coords.clone()
    swap = c[:, 1] > c[:, 3]
    c[swap, 0], c[swap, 2] = coords[swap, 2].clone(), coords[swap, 0].clone()
    c[swap, 1], c[swap, 3] = coords[swap, 3].clone(), coords[swap, 1].clone()
    return c


# ─────────────────────────── mixup ────────────────────────────────────────────

def mixup(images, heatmaps, coords, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    return (
        lam * images   + (1 - lam) * images[idx],
        lam * heatmaps + (1 - lam) * heatmaps[idx],
        lam * coords   + (1 - lam) * coords[idx],
    )


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


# ─────────────────────── training epoch ───────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device, use_amp,
                mixup_prob=0.0):
    model.train()
    total_loss = 0.0
    for imgs, hms, coords in loader:
        imgs, hms, coords = imgs.to(device), hms.to(device), coords.to(device)
        if mixup_prob > 0 and random.random() < mixup_prob:
            imgs, hms, coords = mixup(imgs, hms, coords)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            out = model(imgs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            loss, _ = criterion(out, hms)

            for attr in ("_aux_feat1", "_aux_feat2"):
                feat = getattr(model, attr, None)
                if feat is not None:
                    head_name = "aux_head1" if "1" in attr else "aux_head2"
                    head = getattr(model, head_name, None)
                    if head is not None:
                        a_out = head(feat)
                        tgt   = F.interpolate(hms, size=a_out.shape[2:],
                                              mode="bilinear", align_corners=False)
                        a_loss, _ = criterion(a_out, tgt)
                        loss = loss + 0.3 * a_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


# ─────────────────────────── validation ───────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, criterion, epoch, grid_dir):
    model.eval()
    total_loss  = 0.0
    all_preds, all_gts, all_imgs = [], [], []
    sample_mres = []

    for imgs, hms, coords in loader:
        imgs  = imgs.to(device)
        hms_d = hms.to(device)

        pred_coords, avg_hm = tta_predict(model, imgs, device)
        pred_coords = enforce_superior(pred_coords.cpu())

        log_hm = torch.log(avg_hm.clamp(1e-6, 1 - 1e-6))
        loss, _ = criterion(log_hm, hms_d)
        total_loss += loss.item() * imgs.size(0)

        per_s = compute_per_sample_mre(pred_coords, coords)
        sample_mres.extend(per_s.tolist())

        all_preds.extend(pred_coords.numpy())
        all_gts.extend(coords.numpy())
        all_imgs.extend(imgs[:, 0].cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    pred_t   = torch.tensor(np.array(all_preds))
    gt_t     = torch.tensor(np.array(all_gts))

    mre           = compute_mre(pred_t, gt_t).item()
    mre_lm1, mre_lm2 = compute_mre_per_landmark(pred_t, gt_t)
    sdr           = compute_sdr_multi(pred_t, gt_t, thresholds=(2, 5, 10))
    pct           = compute_mre_percentiles(sample_mres)

    save_epoch_grid(
        all_imgs[:N_VIS], all_preds[:N_VIS], all_gts[:N_VIS],
        epoch, grid_dir, n_samples=N_VIS,
    )
    return avg_loss, mre, mre_lm1, mre_lm2, sdr, pct


# ─────────────────────────── test evaluation ──────────────────────────────────

@torch.no_grad()
def evaluate_test(model, device):
    ds = ACDCLandmarkDataset(
        IMAGE_DIR, MASK_DIR, RVIP_DIR,
        patient_ids=TEST_IDS, in_channels=IN_CHANNELS,
        augment=False, sigma=SIGMA_P3_END,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    model.eval()
    all_preds, all_gts, sample_mres = [], [], []
    for imgs, _, coords in loader:
        pred_coords, _ = tta_predict(model, imgs, device)
        pred_coords = enforce_superior(pred_coords.cpu())
        sample_mres.extend(compute_per_sample_mre(pred_coords, coords).tolist())
        all_preds.extend(pred_coords.numpy())
        all_gts.extend(coords.numpy())

    pred_t = torch.tensor(np.array(all_preds))
    gt_t   = torch.tensor(np.array(all_gts))
    mre              = compute_mre(pred_t, gt_t).item()
    mre_lm1, mre_lm2 = compute_mre_per_landmark(pred_t, gt_t)
    sdr              = compute_sdr_multi(pred_t, gt_t, thresholds=(2, 5, 10))
    pct              = compute_mre_percentiles(sample_mres)
    return mre, mre_lm1, mre_lm2, sdr, pct


# ─────────────────────────── JSON logging ─────────────────────────────────────

def save_epoch_log(run_dir, log_entry):
    path = os.path.join(run_dir, "training_log.json")
    entries = []
    if os.path.exists(path):
        with open(path) as f:
            entries = json.load(f)
    entries.append(log_entry)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


# ──────────────────────────────── main ────────────────────────────────────────

def main():
    seed_everything(SEED)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join("acdc-checkpoints", f"{RUN_TAG}_{timestamp}")
    grid_dir  = os.path.join(run_dir, "grids")
    os.makedirs(grid_dir, exist_ok=True)

    print(f"\nRun    : {run_dir}")
    print(f"Device : {device}  |  AMP : {use_amp}")
    print(f"Channels: {IN_CHANNELS}")

    # ── config.json ───────────────────────────────────────────────────────────
    config = {
        "in_channels": IN_CHANNELS,
        "batch_size":  BATCH_SIZE,
        "seed":        SEED,
        "image_dir":   IMAGE_DIR,
        "mask_dir":    MASK_DIR,
        "rvip_dir":    RVIP_DIR,
        "train_patients": TRAIN_IDS,
        "val_patients":   VAL_IDS,
        "test_patients":  TEST_IDS,
        "phases": {
            "p1": {"epochs": P1_EPOCHS,  "lr": 5e-4,  "sigma": SIGMA_P1},
            "p2": {"epochs": P2_EPOCHS,  "lr_head": 2e-4, "lr_bn": 1e-5,
                   "sigma_start": SIGMA_P1, "sigma_end": SIGMA_P2_END, "early_stop": 15},
            "p3": {"epochs": P3_EPOCHS,  "lr_enc": 5e-6,  "lr_dec": 5e-5, "lr_head": 1e-4,
                   "sigma_start": SIGMA_P2_END, "sigma_end": SIGMA_P3_END, "early_stop": 20},
        },
        "loss": {
            "coord_weight": LOSS_KWARGS["coord_weight"],
            "sep_weight":   LOSS_KWARGS["sep_weight"],
            "lm_weights":   LOSS_KWARGS["lm_weights"],
            "hard_k":       LOSS_KWARGS["hard_k"],
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = ACDCLandmarkDataset(
        IMAGE_DIR, MASK_DIR, RVIP_DIR,
        patient_ids=TRAIN_IDS, in_channels=IN_CHANNELS,
        augment=True, sigma=SIGMA_P1,
    )
    val_ds = ACDCLandmarkDataset(
        IMAGE_DIR, MASK_DIR, RVIP_DIR,
        patient_ids=VAL_IDS, in_channels=IN_CHANNELS,
        augment=False, sigma=SIGMA_P1,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────────
    model     = UNetResNet34(in_channels=IN_CHANNELS, num_classes=2, pretrained=True).to(device)
    criterion = HeatmapLoss(**LOSS_KWARGS)

    curve_history  = []
    best_p90_p2    = float("inf")
    best_p90_p3    = float("inf")
    no_improve_p2  = 0
    no_improve_p3  = 0
    global_epoch   = 0
    best_epoch     = 0
    best_val_mre   = float("inf")
    best_val_sdr5  = 0.0

    best_p2_path    = os.path.join(run_dir, "best_p2.pth")
    best_model_path = os.path.join(run_dir, "best_model.pth")
    last_model_path = os.path.join(run_dir, "last_model.pth")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Warmup
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("PHASE 1 — Warmup  (decoder + head only, encoder frozen)")
    print("=" * 62)

    freeze_encoder(model)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4, weight_decay=WEIGHT_DECAY,
    )

    for ep in range(P1_EPOCHS):
        global_epoch += 1
        sigma = SIGMA_P1
        train_ds.set_sigma(sigma)
        val_ds.set_sigma(sigma)

        t0         = time.time()
        train_loss = train_epoch(model, train_loader, opt, criterion,
                                 scaler, device, use_amp, mixup_prob=0.0)
        val_loss, mre, lm1_e, lm2_e, sdr, pct = validate(
            model, val_loader, device, criterion, global_epoch, grid_dir)

        print(f"  P1 {ep+1:02d}/{P1_EPOCHS} | σ={sigma:.1f} | "
              f"loss {train_loss:.4f}/{val_loss:.4f} | "
              f"MRE={mre:.2f}px | SDR@5={sdr[5]*100:.1f}% | "
              f"P90={pct[90]:.2f}px | {time.time()-t0:.0f}s")

        curve_history.append({"epoch": global_epoch, "train_loss": train_loss,
                               "val_loss": val_loss, "mre": mre, "sdr": sdr[5],
                               "sigma": sigma})
        save_epoch_log(run_dir, {
            "run_dir": run_dir, "in_channels": IN_CHANNELS, "phase": "P1",
            "epoch": global_epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_mre": mre, "val_mre_lm1": lm1_e, "val_mre_lm2": lm2_e,
            "val_sdr2": sdr[2], "val_sdr5": sdr[5], "val_sdr10": sdr[10],
            "val_p50": pct[50], "val_p90": pct[90],
            "sigma": sigma, "lr": current_lr(opt), "best_p90_so_far": best_p90_p2,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Curriculum
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("PHASE 2 — Curriculum  (BN unfrozen, sigma decay, mixup)")
    print("=" * 62)

    unfreeze_encoder_bn(model)
    enc_params  = [p for n, p in model.named_parameters()
                   if n.split(".")[0] in _ENC and p.requires_grad]
    rest_params = [p for n, p in model.named_parameters()
                   if n.split(".")[0] not in _ENC and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": 1e-5},
         {"params": rest_params, "lr": 2e-4}],
        weight_decay=WEIGHT_DECAY,
    )

    for ep in range(P2_EPOCHS):
        global_epoch += 1
        sigma = cosine_anneal(SIGMA_P1, SIGMA_P2_END, ep, P2_EPOCHS)
        train_ds.set_sigma(sigma)
        val_ds.set_sigma(sigma)

        t0         = time.time()
        train_loss = train_epoch(model, train_loader, opt, criterion,
                                 scaler, device, use_amp, mixup_prob=0.3)
        val_loss, mre, lm1_e, lm2_e, sdr, pct = validate(
            model, val_loader, device, criterion, global_epoch, grid_dir)

        print(f"  P2 {ep+1:02d}/{P2_EPOCHS} | σ={sigma:.2f} | "
              f"loss {train_loss:.4f}/{val_loss:.4f} | "
              f"MRE={mre:.2f}px | SDR@5={sdr[5]*100:.1f}% | "
              f"P90={pct[90]:.2f}px | {time.time()-t0:.0f}s")

        curve_history.append({"epoch": global_epoch, "train_loss": train_loss,
                               "val_loss": val_loss, "mre": mre, "sdr": sdr[5],
                               "sigma": sigma})
        save_epoch_log(run_dir, {
            "run_dir": run_dir, "in_channels": IN_CHANNELS, "phase": "P2",
            "epoch": global_epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_mre": mre, "val_mre_lm1": lm1_e, "val_mre_lm2": lm2_e,
            "val_sdr2": sdr[2], "val_sdr5": sdr[5], "val_sdr10": sdr[10],
            "val_p50": pct[50], "val_p90": pct[90],
            "sigma": sigma, "lr": current_lr(opt), "best_p90_so_far": best_p90_p2,
        })

        if pct[90] < best_p90_p2:
            best_p90_p2   = pct[90]
            no_improve_p2 = 0
            best_epoch    = global_epoch
            torch.save(model.state_dict(), best_p2_path)
            print(f"    ↳ saved best_p2.pth  (P90={best_p90_p2:.2f}px)")
        else:
            no_improve_p2 += 1
            if no_improve_p2 >= 15:
                print(f"    Early stop P2 at local epoch {ep+1} (patience=15)")
                break

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Precision
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("PHASE 3 — Precision  (full unfreeze, discriminative LR)")
    print("=" * 62)

    model.load_state_dict(torch.load(best_p2_path, map_location=device))
    unfreeze_all(model)
    opt = torch.optim.AdamW(p3_param_groups(model), weight_decay=WEIGHT_DECAY)

    for ep in range(P3_EPOCHS):
        global_epoch += 1
        sigma = cosine_anneal(SIGMA_P2_END, SIGMA_P3_END, ep, P3_EPOCHS)
        train_ds.set_sigma(sigma)
        val_ds.set_sigma(sigma)

        t0         = time.time()
        train_loss = train_epoch(model, train_loader, opt, criterion,
                                 scaler, device, use_amp, mixup_prob=0.0)
        val_loss, mre, lm1_e, lm2_e, sdr, pct = validate(
            model, val_loader, device, criterion, global_epoch, grid_dir)

        print(f"  P3 {ep+1:02d}/{P3_EPOCHS} | σ={sigma:.3f} | "
              f"loss {train_loss:.4f}/{val_loss:.4f} | "
              f"MRE={mre:.2f}px | SDR@5={sdr[5]*100:.1f}% | "
              f"P90={pct[90]:.2f}px | {time.time()-t0:.0f}s")

        curve_history.append({"epoch": global_epoch, "train_loss": train_loss,
                               "val_loss": val_loss, "mre": mre, "sdr": sdr[5],
                               "sigma": sigma})
        save_epoch_log(run_dir, {
            "run_dir": run_dir, "in_channels": IN_CHANNELS, "phase": "P3",
            "epoch": global_epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_mre": mre, "val_mre_lm1": lm1_e, "val_mre_lm2": lm2_e,
            "val_sdr2": sdr[2], "val_sdr5": sdr[5], "val_sdr10": sdr[10],
            "val_p50": pct[50], "val_p90": pct[90],
            "sigma": sigma, "lr": current_lr(opt), "best_p90_so_far": best_p90_p3,
        })

        if pct[90] < best_p90_p3:
            best_p90_p3   = pct[90]
            no_improve_p3 = 0
            best_epoch    = global_epoch
            best_val_mre  = mre
            best_val_sdr5 = sdr[5]
            torch.save(model.state_dict(), best_model_path)
            print(f"    ↳ saved best_model.pth  (P90={best_p90_p3:.2f}px)")
        else:
            no_improve_p3 += 1
            if no_improve_p3 >= 20:
                print(f"    Early stop P3 at local epoch {ep+1} (patience=20)")
                break

    torch.save(model.state_dict(), last_model_path)

    # ─────────────────────────── test ─────────────────────────────────────────
    print("\nEvaluating on test set (patients 091-100) …")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    t_mre, t_lm1, t_lm2, t_sdr, t_pct = evaluate_test(model, device)

    save_training_curve(curve_history, run_dir)

    results = {
        "best_val_p90":  best_p90_p3,
        "best_val_mre":  best_val_mre,
        "best_val_sdr5": best_val_sdr5,
        "best_epoch":    best_epoch,
        "test_mre":      t_mre,
        "test_mre_lm1":  t_lm1,
        "test_mre_lm2":  t_lm2,
        "test_sdr2":     t_sdr[2],
        "test_sdr5":     t_sdr[5],
        "test_sdr10":    t_sdr[10],
        "test_p50":      t_pct[50],
        "test_p90":      t_pct[90],
        "checkpoint":    best_model_path,
        "total_epochs":  global_epoch,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 44)
    print("TRAINING COMPLETE")
    print(f"Val  MRE  : {best_val_mre:.2f}px")
    print(f"Val  SDR@5: {best_val_sdr5*100:.1f}%")
    print(f"Test MRE  : {t_mre:.2f}px")
    print(f"Test SDR@5: {t_sdr[5]*100:.1f}%")
    print(f"Checkpoint: {best_model_path}")
    print("=" * 44)


if __name__ == "__main__":
    main()
