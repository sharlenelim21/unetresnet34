# ── the only two lines that differ between train_acdc_1ch.py and train_acdc_2ch.py ──
IN_CHANNELS = 1
RUN_TAG     = "acdc_1ch"
# ────────────────────────────────────────────────────────────────────────────────

import os, json, time
import torch
import torch.nn.functional as F_nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from datetime import datetime

from dataset.acdc_landmark_dataset import ACDCLandmarkDataset
from models.unet_resnet34 import UNetResNet34
from utils.loss import HeatmapLoss
from utils.postprocess import gaussian_subpixel_argmax
from utils.metrics import (compute_mre, compute_mre_per_landmark,
                           compute_sdr_multi, compute_per_sample_mre,
                           compute_mre_percentiles)
from utils.visualize import save_epoch_grid, save_training_curve

# ─────────────────────────────────────────────────────────────────────────────
IMAGE_DIR = "data/acdc/images"
MASK_DIR  = "data/acdc/masks"
RVIP_DIR  = "data/acdc/points"

TRAIN_IDS = [f"patient{i:03d}" for i in range(1,  81)]
VAL_IDS   = [f"patient{i:03d}" for i in range(81, 91)]
TEST_IDS  = [f"patient{i:03d}" for i in range(91, 101)]

BATCH_SIZE = 8
SEED       = 42
N_VIS      = 8

P1_EPOCHS = 6
P2_EPOCHS = 49
P3_EPOCHS = 35

LR_HEAD_P1     = 1e-3
LR_HEAD_P2     = 1e-3
LR_BACKBONE_P2 = 5e-5
LR_P3          = 3e-5

SIGMA_P1       = 12.0
SIGMA_START    = 12.0
SIGMA_END      = 2.0
SIGMA_P3_START = 2.0
SIGMA_P3_END   = 1.2

MIXUP_ALPHA  = 0.2
MIXUP_PROB   = 0.3
WEIGHT_DECAY = 1e-4

AUX_WEIGHT_MAX = 0.4
AUX_WEIGHT_MIN = 0.05

EARLY_STOP_P2 = 18
EARLY_STOP_P3 = 18

COORD_WEIGHT = 15.0
LM_WEIGHTS   = [2.5, 1.0]
HARD_K       = BATCH_SIZE - 2

CARDIAC_PRETRAINED = False
# ─────────────────────────────────────────────────────────────────────────────


def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)


def sigma_p2(ep, total):
    t = min(ep / total, 1.0)
    t_cos = (1.0 - np.cos(np.pi * t)) / 2.0
    return SIGMA_START + t_cos * (SIGMA_END - SIGMA_START)


def aux_weight(sigma):
    t = np.clip((sigma - 2.0) / (15.0 - 2.0), 0, 1)
    return AUX_WEIGHT_MIN + t * (AUX_WEIGHT_MAX - AUX_WEIGHT_MIN)


def set_encoder_grad(model, flag):
    for n, p in model.named_parameters():
        if any(n.startswith(x) for x in ["enc0", "enc1", "enc2", "enc3", "enc4"]):
            p.requires_grad = flag


def mixup(images, heatmaps, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(images.size(0), device=images.device)
    return (lam * images   + (1 - lam) * images[idx],
            lam * heatmaps + (1 - lam) * heatmaps[idx])


@torch.no_grad()
def tta_predict(model, image):
    preds = []
    preds.append(torch.sigmoid(model(image)))
    p = torch.sigmoid(model(torch.flip(image, [3])))
    preds.append(torch.flip(p, [3]))
    p = torch.sigmoid(model(torch.flip(image, [2])))
    preds.append(torch.flip(p, [2]))
    return torch.stack(preds).mean(0)


def train_epoch(model, loader, opt, criterion, device,
                sigma, do_mixup, do_aux, aw, scaler, use_amp, clip=5.0):
    model.train()
    tot      = 0.0
    n_finite = 0
    subs     = {"bce": 0.0, "dice": 0.0, "coord": 0.0, "sep": 0.0}

    has_aux = do_aux and hasattr(model, "aux_head1") and hasattr(model, "_aux_feat1")

    for imgs, hms, _ in loader:
        imgs = imgs.to(device)
        hms  = hms.to(device)

        if do_mixup and np.random.rand() < MIXUP_PROB:
            imgs, hms = mixup(imgs, hms, MIXUP_ALPHA)

        opt.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(imgs)
            # fp32 loss — avoids AMP precision issues in BCEWithLogitsLoss
            loss, parts = criterion(logits.float(), hms.float())

            if has_aux and model._aux_feat1 is not None:
                out1 = model.aux_head1(model._aux_feat1)
                out2 = model.aux_head2(model._aux_feat2)
                hm1  = F_nn.interpolate(hms, size=out1.shape[2:],
                                        mode="bilinear", align_corners=False)
                hm2  = F_nn.interpolate(hms, size=out2.shape[2:],
                                        mode="bilinear", align_corners=False)
                a1, _ = criterion(out1.float(), hm1.float())
                a2, _ = criterion(out2.float(), hm2.float())
                loss   = loss + aw * (a1 + a2)

        # Skip non-finite batches — guards against early NaN spikes
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        tot      += loss.item()
        n_finite += 1
        for k in subs:
            subs[k] += parts[k]

    n = max(n_finite, 1)
    return tot / n, {k: v / n for k, v in subs.items()}


def _enforce_and_match(pc_np, gt_np):
    """
    For a single sample (both shape [4]):
    1. Enforce LM1=superior (smaller y) on the prediction.
    2. Check normal vs swapped assignment against GT; keep the lower-error order.
    Returns corrected pred as float32 numpy [4].
    """
    p = pc_np.copy()
    # Step 1 — enforce superior ordering on prediction
    if p[1] > p[3]:
        p = np.array([p[2], p[3], p[0], p[1]], dtype=np.float32)
    # Step 2 — optimal matching: compare normal vs swapped vs GT
    g = gt_np
    e_normal  = np.linalg.norm(p[:2] - g[:2]) + np.linalg.norm(p[2:] - g[2:])
    p_swap    = np.array([p[2], p[3], p[0], p[1]], dtype=np.float32)
    e_swapped = np.linalg.norm(p_swap[:2] - g[:2]) + np.linalg.norm(p_swap[2:] - g[2:])
    return p_swap if e_swapped < e_normal else p


@torch.no_grad()
def validate(model, loader, criterion, device, sigma):
    model.eval()
    vl = mr = mr1 = mr2 = 0.0
    sdr = {2: 0.0, 5: 0.0, 10: 0.0}
    vis_i, vis_p, vis_g = [], [], []
    sample_mres = []

    for imgs, hms, gts in loader:
        imgs = imgs.to(device)
        hms  = hms.to(device)
        gts  = gts.to(device)

        loss, _ = criterion(model(imgs).float(), hms.float())
        vl += loss.item()

        ph = tta_predict(model, imgs)
        pc = gaussian_subpixel_argmax(ph, window=7)   # [B, 4] on device

        # Apply ordering + optimal matching per sample (Issue 7)
        pc_np  = pc.cpu().numpy()
        gt_np  = gts.cpu().numpy()
        pc_matched = np.stack([
            _enforce_and_match(pc_np[b], gt_np[b]) for b in range(pc_np.shape[0])
        ])
        pc_matched = torch.tensor(pc_matched, dtype=torch.float32).to(gts.device)

        mr  += compute_mre(pc_matched, gts).item()
        m1, m2 = compute_mre_per_landmark(pc_matched, gts)
        mr1 += m1;  mr2 += m2
        s = compute_sdr_multi(pc_matched, gts, (2.0, 5.0, 10.0))
        for t in sdr:
            sdr[t] += s[t]
        sample_mres.extend(compute_per_sample_mre(pc_matched, gts).cpu().tolist())

        if len(vis_i) < N_VIS:
            vis_i.append(imgs[0, 0].cpu().numpy())
            vis_p.append(pc_matched[0].cpu().numpy())
            vis_g.append(gts[0].cpu().numpy())

    n   = len(loader)
    pct = compute_mre_percentiles(sample_mres)
    return {
        "val_loss": vl / n,
        "mre":      mr / n,
        "mre1":     mr1 / n,
        "mre2":     mr2 / n,
        "sdr":      {t: sdr[t] / n for t in sdr},
        "pct":      pct,
        "vis":      (vis_i, vis_p, vis_g),
    }


@torch.no_grad()
def evaluate_test(model, device, criterion):
    test_ds = ACDCLandmarkDataset(
        IMAGE_DIR, MASK_DIR, RVIP_DIR,
        patient_ids=TEST_IDS,
        in_channels=IN_CHANNELS,
        augment=False,
        sigma=SIGMA_P3_END,
    )
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    mr = mr1 = mr2 = 0.0
    sdr = {2: 0.0, 5: 0.0, 10: 0.0}
    sample_mres = []

    for imgs, hms, gts in loader:
        imgs = imgs.to(device)
        gts  = gts.to(device)
        ph   = tta_predict(model, imgs)
        pc   = gaussian_subpixel_argmax(ph, window=7)
        mr  += compute_mre(pc, gts).item()
        m1, m2 = compute_mre_per_landmark(pc, gts)
        mr1 += m1;  mr2 += m2
        s = compute_sdr_multi(pc, gts, (2.0, 5.0, 10.0))
        for t in sdr:
            sdr[t] += s[t]
        sample_mres.extend(compute_per_sample_mre(pc, gts).cpu().tolist())

    n   = len(loader)
    pct = compute_mre_percentiles(sample_mres)
    return {
        "mre":  mr / n,
        "mre1": mr1 / n,
        "mre2": mr2 / n,
        "sdr":  {t: sdr[t] / n for t in sdr},
        "pct":  pct,
    }


def save_epoch_log(run_dir, entry):
    path = os.path.join(run_dir, "training_log.json")
    log  = json.load(open(path)) if os.path.exists(path) else []
    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def train(p2_checkpoint=None,
          lm1_coord_weight=3.0, lm1_heatmap_weight=2.0,
          sep_margin=0.15, sep_weight=5.0,
          use_group_norm=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)
    print(f"Device: {device}  AMP: {use_amp}  in_channels: {IN_CHANNELS}")

    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir   = os.path.join("acdc-checkpoints", f"{RUN_TAG}_{ts}")
    grids_dir = os.path.join(run_dir, "grids")
    os.makedirs(run_dir,   exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = ACDCLandmarkDataset(
        IMAGE_DIR, MASK_DIR, RVIP_DIR,
        patient_ids=TRAIN_IDS, in_channels=IN_CHANNELS,
        augment=True,  sigma=SIGMA_P1,
    )
    val_ds = ACDCLandmarkDataset(
        IMAGE_DIR, MASK_DIR, RVIP_DIR,
        patient_ids=VAL_IDS, in_channels=IN_CHANNELS,
        augment=False, sigma=SIGMA_P1,
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=2, pin_memory=True,
                    worker_init_fn=seed_worker, generator=g)
    vl = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=2, pin_memory=True,
                    worker_init_fn=seed_worker)

    # ── model ─────────────────────────────────────────────────────────────────
    model = UNetResNet34(in_channels=IN_CHANNELS, num_classes=2, dropout=0.2,
                         pretrained=True,
                         cardiac_pretrained=CARDIAC_PRETRAINED,
                         use_group_norm=use_group_norm).to(device)
    has_aux = hasattr(model, "aux_head1")
    print(f"Deep supervision: {'enabled' if has_aux else 'disabled'}")

    crit = HeatmapLoss(
        coord_weight=COORD_WEIGHT,
        sep_weight=sep_weight,
        sep_min_dist=sep_margin,
        wing_w=0.04,
        wing_eps=0.008,
        lm_weights=LM_WEIGHTS,
        hard_k=HARD_K,
        lm1_coord_weight=lm1_coord_weight,
        lm1_heatmap_weight=lm1_heatmap_weight,
    )
    crit_p3 = HeatmapLoss(
        coord_weight=25.0,
        sep_weight=sep_weight,
        sep_min_dist=sep_margin,
        wing_w=0.008,
        wing_eps=0.002,
        lm_weights=LM_WEIGHTS,
        hard_k=HARD_K,
        lm1_coord_weight=lm1_coord_weight,
        lm1_heatmap_weight=lm1_heatmap_weight,
    )

    # ── config.json ───────────────────────────────────────────────────────────
    config = {
        "run_tag": RUN_TAG, "in_channels": IN_CHANNELS,
        "image_dir": IMAGE_DIR, "mask_dir": MASK_DIR, "rvip_dir": RVIP_DIR,
        "train_ids": TRAIN_IDS, "val_ids": VAL_IDS, "test_ids": TEST_IDS,
        "batch_size": BATCH_SIZE, "seed": SEED,
        "p1": {"epochs": P1_EPOCHS, "lr": LR_HEAD_P1, "sigma": SIGMA_P1},
        "p2": {"epochs": P2_EPOCHS, "lr_head": LR_HEAD_P2,
               "lr_backbone": LR_BACKBONE_P2,
               "sigma_start": SIGMA_START, "sigma_end": SIGMA_END,
               "early_stop": EARLY_STOP_P2},
        "p3": {"epochs": P3_EPOCHS, "lr": LR_P3,
               "sigma_start": SIGMA_P3_START, "sigma_end": SIGMA_P3_END,
               "early_stop": EARLY_STOP_P3},
        "loss_p1p2": {"coord_weight": COORD_WEIGHT, "wing_w": 0.04,
                      "wing_eps": 0.008, "lm_weights": LM_WEIGHTS},
        "loss_p3":   {"coord_weight": 25.0, "wing_w": 0.008,
                      "wing_eps": 0.002, "lm_weights": LM_WEIGHTS},
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    history    = []
    best_score = float("inf")   # P90 MRE, used for P2 checkpointing
    best_mre   = float("inf")   # P90 MRE at end of P2, baseline for P3
    best_val_mre  = float("inf")
    best_val_sdr5 = 0.0

    if p2_checkpoint is None:
        # ═════════════════════════════════════════════════════════════════════
        # PHASE 1 — warmup, frozen encoder
        # ═════════════════════════════════════════════════════════════════════
        print(f"\n{'='*60}\n  PHASE 1 - warmup ({P1_EPOCHS} ep, frozen enc, sigma={SIGMA_P1})\n{'='*60}")
        set_encoder_grad(model, False)
        op1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=LR_HEAD_P1, weight_decay=WEIGHT_DECAY)
        sc1 = torch.optim.lr_scheduler.CosineAnnealingLR(op1, T_max=P1_EPOCHS, eta_min=1e-5)
        train_ds.set_sigma(SIGMA_P1)
        val_ds.set_sigma(SIGMA_P1)

        for ep in range(1, P1_EPOCHS + 1):
            t0 = time.time()
            aw = aux_weight(SIGMA_P1)
            tl_loss, sub = train_epoch(model, tl, op1, crit, device,
                                       SIGMA_P1, do_mixup=True, do_aux=has_aux,
                                       aw=aw, scaler=scaler, use_amp=use_amp)
            v = validate(model, vl, crit, device, SIGMA_P1)
            sc1.step()
            print(f"[P1] {ep:2d}/{P1_EPOCHS} | sigma={SIGMA_P1} | lr={op1.param_groups[-1]['lr']:.2e} | "
                  f"Train={tl_loss:.4f} | Val={v['val_loss']:.4f} | "
                  f"MRE={v['mre']:.2f}px (LM1={v['mre1']:.2f} LM2={v['mre2']:.2f}) | "
                  f"SDR@2/5/10: {v['sdr'][2]:.3f}/{v['sdr'][5]:.3f}/{v['sdr'][10]:.3f} | "
                  f"P50={v['pct'][50]:.2f} P90={v['pct'][90]:.2f} Max={v['pct'][100]:.2f}px | "
                  f"{time.time()-t0:.0f}s")
            history.append({"epoch": ep, "phase": "P1", "train_loss": tl_loss,
                            "val_loss": v["val_loss"], "mre": v["mre"],
                            "sdr": v["sdr"][5], "sigma": SIGMA_P1})
            save_epoch_log(run_dir, {
                "epoch": ep, "phase": "P1", "in_channels": IN_CHANNELS,
                "train_loss": tl_loss, "val_loss": v["val_loss"],
                "mre": v["mre"], "mre1": v["mre1"], "mre2": v["mre2"],
                "sdr2": v["sdr"][2], "sdr5": v["sdr"][5], "sdr10": v["sdr"][10],
                "p50": v["pct"][50], "p90": v["pct"][90],
                "sigma": SIGMA_P1, "lr": op1.param_groups[-1]["lr"],
                "best_p90_so_far": best_score,
            })

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 2 — cosine sigma curriculum, full model
        # ═════════════════════════════════════════════════════════════════════
        print(f"\n{'='*60}\n  PHASE 2 - curriculum ({P2_EPOCHS} ep, sigma {SIGMA_START}->{SIGMA_END}, cosine)\n{'='*60}")
        set_encoder_grad(model, True)
        enc_pfx = ["enc0", "enc1", "enc2", "enc3", "enc4"]
        op2 = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters()
                        if any(n.startswith(x) for x in enc_pfx)],     "lr": LR_BACKBONE_P2},
            {"params": [p for n, p in model.named_parameters()
                        if not any(n.startswith(x) for x in enc_pfx)], "lr": LR_HEAD_P2},
        ], weight_decay=WEIGHT_DECAY)
        sc2 = torch.optim.lr_scheduler.CosineAnnealingLR(op2, T_max=P2_EPOCHS, eta_min=2e-6)

        no_imp = 0
        for ep_loc in range(1, P2_EPOCHS + 1):
            t0    = time.time()
            ep_g  = P1_EPOCHS + ep_loc
            sigma = sigma_p2(ep_loc - 1, P2_EPOCHS)
            train_ds.set_sigma(sigma)
            val_ds.set_sigma(sigma)

            aw = aux_weight(sigma)
            tl_loss, sub = train_epoch(model, tl, op2, crit, device,
                                       sigma, do_mixup=True, do_aux=has_aux,
                                       aw=aw, scaler=scaler, use_amp=use_amp)
            v = validate(model, vl, crit, device, sigma)
            sc2.step()

            score    = v["pct"][90]
            improved = score < best_score
            print(f"[P2] {ep_g:2d} | sigma={sigma:.2f} | lr={op2.param_groups[-1]['lr']:.2e} | "
                  f"Train={tl_loss:.4f} | Val={v['val_loss']:.4f} | "
                  f"MRE={v['mre']:.2f}px (LM1={v['mre1']:.2f} LM2={v['mre2']:.2f}) | "
                  f"SDR@2/5/10: {v['sdr'][2]:.3f}/{v['sdr'][5]:.3f}/{v['sdr'][10]:.3f} | "
                  f"P50={v['pct'][50]:.2f} P90={v['pct'][90]:.2f} Max={v['pct'][100]:.2f}px"
                  + (" best" if improved else "") + f" | {time.time()-t0:.0f}s")
            history.append({"epoch": ep_g, "phase": "P2", "train_loss": tl_loss,
                            "val_loss": v["val_loss"], "mre": v["mre"],
                            "sdr": v["sdr"][5], "sigma": sigma})
            save_epoch_log(run_dir, {
                "epoch": ep_g, "phase": "P2", "in_channels": IN_CHANNELS,
                "train_loss": tl_loss, "val_loss": v["val_loss"],
                "mre": v["mre"], "mre1": v["mre1"], "mre2": v["mre2"],
                "sdr2": v["sdr"][2], "sdr5": v["sdr"][5], "sdr10": v["sdr"][10],
                "p50": v["pct"][50], "p90": v["pct"][90],
                "sigma": sigma, "lr": op2.param_groups[-1]["lr"],
                "best_p90_so_far": best_score,
            })

            if improved:
                best_score    = score
                best_mre      = score
                best_val_mre  = v["mre"]
                best_val_sdr5 = v["sdr"][5]
                no_imp        = 0
                torch.save(model.state_dict(), os.path.join(run_dir, "best_p2.pth"))
                save_epoch_grid(*v["vis"], epoch=ep_g, save_dir=grids_dir, n_samples=N_VIS)
            else:
                no_imp += 1
                if no_imp >= EARLY_STOP_P2:
                    print("  Warning: P2 early stop")
                    break
            torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))
    else:
        print(f"\n{'='*60}\n  PHASE 2 - skipped (using provided checkpoint)\n{'='*60}")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 3 — subpixel precision squeeze
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PHASE 3 - precision ({P3_EPOCHS} ep, "
          f"sigma {SIGMA_P3_START}->{SIGMA_P3_END}, LR={LR_P3}->1e-6)\n{'='*60}")

    set_encoder_grad(model, True)

    p2_ckpt = p2_checkpoint or os.path.join(run_dir, "best_p2.pth")
    if p2_ckpt and os.path.exists(p2_ckpt):
        model.load_state_dict(torch.load(p2_ckpt, map_location=device, weights_only=True))
        print(f"  Loaded P2 checkpoint: {p2_ckpt}")
        v        = validate(model, vl, crit_p3, device, SIGMA_P3_START)
        best_mre = v["pct"][90]
        print(f"  P2 baseline — P90={v['pct'][90]:.2f}px  MRE={v['mre']:.2f}px")

    op3 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_P3, weight_decay=WEIGHT_DECAY,
    )
    sc3 = torch.optim.lr_scheduler.CosineAnnealingLR(op3, T_max=P3_EPOCHS, eta_min=1e-6)

    best_mre_p3 = best_mre
    no_imp      = 0

    for ep_loc in range(1, P3_EPOCHS + 1):
        t0    = time.time()
        ep_g  = P1_EPOCHS + P2_EPOCHS + ep_loc
        t     = (ep_loc - 1) / max(P3_EPOCHS - 1, 1)
        sigma = SIGMA_P3_START + t * (SIGMA_P3_END - SIGMA_P3_START)
        train_ds.set_sigma(sigma)
        val_ds.set_sigma(sigma)

        tl_loss, sub = train_epoch(model, tl, op3, crit_p3, device,
                                   sigma, do_mixup=False, do_aux=False,
                                   aw=0.0, scaler=scaler, use_amp=use_amp, clip=3.0)
        v = validate(model, vl, crit_p3, device, sigma)
        sc3.step()

        improved = v["pct"][90] < best_mre_p3
        print(f"[P3] {ep_g:2d} | sigma={sigma:.2f} | lr={op3.param_groups[0]['lr']:.2e} | "
              f"Train={tl_loss:.4f} | Val={v['val_loss']:.4f} | "
              f"MRE={v['mre']:.2f}px (LM1={v['mre1']:.2f} LM2={v['mre2']:.2f}) | "
              f"SDR@2/5/10: {v['sdr'][2]:.3f}/{v['sdr'][5]:.3f}/{v['sdr'][10]:.3f} | "
              f"P50={v['pct'][50]:.2f} P90={v['pct'][90]:.2f} Max={v['pct'][100]:.2f}px"
              + (" best" if improved else "") + f" | {time.time()-t0:.0f}s")
        history.append({"epoch": ep_g, "phase": "P3", "train_loss": tl_loss,
                        "val_loss": v["val_loss"], "mre": v["mre"],
                        "sdr": v["sdr"][5], "sigma": sigma})
        save_epoch_log(run_dir, {
            "epoch": ep_g, "phase": "P3", "in_channels": IN_CHANNELS,
            "train_loss": tl_loss, "val_loss": v["val_loss"],
            "mre": v["mre"], "mre1": v["mre1"], "mre2": v["mre2"],
            "sdr2": v["sdr"][2], "sdr5": v["sdr"][5], "sdr10": v["sdr"][10],
            "p50": v["pct"][50], "p90": v["pct"][90],
            "sigma": sigma, "lr": op3.param_groups[0]["lr"],
            "best_p90_so_far": best_mre_p3,
        })

        if improved:
            best_mre_p3   = v["pct"][90]
            best_val_mre  = v["mre"]
            best_val_sdr5 = v["sdr"][5]
            no_imp        = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            save_epoch_grid(*v["vis"], epoch=ep_g, save_dir=grids_dir, n_samples=N_VIS)
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP_P3:
                print("  Warning: P3 early stop")
                break
        torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))

    save_training_curve(history, run_dir)

    # ── test evaluation ────────────────────────────────────────────────────────
    best_model_path = os.path.join(run_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device,
                                         weights_only=True))
    print("\nEvaluating on test set (patients 091-100) …")
    test_v = evaluate_test(model, device, crit_p3)

    # ── results.json ──────────────────────────────────────────────────────────
    results = {
        "in_channels":   IN_CHANNELS,
        "checkpoint":    best_model_path,
        "best_val_p90":  best_mre_p3,
        "best_val_mre":  best_val_mre,
        "best_val_sdr5": best_val_sdr5,
        "test_mre":      test_v["mre"],
        "test_mre_lm1":  test_v["mre1"],
        "test_mre_lm2":  test_v["mre2"],
        "test_sdr2":     test_v["sdr"][2],
        "test_sdr5":     test_v["sdr"][5],
        "test_sdr10":    test_v["sdr"][10],
        "test_p50":      test_v["pct"][50],
        "test_p90":      test_v["pct"][90],
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*44}")
    print("TRAINING COMPLETE")
    print(f"Val  MRE    : {best_val_mre:.2f}px")
    print(f"Val  SDR@5  : {best_val_sdr5*100:.1f}%")
    print(f"Test MRE    : {test_v['mre']:.2f}px")
    print(f"Test MRE LM1: {test_v['mre1']:.2f}px")
    print(f"Test MRE LM2: {test_v['mre2']:.2f}px")
    print(f"Test SDR@2  : {test_v['sdr'][2]*100:.1f}%")
    print(f"Test SDR@5  : {test_v['sdr'][5]*100:.1f}%")
    print(f"Test SDR@10 : {test_v['sdr'][10]*100:.1f}%")
    print(f"Checkpoint  : {best_model_path}")
    print(f"{'='*44}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-checkpoint", default=None,
                        help="Path to a P2 checkpoint to skip straight to P3")
    parser.add_argument("--lm1-coord-weight",   type=float, default=3.0,
                        help="Wing loss weight for LM1 relative to LM2 (default 3.0)")
    parser.add_argument("--lm1-heatmap-weight", type=float, default=2.0,
                        help="BCE+Dice weight for LM1 channel relative to LM2 (default 2.0)")
    parser.add_argument("--sep-margin", type=float, default=0.15,
                        help="Separation loss margin in normalised space (default 0.15 ≈ 38px)")
    parser.add_argument("--sep-weight", type=float, default=5.0,
                        help="Separation loss coefficient (default 5.0)")
    parser.add_argument("--group-norm", action="store_true",
                        help="Use GroupNorm instead of BatchNorm")
    args = parser.parse_args()
    try:
        train(p2_checkpoint=args.p2_checkpoint,
              lm1_coord_weight=args.lm1_coord_weight,
              lm1_heatmap_weight=args.lm1_heatmap_weight,
              sep_margin=args.sep_margin,
              sep_weight=args.sep_weight,
              use_group_norm=args.group_norm)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C)")
        torch.cuda.empty_cache()
        print("You can continue using the terminal normally.")
