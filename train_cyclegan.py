"""
train_cyclegan.py — Unpaired ACDC → rv_landmark domain translation
==================================================================
Trains a CycleGAN to translate ACDC MRI images (domain A) so they
look like rv_landmark MRI images (domain B). The resulting G_A2B
generator is then used by generate_translated_dataset.py to produce
synthetic rv_landmark-style training data with real ACDC annotations.

Domains
  A = ACDC         (data/acdc/images/)
  B = rv_landmark  (data/rv_landmark/train_images/)

Only raw MRI images are used here — no landmarks, no seg masks.

Losses (LSGAN)
  G adversarial  : MSE(D(G(x)), 1)
  Cycle          : L1(G_B2A(G_A2B(a)), a) * lambda_cycle
  Identity       : L1(G_A2B(b), b)        * lambda_identity

Training schedule
  Epochs 1–100   : constant LR=2e-4
  Epochs 101–200 : linear LR decay to 0

Usage:
  python train_cyclegan.py
  python train_cyclegan.py --epochs 200 --start-epoch 1
  python train_cyclegan.py --start-epoch 101 \
      --checkpoint-dir checkpoints/cyclegan_TIMESTAMP
"""

import argparse
import itertools
import os
import random
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import cv2
import matplotlib.pyplot as plt

from models.cyclegan import Generator, Discriminator, ImageBuffer


# ── hyperparameters ────────────────────────────────────────────────────────────
LAMBDA_CYCLE    = 10.0
LAMBDA_IDENTITY = 5.0
LR              = 2e-4
BETAS           = (0.5, 0.999)
BATCH_SIZE      = 1
N_EPOCHS        = 200
DECAY_EPOCH     = 100      # start linear LR decay here
SAVE_IMG_EVERY  = 10       # save sample grid every N epochs
SAVE_CKPT_EVERY = 20       # save checkpoint every N epochs
PRINT_EVERY     = 50       # print loss every N iterations
GRAD_CLIP       = 1.0      # gradient clipping norm
NUM_WORKERS     = 2

ACDC_IMG_DIR = "data/acdc/images"
RV_IMG_DIR   = "data/rv_landmark/train_images"


# ── dataset ────────────────────────────────────────────────────────────────────

class MRISliceDataset(Dataset):
    """
    Loads all 2D slices from a directory of .nii.gz volumes.
    Returns single-channel tensors in [-1, 1] (required by CycleGAN).
    Preprocessing: per-slice z-score then clip to [-3, 3] then scale to [-1, 1].
    """

    def __init__(self, image_dir, min_variance=0.01):
        self.slices = []
        files = sorted(f for f in os.listdir(image_dir) if f.endswith(".nii.gz"))
        for fname in files:
            vol = nib.load(os.path.join(image_dir, fname)).get_fdata().astype(np.float32)
            for i in range(vol.shape[2]):
                sl = vol[:, :, i]
                if sl.var() < min_variance:
                    continue
                self.slices.append(sl)
        print(f"  {image_dir}: {len(self.slices)} slices from {len(files)} volumes")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        sl = self.slices[idx].copy()
        sl = cv2.resize(sl, (256, 256), interpolation=cv2.INTER_LINEAR)
        mu, std = sl.mean(), sl.std() + 1e-8
        sl = (sl - mu) / std
        sl = np.clip(sl, -3.0, 3.0) / 3.0   # → [-1, 1]
        return torch.tensor(sl[None], dtype=torch.float32)   # [1, 256, 256]


# ── LR scheduler: linear decay ─────────────────────────────────────────────────

def make_lr_lambda(n_epochs, decay_epoch):
    def lr_lambda(epoch):
        if epoch < decay_epoch:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_epoch) / (n_epochs - decay_epoch))
    return lr_lambda


# ── loss helpers ───────────────────────────────────────────────────────────────

def lsgan_loss_D(real_pred, fake_pred):
    """Discriminator LSGAN loss: MSE(real, 1) + MSE(fake, 0)."""
    return 0.5 * (nn.functional.mse_loss(real_pred, torch.ones_like(real_pred)) +
                  nn.functional.mse_loss(fake_pred, torch.zeros_like(fake_pred)))


def lsgan_loss_G(fake_pred):
    """Generator LSGAN loss: MSE(fake, 1)."""
    return nn.functional.mse_loss(fake_pred, torch.ones_like(fake_pred))


# ── sample grid saver ─────────────────────────────────────────────────────────

def save_sample_grid(real_A, real_B, fake_B, fake_A, rec_A, rec_B,
                     epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def t2np(t):
        return t[0, 0].detach().cpu().numpy()

    imgs = [
        ("Real A (ACDC)",      t2np(real_A)),
        ("Fake B (→rv_lm)",    t2np(fake_B)),
        ("Rec A (cycle)",      t2np(rec_A)),
        ("Real B (rv_lm)",     t2np(real_B)),
        ("Fake A (→ACDC)",     t2np(fake_A)),
        ("Rec B (cycle)",      t2np(rec_B)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (title, img) in zip(axes.flat, imgs):
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Epoch {epoch}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ── training loop ──────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Epochs : {args.start_epoch} → {args.epochs}")

    # ── datasets
    print("\nLoading datasets...")
    ds_A = MRISliceDataset(ACDC_IMG_DIR)
    ds_B = MRISliceDataset(RV_IMG_DIR)
    loader_A = DataLoader(ds_A, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
    loader_B = DataLoader(ds_B, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)

    # ── models
    G_A2B = Generator().to(device)   # ACDC → rv_landmark
    G_B2A = Generator().to(device)   # rv_landmark → ACDC
    D_A   = Discriminator().to(device)
    D_B   = Discriminator().to(device)

    # ── image buffers (stabilise discriminator training)
    buf_A = ImageBuffer()
    buf_B = ImageBuffer()

    # ── optimisers
    opt_G = torch.optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=LR, betas=BETAS,
    )
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=LR, betas=BETAS)
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=LR, betas=BETAS)

    # ── LR schedulers
    lr_lambda = make_lr_lambda(args.epochs, DECAY_EPOCH)
    sch_G   = torch.optim.lr_scheduler.LambdaLR(
        opt_G,   lr_lambda=lambda ep: lr_lambda(ep + args.start_epoch - 1))
    sch_D_A = torch.optim.lr_scheduler.LambdaLR(
        opt_D_A, lr_lambda=lambda ep: lr_lambda(ep + args.start_epoch - 1))
    sch_D_B = torch.optim.lr_scheduler.LambdaLR(
        opt_D_B, lr_lambda=lambda ep: lr_lambda(ep + args.start_epoch - 1))

    # ── resume from checkpoint
    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = args.checkpoint_dir or os.path.join("checkpoints", f"cyclegan_{ts}")
    os.makedirs(ckpt_dir, exist_ok=True)
    samples_dir = os.path.join("cyclegan_results", "samples")

    if args.checkpoint_dir and os.path.isdir(args.checkpoint_dir):
        g_a2b_path = os.path.join(args.checkpoint_dir, "G_A2B_latest.pth")
        g_b2a_path = os.path.join(args.checkpoint_dir, "G_B2A_latest.pth")
        d_a_path   = os.path.join(args.checkpoint_dir, "D_A_latest.pth")
        d_b_path   = os.path.join(args.checkpoint_dir, "D_B_latest.pth")
        if os.path.exists(g_a2b_path):
            G_A2B.load_state_dict(torch.load(g_a2b_path, map_location=device, weights_only=True))
            G_B2A.load_state_dict(torch.load(g_b2a_path, map_location=device, weights_only=True))
            D_A.load_state_dict(torch.load(d_a_path,     map_location=device, weights_only=True))
            D_B.load_state_dict(torch.load(d_b_path,     map_location=device, weights_only=True))
            print(f"Resumed from {args.checkpoint_dir}")

    print(f"Checkpoints → {ckpt_dir}")
    print(f"Samples     → {samples_dir}\n")

    # ── loss history for final plot
    history = {"G": [], "D_A": [], "D_B": [], "cycle": [], "idt": []}

    l1 = nn.L1Loss()

    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_G = epoch_DA = epoch_DB = epoch_cyc = epoch_idt = 0.0
        n_batches = min(len(loader_A), len(loader_B))
        iter_A = iter(loader_A)
        iter_B = iter(loader_B)

        for i in range(n_batches):
            real_A = next(iter_A).to(device)
            real_B = next(iter_B).to(device)

            # ── Generator update ─────────────────────────────────────────────
            G_A2B.train(); G_B2A.train()
            D_A.eval();    D_B.eval()
            opt_G.zero_grad()

            # identity loss (A generator fed domain-B input should be no-op)
            idt_B = G_A2B(real_B)
            idt_A = G_B2A(real_A)
            loss_idt = (l1(idt_B, real_B) + l1(idt_A, real_A)) * LAMBDA_IDENTITY

            # adversarial
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            loss_adv = lsgan_loss_G(D_B(fake_B)) + lsgan_loss_G(D_A(fake_A))

            # cycle consistency
            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)
            loss_cycle = (l1(rec_A, real_A) + l1(rec_B, real_B)) * LAMBDA_CYCLE

            loss_G = loss_adv + loss_idt + loss_cycle
            loss_G.backward()
            nn.utils.clip_grad_norm_(
                list(G_A2B.parameters()) + list(G_B2A.parameters()), GRAD_CLIP
            )
            opt_G.step()

            # ── Discriminator A update ────────────────────────────────────────
            G_A2B.eval(); G_B2A.eval()
            D_A.train()
            opt_D_A.zero_grad()

            fake_A_buf = buf_A.push_and_pop(fake_A.detach())
            loss_D_A   = lsgan_loss_D(D_A(real_A), D_A(fake_A_buf))
            loss_D_A.backward()
            nn.utils.clip_grad_norm_(D_A.parameters(), GRAD_CLIP)
            opt_D_A.step()

            # ── Discriminator B update ────────────────────────────────────────
            D_B.train()
            opt_D_B.zero_grad()

            fake_B_buf = buf_B.push_and_pop(fake_B.detach())
            loss_D_B   = lsgan_loss_D(D_B(real_B), D_B(fake_B_buf))
            loss_D_B.backward()
            nn.utils.clip_grad_norm_(D_B.parameters(), GRAD_CLIP)
            opt_D_B.step()

            epoch_G   += loss_G.item()
            epoch_DA  += loss_D_A.item()
            epoch_DB  += loss_D_B.item()
            epoch_cyc += loss_cycle.item()
            epoch_idt += loss_idt.item()

            if (i + 1) % PRINT_EVERY == 0:
                lr_now = opt_G.param_groups[0]["lr"]
                print(
                    f"[Ep {epoch:3d}/{args.epochs}  it {i+1:4d}/{n_batches}]  "
                    f"G={loss_G.item():.3f}  "
                    f"D_A={loss_D_A.item():.3f}  D_B={loss_D_B.item():.3f}  "
                    f"cycle={loss_cycle.item():.3f}  idt={loss_idt.item():.3f}  "
                    f"lr={lr_now:.2e}"
                )

        # ── epoch-level logging
        nb = max(n_batches, 1)
        history["G"].append(epoch_G / nb)
        history["D_A"].append(epoch_DA / nb)
        history["D_B"].append(epoch_DB / nb)
        history["cycle"].append(epoch_cyc / nb)
        history["idt"].append(epoch_idt / nb)

        lr_now = opt_G.param_groups[0]["lr"]
        print(
            f"\n[Ep {epoch:3d}]  avg  "
            f"G={history['G'][-1]:.4f}  "
            f"D_A={history['D_A'][-1]:.4f}  D_B={history['D_B'][-1]:.4f}  "
            f"cycle={history['cycle'][-1]:.4f}  idt={history['idt'][-1]:.4f}  "
            f"lr={lr_now:.2e}\n"
        )

        sch_G.step(); sch_D_A.step(); sch_D_B.step()

        # ── save sample images
        if epoch % SAVE_IMG_EVERY == 0:
            with torch.no_grad():
                G_A2B.eval(); G_B2A.eval()
                fake_B_s = G_A2B(real_A)
                fake_A_s = G_B2A(real_B)
                rec_A_s  = G_B2A(fake_B_s)
                rec_B_s  = G_A2B(fake_A_s)
            save_sample_grid(
                real_A, real_B, fake_B_s, fake_A_s, rec_A_s, rec_B_s,
                epoch, os.path.join(samples_dir, f"epoch_{epoch:03d}"),
            )
            print(f"  Saved sample grid → {samples_dir}/epoch_{epoch:03d}/epoch_{epoch:03d}.png")

        # ── save checkpoints
        torch.save(G_A2B.state_dict(), os.path.join(ckpt_dir, "G_A2B_latest.pth"))
        torch.save(G_B2A.state_dict(), os.path.join(ckpt_dir, "G_B2A_latest.pth"))
        torch.save(D_A.state_dict(),   os.path.join(ckpt_dir, "D_A_latest.pth"))
        torch.save(D_B.state_dict(),   os.path.join(ckpt_dir, "D_B_latest.pth"))

        if epoch % SAVE_CKPT_EVERY == 0:
            for name, net in [("G_A2B", G_A2B), ("G_B2A", G_B2A),
                               ("D_A",   D_A),   ("D_B",   D_B)]:
                torch.save(net.state_dict(),
                           os.path.join(ckpt_dir, f"{name}_ep{epoch:03d}.pth"))
            print(f"  Saved epoch checkpoint (ep {epoch}) → {ckpt_dir}/")

    # ── final loss curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs_x = list(range(args.start_epoch, args.epochs + 1))
    axes[0].plot(epochs_x, history["G"],    label="G total"); axes[0].set_title("Generator Loss")
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs_x, history["D_A"],  label="D_A")
    axes[1].plot(epochs_x, history["D_B"],  label="D_B")
    axes[1].set_title("Discriminator Loss"); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(epochs_x, history["cycle"], label="Cycle")
    axes[2].plot(epochs_x, history["idt"],   label="Identity")
    axes[2].set_title("Consistency Losses"); axes[2].legend(); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "training_curve.png"), dpi=120)
    plt.close()

    print(f"\n{'='*50}")
    print(f"  CycleGAN training complete")
    print(f"  Generator checkpoint : {ckpt_dir}/G_A2B_latest.pth")
    print(f"  Run next:")
    print(f"    python generate_translated_dataset.py \\")
    print(f"      --checkpoint-dir {ckpt_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int, default=N_EPOCHS)
    parser.add_argument("--start-epoch",     type=int, default=1)
    parser.add_argument("--checkpoint-dir",  type=str, default=None,
                        help="Resume from this directory (also used as output dir)")
    args = parser.parse_args()
    train(args)
