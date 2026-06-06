"""
generate_translated_dataset.py
================================
Uses a trained CycleGAN G_A2B generator to translate all ACDC MRI slices
to rv_landmark scanner style, then saves them as a new dataset.

Output structure (ready to plug into ACDCLandmarkDataset or train_acdc_*.py):
  data/acdc_translated/images/   — translated MRI volumes (.nii.gz)
  data/acdc_translated/masks/    — original ACDC seg masks (unchanged copy)
  data/acdc_translated/points/   — original ACDC RVIP annotations (unchanged copy)

Preprocessing for the generator (must match train_cyclegan.py):
  z-score → clip [-3, 3] → divide by 3  → range [-1, 1]

Post-processing back to NIfTI:
  The translated values are in [-1, 1]. We rescale back to a plausible
  HU/intensity range by matching the per-slice mean and std of the original
  ACDC slice, so downstream z-score normalisation in the landmark dataset
  sees values in the same ballpark as genuine rv_landmark data.

Usage:
  python generate_translated_dataset.py \
      --checkpoint-dir checkpoints/cyclegan_TIMESTAMP

  # evaluate translation quality on a few samples first (no --translate-all):
  python generate_translated_dataset.py \
      --checkpoint-dir checkpoints/cyclegan_TIMESTAMP \
      --n-preview 10
"""

import argparse
import os
import shutil

import cv2
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.cyclegan import Generator


ACDC_IMG_DIR   = "data/acdc/images"
ACDC_MASK_DIR  = "data/acdc/masks"
ACDC_POINT_DIR = "data/acdc/points"

OUT_IMG_DIR    = "data/acdc_translated/images"
OUT_MASK_DIR   = "data/acdc_translated/masks"
OUT_POINT_DIR  = "data/acdc_translated/points"

PREVIEW_DIR    = "cyclegan_results/translation_preview"


def load_generator(checkpoint_dir, device):
    path = os.path.join(checkpoint_dir, "G_A2B_latest.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"G_A2B_latest.pth not found in {checkpoint_dir}.\n"
            f"Check that training completed and the directory is correct."
        )
    G = Generator().to(device)
    G.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    G.eval()
    print(f"Loaded G_A2B from {path}")
    return G


def preprocess_slice(sl):
    """z-score → clip [-3,3] → /3 → tensor [1,1,256,256] in [-1,1]."""
    sl = cv2.resize(sl.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
    mu, std = sl.mean(), sl.std() + 1e-8
    sl = (sl - mu) / std
    sl = np.clip(sl, -3.0, 3.0) / 3.0
    return torch.tensor(sl[None, None], dtype=torch.float32)


@torch.no_grad()
def translate_volume(vol, G, device):
    """
    Translate every slice of a 3D volume (H, W, S).
    Returns translated volume as numpy float32 in the same value range
    as the original (stats matched per-slice so ACDCLandmarkDataset's
    z-score normalisation works correctly downstream).
    """
    H, W, S = vol.shape
    out = np.zeros_like(vol)

    for i in range(S):
        sl_orig = vol[:, :, i].astype(np.float32)
        if sl_orig.var() < 0.01:
            out[:, :, i] = sl_orig   # blank slice — no translation needed
            continue

        inp = preprocess_slice(sl_orig).to(device)
        translated = G(inp)[0, 0].cpu().numpy()   # [-1, 1]

        # rescale to match original slice statistics
        mu_orig, std_orig = sl_orig.mean(), sl_orig.std() + 1e-8
        t_mu, t_std = translated.mean(), translated.std() + 1e-8
        translated = (translated - t_mu) / t_std * std_orig + mu_orig

        # resize back to original spatial dims if needed
        if (H, W) != (256, 256):
            translated = cv2.resize(translated, (W, H), interpolation=cv2.INTER_LINEAR)

        out[:, :, i] = translated

    return out


def save_preview(orig_sl, trans_sl, idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_sl,  cmap="gray"); axes[0].set_title("Original (ACDC)"); axes[0].axis("off")
    axes[1].imshow(trans_sl, cmap="gray"); axes[1].set_title("Translated (→rv_lm)"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"preview_{idx:04d}.png"), dpi=120, bbox_inches="tight")
    plt.close()


def copy_dir(src, dst):
    """Copy all .nii.gz files from src to dst."""
    os.makedirs(dst, exist_ok=True)
    files = [f for f in os.listdir(src) if f.endswith(".nii.gz")]
    for fname in files:
        shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))
    print(f"  Copied {len(files)} files: {src} → {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True,
                        help="CycleGAN checkpoint directory (contains G_A2B_latest.pth)")
    parser.add_argument("--n-preview",  type=int, default=0,
                        help="Save N side-by-side preview PNGs without writing the dataset")
    parser.add_argument("--translate-all", action="store_true", default=True,
                        help="Translate all ACDC volumes and write output dataset (default True)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    G = load_generator(args.checkpoint_dir, device)

    files = sorted(f for f in os.listdir(ACDC_IMG_DIR) if f.endswith(".nii.gz"))
    print(f"ACDC volumes to translate: {len(files)}")

    # ── preview mode
    if args.n_preview > 0:
        print(f"\nGenerating {args.n_preview} preview images → {PREVIEW_DIR}/")
        count = 0
        for fname in files:
            vol = nib.load(os.path.join(ACDC_IMG_DIR, fname)).get_fdata().astype(np.float32)
            for i in range(vol.shape[2]):
                sl = vol[:, :, i]
                if sl.var() < 0.01:
                    continue
                sl_r = cv2.resize(sl, (256, 256), interpolation=cv2.INTER_LINEAR)
                inp  = preprocess_slice(sl)
                with torch.no_grad():
                    trans = G(inp.to(device))[0, 0].cpu().numpy()
                trans_r = cv2.resize(trans, (256, 256), interpolation=cv2.INTER_LINEAR)
                save_preview(sl_r, trans_r, count, PREVIEW_DIR)
                count += 1
                if count >= args.n_preview:
                    break
            if count >= args.n_preview:
                break
        print(f"Previews saved → {PREVIEW_DIR}/")
        if not args.translate_all:
            return

    # ── full dataset translation
    print(f"\nTranslating {len(files)} volumes...")
    os.makedirs(OUT_IMG_DIR, exist_ok=True)

    total_slices = 0
    for idx, fname in enumerate(files, 1):
        img_path = os.path.join(ACDC_IMG_DIR, fname)
        vol      = nib.load(img_path)
        vol_data = vol.get_fdata().astype(np.float32)

        translated = translate_volume(vol_data, G, device)
        total_slices += vol_data.shape[2]

        out_img = nib.Nifti1Image(translated.astype(np.float32), vol.affine, vol.header)
        nib.save(out_img, os.path.join(OUT_IMG_DIR, fname))

        if idx % 10 == 0 or idx == len(files):
            print(f"  [{idx:3d}/{len(files)}] {fname}  "
                  f"shape={vol_data.shape}  total_slices={total_slices}")

    # ── copy masks and points unchanged
    print("\nCopying seg masks and RVIP annotations...")
    copy_dir(ACDC_MASK_DIR,  OUT_MASK_DIR)
    copy_dir(ACDC_POINT_DIR, OUT_POINT_DIR)

    print(f"\n{'='*55}")
    print(f"  Dataset ready at data/acdc_translated/")
    print(f"  Images   : {OUT_IMG_DIR}  ({len(files)} volumes, {total_slices} slices)")
    print(f"  Masks    : {OUT_MASK_DIR}")
    print(f"  Points   : {OUT_POINT_DIR}")
    print(f"")
    print(f"  Next steps:")
    print(f"  1. Retrain landmark detector on translated data:")
    print(f"       python train_acdc_2ch.py")
    print(f"     (edit TRAIN_IMAGE_DIR/MASK_DIR/POINT_DIR to point at data/acdc_translated/)")
    print(f"  2. Fine-tune on real rv_landmark training data:")
    print(f"       python finetune_rv.py \\")
    print(f"         --base-checkpoint checkpoints/<new_run>/best_model.pth \\")
    print(f"         --in-channels 2 --epochs-p1 4 --epochs-p2 6 --epochs-p3 15")
    print(f"  3. Evaluate:")
    print(f"       python inference_rv.py \\")
    print(f"         --checkpoint checkpoints/<finetune_run>/best_model.pth \\")
    print(f"         --in-channels 2 --seg-dir data/rv_landmark/test_seg_multi --eval")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
