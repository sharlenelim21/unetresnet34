"""
Sanity-check rv_landmark train_gt blob-extraction.

Loads a few train volumes, runs the same blob-extraction logic that the
Dataset will use, and overlays the extracted (LM1, LM2) on the MRI.

If the overlays look right (both points on the RV/septum boundary,
LM1 above LM2), the Dataset can safely be trained against this GT.

Run:
    python scripts/check_rv_gt.py
Outputs:
    debug/rv_gt_check/<volume>_slice<idx>.png
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass

IMG_DIR  = "data/rv_landmark/train_images"
GT_DIR   = "data/rv_landmark/train_gt"
SEG_DIR  = "data/rv_landmark/train_seg_multi"
OUT_DIR  = "debug/rv_gt_check"
N_VOLS   = 5
THRESH_FRAC = 0.1   # fraction of slice max
MIN_AREA    = 3     # pixels


def extract_two_blobs(gt_slice):
    """Return (x1, y1, x2, y2) sorted by y, or None."""
    mx = gt_slice.max()
    if mx <= 0:
        return None
    bw = gt_slice > THRESH_FRAC * mx
    lbl, n = label(bw)
    if n == 0:
        return None
    # collect components with sufficient area
    comps = []
    for k in range(1, n + 1):
        area = int((lbl == k).sum())
        if area >= MIN_AREA:
            cy, cx = center_of_mass(gt_slice, lbl, k)
            comps.append((float(cx), float(cy), area))
    if len(comps) < 2:
        return None
    # take the two largest by area
    comps.sort(key=lambda t: -t[2])
    (x1, y1, _), (x2, y2, _) = comps[0], comps[1]
    # sort by y so LM1 is superior (smaller y)
    if y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".nii.gz"))
    rng = np.random.RandomState(42)
    pick = rng.choice(len(files), size=min(N_VOLS, len(files)), replace=False)

    n_total, n_ok, n_dropped = 0, 0, 0
    comp_counts = {}

    for vi in pick:
        fname = files[vi]
        img  = nib.load(os.path.join(IMG_DIR, fname)).get_fdata().astype(np.float32)
        gt   = nib.load(os.path.join(GT_DIR,  fname)).get_fdata().astype(np.float32)
        seg_path = os.path.join(SEG_DIR, fname)
        seg = nib.load(seg_path).get_fdata().astype(np.float32) if os.path.exists(seg_path) else None

        nz = min(img.shape[2], gt.shape[2])
        annotated_slices = []
        for i in range(nz):
            if gt[:, :, i].max() > 0:
                annotated_slices.append(i)

        # Inspect every annotated slice (or up to 6)
        for sl in annotated_slices[:6]:
            n_total += 1
            gs = gt[:, :, sl]
            # count blobs for diagnostics
            bw = gs > THRESH_FRAC * gs.max()
            _, nblobs = label(bw)
            comp_counts[nblobs] = comp_counts.get(nblobs, 0) + 1

            coords = extract_two_blobs(gs)
            if coords is None:
                n_dropped += 1
                continue
            n_ok += 1

            img_2d = img[:, :, sl]
            seg_2d = seg[:, :, sl] if seg is not None else None

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_2d, cmap="gray")
            axes[0].scatter([coords[0]], [coords[1]], c="red",        s=80, edgecolors="white", label="LM1 (superior)")
            axes[0].scatter([coords[2]], [coords[3]], c="deepskyblue", s=80, edgecolors="white", label="LM2 (inferior)")
            axes[0].set_title(f"{fname}  slice {sl}  ({nblobs} blobs)")
            axes[0].legend(fontsize=8)
            axes[0].axis("off")

            axes[1].imshow(img_2d, cmap="gray")
            axes[1].imshow(gs, cmap="hot", alpha=0.5)
            axes[1].scatter([coords[0]], [coords[1]], c="red",        s=80, edgecolors="white")
            axes[1].scatter([coords[2]], [coords[3]], c="deepskyblue", s=80, edgecolors="white")
            axes[1].set_title("GT heatmap overlay")
            axes[1].axis("off")

            if seg_2d is not None:
                axes[2].imshow(img_2d, cmap="gray")
                axes[2].imshow(seg_2d, alpha=0.4, cmap="nipy_spectral", vmin=0, vmax=3)
                axes[2].scatter([coords[0]], [coords[1]], c="red",        s=80, edgecolors="white")
                axes[2].scatter([coords[2]], [coords[3]], c="deepskyblue", s=80, edgecolors="white")
                axes[2].set_title("Seg-multi overlay")
            else:
                axes[2].text(0.5, 0.5, "no seg", ha="center", va="center")
            axes[2].axis("off")

            plt.tight_layout()
            out = os.path.join(OUT_DIR, f"{fname.replace('.nii.gz','')}_slice{sl:03d}.png")
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()

    print(f"\nInspected {n_total} annotated slices across {len(pick)} volumes")
    print(f"  OK    : {n_ok}")
    print(f"  Drop  : {n_dropped}")
    print(f"  Blob-count histogram: {sorted(comp_counts.items())}")
    print(f"\nOverlays -> {OUT_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
