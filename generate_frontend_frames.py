"""
Generate frontend-friendly cine frames with segmentation overlay and landmarks.

This is separate from inference.py on purpose: inference.py is useful for model
debugging because it compares predicted vs ground-truth points and heatmaps.
This script is for presentation/frontend output only:

  - base MRI slice
  - segmentation mask overlay
  - predicted landmark points only
  - optional JSON manifest for viewer/index.html

Example:
    python generate_frontend_frames.py ^
        --image data/lv-landmark/Testing/images/DET0026101.nii.gz ^
        --segmentation DET0026101_segmentation.nii ^
        --checkpoint checkpoints/YOUR_RUN/best_model.pth ^
        --out viewer_frames/DET0026101 ^
        --case-id DET0026101 ^
        --all-slices
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import nibabel as nib
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: nibabel. Install project requirements first:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc

from inference import load_model, predict_landmarks


def normalize_for_display(slice_2d):
    """Robust grayscale normalization for MRI display."""
    image = slice_2d.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip((image - low) / (high - low), 0, 1)
    return image


def resize_mask_to_image(mask_2d, shape):
    """Resize a mask slice to match the image slice if needed."""
    target_h, target_w = shape
    if mask_2d.shape == shape:
        return mask_2d
    return cv2.resize(
        mask_2d.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )


def squeeze_volume(volume, name):
    """Accept plain 3-D NIfTI volumes and 4-D volumes with one channel/timepoint."""
    volume = np.asarray(volume)
    squeezed = np.squeeze(volume)
    if squeezed.ndim != 3:
        raise ValueError(
            f"{name} must be a 3-D volume, or a 4-D volume with singleton extra axes. "
            f"Got shape {volume.shape}."
        )
    return squeezed


def auto_slices(image_vol, seg_vol, min_mask_pixels):
    """Choose slices that have segmentation, otherwise image content."""
    n_slices = min(image_vol.shape[2], seg_vol.shape[2])
    masked = [
        i
        for i in range(n_slices)
        if np.count_nonzero(seg_vol[:, :, i] > 0) >= min_mask_pixels
    ]
    if masked:
        return masked

    return [
        i
        for i in range(n_slices)
        if float(np.nanvar(image_vol[:, :, i])) > 0.001
    ]


def parse_slices(raw):
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def draw_frame(
    image_2d,
    mask_2d,
    coords,
    slice_idx,
    out_path,
    mask_alpha,
    point_size,
):
    display_image = normalize_for_display(image_2d)
    display_mask = mask_2d > 0

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(display_image, cmap="gray", aspect="auto")

    if np.any(display_mask):
        overlay = np.zeros((*display_mask.shape, 4), dtype=np.float32)
        overlay[..., 0] = 0.05
        overlay[..., 1] = 0.90
        overlay[..., 2] = 0.68
        overlay[..., 3] = display_mask.astype(np.float32) * mask_alpha
        ax.imshow(overlay, aspect="auto")

    ax.scatter(
        [coords[0]],
        [coords[1]],
        c="#ff3b30",
        s=point_size,
        edgecolors="white",
        linewidths=1.2,
        zorder=4,
    )
    ax.scatter(
        [coords[2]],
        [coords[3]],
        c="#2ecbff",
        s=point_size,
        edgecolors="white",
        linewidths=1.2,
        zorder=4,
    )
    ax.plot(
        [coords[0], coords[2]],
        [coords[1], coords[3]],
        color="white",
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
        zorder=3,
    )

    ax.set_title(f"Slice {slice_idx}", fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Export presentation frames with mask overlay and predicted points only."
    )
    parser.add_argument("--image", required=True, help="Raw MRI volume (.nii or .nii.gz).")
    parser.add_argument("--segmentation", required=True, help="Segmentation volume (.nii or .nii.gz).")
    parser.add_argument("--checkpoint", required=True, help="Landmark model checkpoint (.pth).")
    parser.add_argument("--out", default="viewer_frames", help="Output directory.")
    parser.add_argument("--case-id", default=None, help="Case id for filenames and manifest.")
    parser.add_argument("--slices", default=None, help="Comma/range list, for example 0,1,2,4-8.")
    parser.add_argument("--all-slices", action="store_true", help="Export every nonblank or masked slice.")
    parser.add_argument("--min-mask-pixels", type=int, default=10)
    parser.add_argument("--mask-alpha", type=float, default=0.38)
    parser.add_argument("--point-size", type=float, default=70)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path. Use .js for viewer/manifest.js or .json for JSON.",
    )
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA for faster prediction.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    image_path = Path(args.image)
    seg_path = Path(args.segmentation)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_id = args.case_id or image_path.name.split(".nii")[0]

    image_vol = squeeze_volume(
        nib.load(str(image_path)).get_fdata().astype(np.float32),
        "image",
    )
    seg_vol = squeeze_volume(
        nib.load(str(seg_path)).get_fdata().astype(np.float32),
        "segmentation",
    )

    if image_vol.ndim != 3 or seg_vol.ndim != 3:
        raise ValueError("Both image and segmentation inputs must be 3-D volumes.")

    if args.slices:
        slice_indices = parse_slices(args.slices)
    elif args.all_slices:
        slice_indices = auto_slices(image_vol, seg_vol, args.min_mask_pixels)
    else:
        slice_indices = auto_slices(image_vol, seg_vol, args.min_mask_pixels)[:5]

    max_slice = min(image_vol.shape[2], seg_vol.shape[2]) - 1
    slice_indices = [i for i in slice_indices if 0 <= i <= max_slice]
    if not slice_indices:
        raise ValueError("No valid slices selected.")

    model = load_model(args.checkpoint)
    use_tta = not args.no_tta

    generated_frames = []
    for slice_idx in slice_indices:
        image_2d = image_vol[:, :, slice_idx]
        mask_2d = resize_mask_to_image(seg_vol[:, :, slice_idx], image_2d.shape)
        coords, _heatmap = predict_landmarks(image_2d, model=model, use_tta=use_tta)

        filename = f"{case_id}_slice{slice_idx:03d}.png"
        out_path = out_dir / filename
        draw_frame(
            image_2d=image_2d,
            mask_2d=mask_2d,
            coords=coords,
            slice_idx=slice_idx,
            out_path=out_path,
            mask_alpha=args.mask_alpha,
            point_size=args.point_size,
        )
        generated_frames.append({"slice": slice_idx, "path": out_path})
        print(f"Saved {out_path}")

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_frames = [
            {
                "slice": frame["slice"],
                "image": os.path.relpath(frame["path"], manifest_path.parent).replace("\\", "/"),
            }
            for frame in generated_frames
        ]
        payload = [{"id": case_id, "label": case_id, "frames": manifest_frames}]
        with manifest_path.open("w", encoding="utf-8") as f:
            if manifest_path.suffix.lower() == ".js":
                f.write("window.RESULT_FRAMES = ")
                json.dump(payload, f, indent=2)
                f.write(";\n")
            else:
                json.dump(payload, f, indent=2)
                f.write("\n")
        print(f"Saved manifest {manifest_path}")


if __name__ == "__main__":
    main()
