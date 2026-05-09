import nrrd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load all three for one patient
img     = nib.load("data/acdc-cleaned/train_split/images/patient001_frame01.nii.gz").get_fdata()
seg     = nib.load("data/acdc-cleaned/train_split/masks/patient001_frame01_gt.nii.gz").get_fdata()
rvip, _ = nrrd.read("data/acdc-cleaned/train_split/rvip/patient001_frame01_rvip.nrrd")

# Fix floating point labels
seg = np.round(seg).astype(int)

print(f"Image shape : {img.shape}")
print(f"Seg shape   : {seg.shape}")
print(f"RVIP shape  : {rvip.shape}")
print(f"Seg labels  : {np.unique(seg)}")
print(f"RVIP labels : {np.unique(rvip)}")

# Find a slice with RVIP annotations
found = False
for i in range(rvip.shape[2]):
    slc = rvip[:, :, i]
    if np.any(slc == 1) and np.any(slc == 2):
        p1 = np.argwhere(slc == 1).mean(axis=0)  # upper RVIP
        p2 = np.argwhere(slc == 2).mean(axis=0)  # lower RVIP

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: MRI + RVIP points
        axes[0].imshow(img[:, :, i], cmap='gray')
        axes[0].scatter(p1[1], p1[0], c='red',  s=100, label='Upper RVIP (label 1)')
        axes[0].scatter(p2[1], p2[0], c='blue', s=100, label='Lower RVIP (label 2)')
        axes[0].set_title(f'MRI + RVIP (slice {i})')
        axes[0].legend(fontsize=8)
        axes[0].axis('off')

        # Panel 2: Seg mask + RVIP points
        axes[1].imshow(seg[:, :, i], cmap='nipy_spectral', vmin=0, vmax=3)
        axes[1].scatter(p1[1], p1[0], c='red',  s=100)
        axes[1].scatter(p2[1], p2[0], c='blue', s=100)
        axes[1].set_title('Seg mask\n(0=bg 1=RV 2=Myo 3=LV)')
        axes[1].axis('off')

        # Panel 3: MRI + seg overlay + RVIP points
        axes[2].imshow(img[:, :, i], cmap='gray')
        axes[2].imshow(seg[:, :, i], alpha=0.35, cmap='nipy_spectral', vmin=0, vmax=3)
        axes[2].scatter(p1[1], p1[0], c='red',  s=100, label='Upper RVIP')
        axes[2].scatter(p2[1], p2[0], c='blue', s=100, label='Lower RVIP')
        axes[2].set_title('MRI + Seg overlay + RVIP')
        axes[2].legend(fontsize=8)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig("acdc_validation.png", dpi=150, bbox_inches='tight')
        print(f"\nSaved acdc_validation.png for slice {i}")
        print(f"Upper RVIP (label 1) centroid: row={p1[0]:.1f}, col={p1[1]:.1f}")
        print(f"Lower RVIP (label 2) centroid: row={p2[0]:.1f}, col={p2[1]:.1f}")
        found = True
        break

if not found:
    print("No slice found with both RVIP labels — check the rvip file")