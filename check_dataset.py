import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

gt_dir    = "data/testing_nifti_GT"
img_dir   = "data/rv_landmark/test_images"

fname = "DET0000301.nii.gz"
gt    = nib.load(os.path.join(gt_dir, fname)).get_fdata()
img   = nib.load(os.path.join(img_dir, fname)).get_fdata()

print(f"GT shape:  {gt.shape}")
print(f"Img shape: {img.shape}")

# Find a slice with annotations
for i in range(gt.shape[2]):
    slc = gt[:, :, i]
    unique = np.unique(slc)
    if len(unique) > 1:
        print(f"\nSlice {i} labels: {unique}")
        print(f"  Label 1 pixels: {np.sum(slc == 1)}")
        print(f"  Label 2 pixels: {np.sum(slc == 2)}")

        # Find centroids
        p1 = np.argwhere(slc == 1).mean(axis=0)
        p2 = np.argwhere(slc == 2).mean(axis=0)
        print(f"  Label 1 centroid (row,col): {p1}")
        print(f"  Label 2 centroid (row,col): {p2}")

        # Visualise
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img[:, :, i], cmap='gray')
        axes[0].scatter(p1[1], p1[0], c='red',  s=100, label='Label 1')
        axes[0].scatter(p2[1], p2[0], c='blue', s=100, label='Label 2')
        axes[0].legend()
        axes[0].set_title(f'MRI + GT (slice {i})')
        axes[1].imshow(slc, cmap='nipy_spectral', vmin=0, vmax=2)
        axes[1].set_title('GT mask')
        plt.savefig("gt_check.png")
        print(f"  Saved gt_check.png")
        break