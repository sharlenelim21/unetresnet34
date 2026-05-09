import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

mask_dir = r"data/lv-landmark/Testing/masks"

files = os.listdir(mask_dir)

print(files[:5])  # optional: check filenames

mask = nib.load(os.path.join(mask_dir, files[0])).get_fdata()

for i in range(mask.shape[2]):
    slc = mask[:, :, i]

    if np.any(slc == 1) and np.any(slc == 2):
        plt.figure(figsize=(6,6))
        plt.imshow(slc, cmap='nipy_spectral')
        plt.colorbar()
        plt.title(f"Slice {i}")
        plt.savefig("mask_check.png")

        print(f"Saved mask_check.png for slice {i}")
        break