import nibabel as nib
import numpy as np
import os

mask_dir = "data/lv-landmark/Training/masks"
files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")])

for fname in files[:5]:  # check first 5 patients
    mask = nib.load(os.path.join(mask_dir, fname)).get_fdata()
    pts1 = np.argwhere(mask == 1)
    pts2 = np.argwhere(mask == 2)
    print(f"{fname}: label1={len(pts1)} pixels, label2={len(pts2)} pixels")