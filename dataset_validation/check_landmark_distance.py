import nibabel as nib
import numpy as np
import os

mask_dir = "data/lv-landmark/Training/masks"
files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")])

for fname in files[:3]:
    mask = nib.load(os.path.join(mask_dir, fname)).get_fdata()
    for i in range(mask.shape[2]):
        slc = mask[:,:,i]
        pts1 = np.argwhere(slc == 1)
        pts2 = np.argwhere(slc == 2)
        if len(pts1) > 0 and len(pts2) > 0:
            c1 = pts1.mean(axis=0)
            c2 = pts2.mean(axis=0)
            dist = np.linalg.norm(c1 - c2)
            print(f"{fname} slice {i}: dist={dist:.1f}px  c1={c1}  c2={c2}")