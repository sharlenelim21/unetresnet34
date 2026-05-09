import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

img  = nib.load("data/lv-landmark/Training/images/DET0000201.nii.gz").get_fdata()
mask = nib.load("data/lv-landmark/Training/masks/DET0000201.nii.gz").get_fdata()

slice_idx = 2
img_slice  = img[:, :, slice_idx]
mask_slice = mask[:, :, slice_idx]

# Extract corrected landmarks
p1 = np.argwhere(mask_slice == 1).mean(axis=0)  # row, col
p2 = np.argwhere(mask_slice == 2).mean(axis=0)

plt.imshow(img_slice, cmap="gray")
plt.scatter(p1[1], p1[0], c="red",  s=100, label="LM1 (label 1)")
plt.scatter(p2[1], p2[0], c="blue", s=100, label="LM2 (label 2)")
plt.legend()
plt.title("Do these dots land on the RV insertion points?")
plt.savefig("gt_check.png")