import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load one image and check shape
img = nib.load("data/acdc/images/patient001_frame01.nii.gz").get_fdata()
print(f"Image shape: {img.shape}")

# Load one seg mask and check labels
seg = nib.load("data/acdc/masks/patient001_frame01.nii.gz").get_fdata()
print(f"Seg shape:   {seg.shape}")
print(f"Seg labels:  {np.unique(np.round(seg))}")