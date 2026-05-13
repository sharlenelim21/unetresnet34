# Check what's in your M&Ms-2 directory
ls data/mm2_processed/

# Check one sample to see structure
ls data/mm2_processed/train/
# or whatever the folder structure is

# Check what labels the seg masks have
python -c "
import nibabel as nib
import numpy as np
import os

# Find first seg mask file
for root, dirs, files in os.walk('data/mm2s_cleaned'):
    for f in files:
        if 'seg' in f.lower() or 'mask' in f.lower() or 'gt' in f.lower():
            path = os.path.join(root, f)
            mask = nib.load(path).get_fdata()
            print(f'File: {path}')
            print(f'Shape: {mask.shape}')
            print(f'Labels: {np.unique(np.round(mask))}')
            break
    else:
        continue
    break
