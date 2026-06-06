import os

img_dir = 'data/acdc/images'
pts_dir = 'data/acdc/points'
msk_dir = 'data/acdc/masks'

# Check exact filenames
print('Image files:')
for f in sorted(os.listdir(img_dir))[:5]:
    print(f'  [{f}]')

print('Points files:')
for f in sorted(os.listdir(pts_dir))[:5]:
    print(f'  [{f}]')

print('Mask files:')
for f in sorted(os.listdir(msk_dir))[:5]:
    print(f'  [{f}]')

# Test the find functions
from dataset.acdc_landmark_dataset import _find_image, _find_mask, _find_rvip

pid = 'patient001'
for frame in ['ED', 'ES']:
    img  = _find_image(img_dir, pid, frame)
    msk  = _find_mask(msk_dir,  pid, frame)
    rvip = _find_rvip(pts_dir,  pid, frame)
    print(f'{pid}/{frame}:')
    print(f'  img:  {img}')
    print(f'  msk:  {msk}')
    print(f'  rvip: {rvip}')