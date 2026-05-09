import os
import shutil

# ── paths ─────────────────────────────────────────────────────────────────────
rvip_src  = "data/rvip"                      # where your 201 .nrrd files are
acdc_dir  = "data/acdc-cleaned"

src_img   = f"{acdc_dir}/training/images"
src_mask  = f"{acdc_dir}/training/masks"

# ── output folders ────────────────────────────────────────────────────────────
new_train_img  = f"{acdc_dir}/train_split/images"
new_train_mask = f"{acdc_dir}/train_split/masks"
new_train_rvip = f"{acdc_dir}/train_split/rvip"
new_test_img   = f"{acdc_dir}/test_split/images"
new_test_mask  = f"{acdc_dir}/test_split/masks"
new_test_rvip  = f"{acdc_dir}/test_split/rvip"

for d in [new_train_img, new_train_mask, new_train_rvip,
          new_test_img,  new_test_mask,  new_test_rvip]:
    os.makedirs(d, exist_ok=True)

train_count = 0
test_count  = 0
skip_count  = 0

for fname in sorted(os.listdir(src_img)):
    if not fname.endswith(".nii.gz"):
        continue

    # Extract patient number e.g. patient001_frame01.nii.gz → 1
    patient_num = int(fname.split("_")[0].replace("patient", ""))

    # Derive matching filenames
    base   = fname.replace(".nii.gz", "")   # patient001_frame01
    img_f  = fname                           # patient001_frame01.nii.gz
    mask_f = base + "_gt.nii.gz"            # patient001_frame01_gt.nii.gz
    rvip_f = base + "_rvip.nrrd"            # patient001_frame01_rvip.nrrd

    # Check all three files exist
    img_path  = os.path.join(src_img,  img_f)
    mask_path = os.path.join(src_mask, mask_f)
    rvip_path = os.path.join(rvip_src, rvip_f)   # ← fixed: read from data/rvip/

    if not os.path.exists(mask_path):
        print(f"SKIP (no mask): {fname}")
        skip_count += 1
        continue

    if not os.path.exists(rvip_path):
        print(f"SKIP (no rvip): {fname}")
        skip_count += 1
        continue

    # Split: patients 001-080 → train, 081-100 → test
    if patient_num <= 80:
        shutil.copy(img_path,  os.path.join(new_train_img,  img_f))
        shutil.copy(mask_path, os.path.join(new_train_mask, mask_f))
        shutil.copy(rvip_path, os.path.join(new_train_rvip, rvip_f))
        train_count += 1
    else:
        shutil.copy(img_path,  os.path.join(new_test_img,  img_f))
        shutil.copy(mask_path, os.path.join(new_test_mask, mask_f))
        shutil.copy(rvip_path, os.path.join(new_test_rvip, rvip_f))
        test_count += 1

print(f"\nDone!")
print(f"Training samples : {train_count}")
print(f"Testing  samples : {test_count}")
print(f"Skipped          : {skip_count}")
print(f"\nFinal structure:")
print(f"  {new_train_img}  : {len(os.listdir(new_train_img))} files")
print(f"  {new_train_mask} : {len(os.listdir(new_train_mask))} files")
print(f"  {new_train_rvip} : {len(os.listdir(new_train_rvip))} files")
print(f"  {new_test_img}   : {len(os.listdir(new_test_img))} files")
print(f"  {new_test_mask}  : {len(os.listdir(new_test_mask))} files")
print(f"  {new_test_rvip}  : {len(os.listdir(new_test_rvip))} files")