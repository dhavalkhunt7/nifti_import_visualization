# %%
from pathlib import Path

import numpy as np
import nibabel as nib

# %%data import paths
img_dir = Path("../../../Documents/data/Adrian_chamba_strokes_data/Human")
print(img_dir)

# %% rat data
rat_img_dir = Path("../../../Documents/data/Adrian_chamba_strokes_data/Rats24h")

# %%
# count = 0
# all_values =[]
for i in img_dir.glob("*"):
    # print(i.name)
    new_dir = img_dir / i.name
    if (i.name == 'ANON178033039144'):
        for j in new_dir.glob("*"):
            # print(j.name)
            if (j.name == 'GroundTrouth.nii'):
                print("ground truth")
                img_gt = nib.load(j)

            elif (j.name == 'Masked_ADC.nii'):
                print("masked _adc")
                img_adc = nib.load(j)

            elif (j.name == 'T2_norm.nii'):
                print("te norm")
                img_t2 = nib.load(j)
    # img = nib.load(i)
    # data = img.get_fdata()
    # print(img)
    # unique_values = np.unique(data)
    # all_values.append(unique_values)
    # count += 1

# print(count)

# %%
gt_data = img_gt.get_fdata()
adc_data = img_adc.get_fdata()
t2_data = img_t2.get_fdata()

# %%
adc_data.shape
gt_data.shape
t2_data.shape

# %%
np.unique()

# %% for rat dataset
for i in rat_img_dir.glob("*"):
    # print(i.name)
    new_dir = rat_img_dir / i.name
    if (i.name == 'Rat102-24h'):
        for j in new_dir.glob("*"):
            print(j.name)
            if (j.name == 'GroundTruth24h.nii'):
                print("ground truth")
                rat_img_gt = nib.load(j)

            elif (j.name == 'Masked_ADC.nii'):
                print("masked _adc")
                rat_img_adc = nib.load(j)

            elif (j.name == 'Masked_T2.nii'):
                print("t2 masked")
                rat_img_t2 = nib.load(j)

# %%

rat_gt_data = rat_img_gt.get_fdata()
rat_adc_data = rat_img_adc.get_fdata()
rat_t2_data = rat_img_t2.get_fdata()

# %%
rat_gt_data.shape

np.unique(rat_gt_data)

# %%
from pathlib import Path
import nibabel as nib
import numpy as np

# %%
database = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task505_BrainTumour/PETMRI")
pred_dir = database / "result_2d"
gt_dir = database / "labelsTs"
darfluid_dir = database / "imagesTs"

# %% check if 8079.nii.gz, 175, 8051 has more than one label in pred_dir
for i in pred_dir.glob("*"):
    # print(i.name)
    if i.name == '8079.nii.gz':
        img = nib.load(i)
        data = img.get_fdata()
        # print i.name and unique_values
        print(i.name)
        print(np.unique(data))
    elif i.name == '175.nii.gz':
        img = nib.load(i)
        data = img.get_fdata()
        # print i.name and unique_values
        print(i.name)
        print(np.unique(data))
    elif i.name == '8051.nii.gz':
        img = nib.load(i)
        data = img.get_fdata()
        # print i.name and unique_values
        print(i.name)
        print(np.unique(data))