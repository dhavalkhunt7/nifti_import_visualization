# %% calculating dice for predictions 2d  multi gpu
import math
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import nibabel as nib

#%% function to compute the dice coefficient
def dc(p, g):
    dice = np.sum(p[g == 1]) * 2.0 / (np.sum(p) + np.sum(g))
    if math.isnan(dice):
        dice = 0.0
    # print('Dice similarity score is {}'.format(dice))
    return dice

#%%
img_dir= Path("../data/resultTs/Rat0102.nii.gz")

img = nib.load(img_dir)

#%% loading the data

img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task600_rat")
pred_dir = img_dir / "resultTs/Rat0113.nii.gz"
print(img_dir)

img = nib.load(pred_dir)
# for i in pred_dir.glob("*"):
#     print(i.name)


#%%
pred_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task505_BrainTumour/test_results_2d")
grd_t_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task505_BrainTumour/labelsTs")


dice_all = []
for i in grd_t_dir.glob("*.nii.gz"):
    gt_name = i.name
    # print(i.name)
    for j in pred_dir.glob("*.nii.gz"):
        pre_name = j.name
        if pre_name == gt_name:
            pred_data = nib.load(j).get_fdata()
            gt_data = nib.load(i).get_fdata()
            dice_co = []

            for c in range(1, 4):
                temp_pred = np.zeros_like(pred_data)
                temp_gt = np.zeros_like(gt_data)

                temp_pred[pred_data == c] = 1
                temp_gt[gt_data == c] = 1
                dice = dc(temp_pred, temp_gt)
                dice_co.append(dice)
            dice_all.append(np.mean(dice_co))
            coeff = np.mean(dice_co)
            print("for" + pre_name + " dc : " + str(coeff))

# %%
np.mean(dice_all)
