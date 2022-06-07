import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# %%  code to calculate dice coefficient
import numpy as np

pred = np.zeros((100, 100, 100), dtype=int)
gt = np.zeros((100, 100, 100), dtype=int)

pred[30:70, 40:80, 30:40] = 1
gt[30:70, 40:80, 30:40] = 1

dice = np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))

print('Dice similarity score is {}'.format(dice))

# %% loading data
pred_dir = '../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/prediction_files_for_dc/BRATS_1023.nii.gz'
gt_dir = '../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/labels/BRATS_1023.nii.gz'

pred_data = nib.load(pred_dir).get_fdata()
gt_data = nib.load(gt_dir).get_fdata()

print(pred_data.shape)
print(gt_data.shape)

# %% function to compute the dice coefficient
def dc(p, g):
    dice = np.sum(p[g == 1]) * 2.0 / (np.sum(p) + np.sum(g))
    if math.isnan(dice):
        dice = 0.0
    # print('Dice similarity score is {}'.format(dice))
    return dice


# %% checking it for a single sample weather function ois properly working ?
dice_co = []
for c in range(1, 4):
    temp_pred = np.zeros_like(pred_data)
    temp_gt = np.zeros_like(gt_data)

    temp_pred[pred_data == c] = 1
    temp_gt[gt_data == c] = 1
    dice = dc(temp_pred, temp_gt)
    dice_co.append(dice)

np.mean(dice_co)

#%%
# temp_pred = np.zeros_like(pred_data)
# temp_gt = np.zeros_like(gt_data)
# c=3
# temp_pred[pred_data == c] = 1
# temp_gt[gt_data == c] = 1
# dice = dc(temp_pred, temp_gt)
# dice_co.append(dice)
# print(type(dice))

# %% calculating dice for predictions
pred_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/prediction_files_for_dc")
grd_t_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/labels")

dice_all = []
for i in grd_t_dir.glob("*.nii.gz"):
    gt_name = i.name
    # print(gt_name)
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

# %%

# i = 1
# for i in range(5):
#     s = f"variable i is {i}"
#     print(s)



#%% dice co-efficient for 3d full res
# %% loading data
pred_dir = '../nnUNet_raw_data_base/nnUNet_raw_data/Task505_BrainTumour/test_results_3d/BRATS_341.nii.gz'
gt_dir = '../nnUNet_raw_data_base/nnUNet_raw_data/Task505_BrainTumour/labelsTs/BRATS_341.nii.gz'

pred_data = nib.load(pred_dir).get_fdata()
gt_data = nib.load(gt_dir).get_fdata()

print(pred_data.shape)
print(gt_data.shape)

# %% calculating dice for predictions 3d full res
pred_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task505_BrainTumour/test_results_3d")
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

# %% calculating dice for predictions 2d  multi gpu
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




