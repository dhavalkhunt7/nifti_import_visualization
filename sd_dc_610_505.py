# %%
import math
from pathlib import Path

import nibabel as nib
import numpy as np
# from sklearn.metrics import matthews_corrcoef, roc_auc_score
import pandas as pd

# %%
database = Path("../../Documents/exp_results/")

task_name = "Task610"

gt_dir = database / task_name / "labelsTs"
pred_dir = database / task_name / "resultTs"
pred_3d_dir = database / task_name / "resultTs_3d"


# %% function to compute the dice coefficient
def dc(p, g):
    dice = np.sum(p[g == 1]) * 2.0 / (np.sum(p) + np.sum(g))
    if math.isnan(dice):
        dice = 0.0
    # print('Dice similarity score is {}'.format(dice))
    return dice


# %% for 2d
log_dict = []
log_dict_3d = []
for i in gt_dir.glob("*.nii.gz"):
    # new_name = i.name.split(".nii.gz")[0]
    # print(new_name)
    # log_dict[i.name] = {}
    gt_data = nib.load(i).get_fdata()
    gt_array = gt_data.flatten()

    for k in pred_dir.glob(i.name):

        if i.name == k.name:

            pred_data = nib.load(k).get_fdata()
            pred_array = pred_data.flatten()

            # calculate dice score and add it in dict
            temp_pred = np.zeros_like(pred_data)
            temp_gt = np.zeros_like(gt_data)

            temp_pred[pred_data == 1] = 1
            temp_gt[gt_data == 1] = 1
            dice = dc(temp_pred, temp_gt)
            # log_dict[i.name]["dice_score"] = dice
            log_dict.append(dice)
            print(i.name + " " + "  dice 2d : " + " " + str(dice))

    for j in pred_3d_dir.glob(i.name):

        if i.name == j.name:
            pred_data = nib.load(j).get_fdata()
            pred_array = pred_data.flatten()

            # calculate dise score and add it in dict
            temp_pred = np.zeros_like(pred_data)
            temp_gt = np.zeros_like(gt_data)

            temp_pred[pred_data == 1] = 1
            temp_gt[gt_data == 1] = 1
            dice = dc(temp_pred, temp_gt)
            # log_dict[i.name]["dice_score_3d"] = dice
            log_dict_3d.append(dice)
            print("3d dice : ", dice)

# %% find standard deviation from the list
print("2d dice score : ", np.mean(log_dict))

# find the boundary of the std
print("2d dice score std : ", np.std(log_dict))

print("2d dice std : ", np.std(log_dict))

# %% find standard deviation from the list
print("3d dice score : ", np.mean(log_dict_3d))
print("3d dice score std : ", np.std(log_dict_3d))

#%% task 505
task_name = "Task505"

gt_dir = database / task_name / "labelsTs"
pred_dir = database / task_name / "test_results_2d"
pred_3d_dir = database / task_name / "test_results_3d"


#%%
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
log_dict = []
log_dict_3d = []
for i in gt_dir.glob("*.nii.gz"):
    # new_name = i.name.split(".nii.gz")[0]
    # print(new_name)
    # log_dict[i.name] = {}
    gt_data = nib.load(i).get_fdata()
    gt_array = gt_data.flatten()

    for k in pred_dir.glob(i.name):

        if i.name == k.name:
            pred_data = nib.load(k).get_fdata()
            pred_array = pred_data.flatten()

            dice_co = []

            for c in range(1, 4):
                temp_pred = np.zeros_like(pred_data)
                temp_gt = np.zeros_like(gt_data)

                temp_pred[pred_data == c] = 1
                temp_gt[gt_data == c] = 1
                dice = dc(temp_pred, temp_gt)
                dice_co.append(dice)
            log_dict.append(np.mean(dice_co))
            coeff = np.mean(dice_co)
            print("for" + i.name + " dc : " + str(coeff))

    for j in pred_3d_dir.glob(i.name):

        if i.name == j.name:
            pred_data = nib.load(j).get_fdata()
            pred_array = pred_data.flatten()

            dice_co = []

            for c in range(1, 4):
                temp_pred = np.zeros_like(pred_data)
                temp_gt = np.zeros_like(gt_data)

                temp_pred[pred_data == c] = 1
                temp_gt[gt_data == c] = 1
                dice = dc(temp_pred, temp_gt)
                dice_co.append(dice)
            log_dict_3d.append(np.mean(dice_co))
            coeff = np.mean(dice_co)
            print("for" + i.name + " dc : " + str(coeff))


#%%
print("2d dice score : ", np.mean(log_dict))
print("2d dice score std : ", np.std(log_dict))

print("3d dice score : ", np.mean(log_dict_3d))
print("3d dice score std : ", np.std(log_dict_3d))
