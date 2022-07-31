# %% calculating dice for predictions 2d  multi gpu
import math
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import nibabel as nib

#%%
# %% function to compute the dice coefficient
def dc(p, g):
    dice = np.sum(p[g == 1]) * 2.0 / (np.sum(p) + np.sum(g))
    if math.isnan(dice):
        dice = 0.0
    # print('Dice similarity score is {}'.format(dice))
    return dice


#%%
img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task600_rat")
test_result_dir = img_dir / "resultTs"
gt_dir = img_dir / "all_other_images_connected_ts"

# %% extracting dir
dice_all = []
for i in test_result_dir.glob("*.nii.gz"):
    result_files = i.name.split(".nii.gz")

    for j in gt_dir.glob("*"):
        gt_files = j.name.split("_GroundTruth24h.nii.gz")

        if gt_files[0] == result_files[0]:
            pred_data = nib.load(i).get_fdata()
            gt_data = nib.load(j).get_fdata()

            # print(np.unique(gt_data))
            temp_pred = np.zeros_like(pred_data)
            temp_gt = np.zeros_like(gt_data)

            temp_pred[pred_data == 1] = 1
            temp_gt[gt_data == 1] = 1
            dice = dc(temp_pred, temp_gt)
            dice_all.append(dice)

            print("for " + result_files[0] + "dc : " + str(dice))

# %% check it for human testing predictions
human_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task600_rat/human_data_testing_set/")
human_gt_dir = human_dir / "all_others_ts"
human_test_dir = human_dir / "result_ts"

#%%
dice_all =[]
for i in human_test_dir.glob("*.nii.gz"):
    # print(i.name)
    test_file = i.name.split(".nii.gz")[0]
    # print(test_file)

    for j in human_gt_dir.glob("*.nii.gz"):
        # print(j.name)
        gt_file = j.name.split("_GroundTrouth.nii.gz")[0]
        # print(gt_file)

        if test_file == gt_file:
            # print(test_file)

            pred_data = nib.load(i).get_fdata()
            gt_data = nib.load(j).get_fdata()

            # print(np.unique(pred_data))
            temp_pred = np.zeros_like(pred_data)
            temp_gt = np.zeros_like(gt_data)
            # print(np.unique(temp_gt))
            #
            temp_pred[pred_data == 1] = 1
            temp_gt[gt_data == 1] = 1
            dice = dc(temp_pred, temp_gt)
            dice_all.append(dice)

            if dice > 0.7:
                print("for " + test_file + "dc : " + str(dice))

#%%
for i in range(len(dice_all)):
    if dice_all[i] >0.7:
        print(dice_all[i])

#%%
len(dice_all)