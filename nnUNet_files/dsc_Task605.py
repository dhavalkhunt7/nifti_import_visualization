# %% loading data
from pathlib import Path
import numpy as np
import math
import nibabel as nib

# %%
database_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task645_Patch_2d_Rat24h")
resultTs_dir = database_dir / "results"
labelsTs_dir = database_dir / "labelsTs"

# %%
label_data = nib.load(labelsTs_dir / "Rat140_90.nii.gz").get_fdata()
results_data = nib.load(resultTs_dir / "Rat140_90.nii.gz").get_fdata()

# %%
result = dc(results_data, label_data)

# %%
print(result)


# %%
def calculate_dice(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    if len(np.unique(pred)) == 1 and len(np.unique(gt)) == 1:
        return 1
    else:
        intersection = np.sum(pred * gt)
        return 2 * intersection / (np.sum(pred) + np.sum(gt))


# %% function to compute the dice coefficient
def dc(p, g):
    # if p and q has only one unique value, then return 1
    if len(np.unique(p)) == 1 and len(np.unique(g)) == 1:
        return 1
    else:
        dice = np.sum(p[g == 1]) * 2.0 / (np.sum(p) + np.sum(g))
    # print('Dice similarity score is {}'.format(dice))
    return dice


# %%
dice_all = []
for i in labelsTs_dir.glob("*"):
    # print(i.name)

    for j in resultTs_dir.glob("*.nii.gz"):
        # print(j.name)

        if i.name == j.name:
            pred_data = nib.load(i).get_fdata()
            gt_data = nib.load(j).get_fdata()

            # gt has nan values then replace them with 0
            gt_data[np.isnan(gt_data)] = 0

            # print(np.unique(gt_data))
            temp_pred = np.zeros_like(pred_data)
            temp_gt = np.zeros_like(gt_data)

            temp_pred[pred_data == 1] = 1
            temp_gt[gt_data == 1] = 1
            # dice = dc(temp_pred, temp_gt)
            dice = calculate_dice(temp_pred, temp_gt)
            dice_all.append(dice)

            print("for " + i.name + "  dc : " + str(dice))

        # print(dice_all)
        # np.mean(dice_all)

# %%
print(np.mean(dice_all))



