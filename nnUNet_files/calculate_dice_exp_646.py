# %% managing imports
import numpy as np
import nibabel as nb
from pathlib import Path
import pandas as pd


# %% calculate dice
def calculate_dice(pred, gt):
    # if pred and gt has nan values then replace them with 0
    if np.isnan(pred).any():
        pred[np.isnan(pred)] = 0
    if np.isnan(gt).any():
        gt[np.isnan(gt)] = 0

    pred = pred.flatten()
    gt = gt.flatten()

    # pred and gt have one unique value the print dice = 1
    if len(np.unique(pred)) == 1 and len(np.unique(gt)) == 1:
        return 0
    else:
        intersection = np.sum(pred * gt)
        return 2 * intersection / (np.sum(pred) + np.sum(gt))


# %% importing test data and labels
database = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task646_combined_patch")
pred_2d_dir = database / "result_2d_reconstructed"
pred_3d_dir = database / "result_3d_reconstructed"
gt_dir = database / "labels_reconstructed"

# %% calculate dice for all the images and store in a list
list_dice_2d = []
list_dice_3d = []
for i in gt_dir.glob("*.nii.gz"):
    gt_file = i
    pred_2d_file = pred_2d_dir / i.name
    pred_3d_file = pred_3d_dir / i.name
    if pred_2d_file.exists() and pred_3d_file.exists():
        gt = np.array(nb.load(gt_file).dataobj)
        pred_2d = np.array(nb.load(pred_2d_file).dataobj)
        pred_3d = np.array(nb.load(pred_3d_file).dataobj)
        dice_2d = calculate_dice(pred_2d, gt)
        dice_3d = calculate_dice(pred_3d, gt)
        if type(dice_2d) == str:
            print(f"{i.name} has only one unique value")
        elif type(dice_3d) == str:
            print(f"{i.name} has only one unique value")
        else:
            list_dice_2d.append(dice_2d)
            print(type(dice_2d))
            list_dice_3d.append(dice_3d)

#%%'

list_dice_3d
# print(np.std(list_dice_3d))

# %% calculate mean and std
print(f"mean dice 2d: {np.mean(list_dice_2d)}")
print(f"std dice 2d: {np.std(list_dice_2d)}")
print(f"mean dice 3d: {np.mean(list_dice_3d)}")
print(f"std dice 3d: {np.std(list_dice_3d)}")




# %% contruct box plot for list_dice
import matplotlib.pyplot as plt

plt.boxplot(list_dice)
plt.show()

# %% calculate the mean dice for each subject for human and rat separately using the name of the file starts with
# Human or Rat create a dictionary to store the dice for each subject
human_dice = []
rat_dice = []
for i in pred_dir.glob("*.nii.gz"):
    pred_data = nb.load(i).get_fdata()

    # find file name of the corresponding gt
    gt_file = gt_dir / i.name

    # if gt file exists then calculate dice else print file not found
    if gt_file.exists():
        gt_data = nb.load(gt_file).get_fdata()
        dice = calculate_dice(pred_data, gt_data)
        if i.name.startswith("Human"):
            human_dice.append(dice)
        elif i.name.startswith("Rat"):
            rat_dice.append(dice)
        # list_dice.append(dice)
        print(i.name, dice)
    else:
        print("file not found")

# %%
# calculate mean dice and standard deviation for human and rat and print them
mean_human_dice = np.mean(human_dice)
std_human_dice = np.std(human_dice)
print("mean dice for human is: ", mean_human_dice)
print("standard deviation for human is: ", std_human_dice)
# for rat_dice
mean_rat_dice = np.mean(rat_dice)
std_rat_dice = np.std(rat_dice)
print("mean dice for rat is: ", mean_rat_dice)
print("standard deviation for rat is: ", std_rat_dice)

#%%print leng of human and rat dice
print(len(human_dice))
print(len(rat_dice))

