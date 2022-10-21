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
        return 1
    else:
        intersection = np.sum(pred * gt)
        return 2 * intersection / (np.sum(pred) + np.sum(gt))


# %% importing test data and labels
database = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task646_combined_patch")
pred_dir = database / "result_2d_reconstructed"
gt_dir = database / "labels_reconstructed"

# %% calculate dice for all the images and store in a list
list_dice = []
for i in pred_dir.glob("*.nii.gz"):
    pred_data = nb.load(i).get_fdata()

    # find file name of the corresponding gt
    gt_file = gt_dir / i.name

    # if gt file exists then calculate dice else print file not found
    if gt_file.exists():
        gt_data = nb.load(gt_file).get_fdata()
        dice = calculate_dice(pred_data, gt_data)
        list_dice.append(dice)
        print(i.name, dice)
    else:
        print("file not found")

# %%
# calculate mean dice
mean_dice = np.mean(list_dice)
print("mean dice is: ", mean_dice)
print(len(list_dice))
# standard deviation of dice print
std_dice = np.std(list_dice)
print("standard deviation of dice is: ", std_dice)
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

