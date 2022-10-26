# %% managing imports

import numpy as np
import nibabel as nb
from pathlib import Path
import pandas as pd
import torch
from IPython.core import macro
from torchmetrics.functional import dice
import matplotlib.pyplot as plt


# %% create a function for whole process
def calculate_dice(gt, pred):
    # load nii_file
    gt_file = nb.load(gt)
    pred_file = nb.load(pred)
    # load data
    gt_data = np.array(gt_file.dataobj)
    pred_data = np.array(pred_file.dataobj)
    # change nan values to 0
    if np.isnan(gt_data).any():
        gt_data[np.isnan(gt_data)] = 0
    if np.isnan(pred_data).any():
        pred_data[np.isnan(pred_data)] = 0
    # flatten the arrays
    gt_flat = gt_data.flatten()
    pred_flat = pred_data.flatten()
    # change dtype to int
    gt_flat = gt_flat.astype(int)
    pred_flat = pred_flat.astype(int)
    # nparray to torch.tensor
    gt_tensor = torch.from_numpy(gt_flat)
    pred_tensor = torch.from_numpy(pred_flat)
    # calculate dice
    macro_dc = dice(pred_tensor, gt_tensor, average="macro", num_classes=2)
    micro_dc = dice(pred_tensor, gt_tensor, average="micro")
    return macro_dc, micro_dc


# %%
database = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task646_combined_patch")
gt_dir = database / "labels_reconstructed"
pred_2d_dir = database / "result_2d_reconstructed"
pred_3d_dir = database / "result_3d_reconstructed"

# %%
# dict to store dice values
macro_human_dice = {'2d': {}, '3d': {}}
micro_human_dice = {'2d': {}, '3d': {}}
macro_rat_dice = {'2d': {}, '3d': {}}
micro_rat_dice = {'2d': {}, '3d': {}}

for i in gt_dir.glob("*.nii.gz"):
    gt_file = i
    pred_2d_file = pred_2d_dir / i.name
    pred_3d_file = pred_3d_dir / i.name
    if pred_2d_file.exists() and pred_3d_file.exists():
        macro_dc_2d, micro_dc_2d = calculate_dice(gt_file, pred_2d_file)
        macro_dc_3d, micro_dc_3d = calculate_dice(gt_file, pred_3d_file)
        print(i.name, macro_dc_2d, micro_dc_3d)
        # tensor to float
        macro_dc_2d = macro_dc_2d.item()
        micro_dc_2d = micro_dc_2d.item()
        macro_dc_3d = macro_dc_3d.item()
        micro_dc_3d = micro_dc_3d.item()
        # store dice values
        if 'Human' in i.name:
            macro_human_dice['2d'][i.name] = macro_dc_2d
            micro_human_dice['2d'][i.name] = micro_dc_2d
            macro_human_dice['3d'][i.name] = macro_dc_3d
            micro_human_dice['3d'][i.name] = micro_dc_3d
        elif 'Rat' in i.name:
            macro_rat_dice['2d'][i.name] = macro_dc_2d
            micro_rat_dice['2d'][i.name] = micro_dc_2d
            macro_rat_dice['3d'][i.name] = macro_dc_3d
            micro_rat_dice['3d'][i.name] = micro_dc_3d
    else:
        print("file not found")

# %% duct to df
macro_human_dice_df = pd.DataFrame(macro_human_dice)
micro_human_dice_df = pd.DataFrame(micro_human_dice)
macro_rat_dice_df = pd.DataFrame(macro_rat_dice)
micro_rat_dice_df = pd.DataFrame(micro_rat_dice)

# %% save boxplot as pdf
# macro_human_dice_df.boxplot(column=['2d', '3d'])
# plt.savefig('imagesTr/macro_human_dice.pdf')
micro_rat_dice_df.boxplot(column=['2d', '3d'])
plt.savefig('imagesTr/micro_rat_dice.pdf')
macro_rat_dice_df.boxplot(column=['2d', '3d'])
plt.savefig('imagesTr/macro_rat_dice.pdf')
micro_human_dice_df.boxplot(column=['2d', '3d'])
plt.savefig('imagesTr/micro_human_dice.pdf')

#%%df to list and combined
macro_human_dice_2d = macro_human_dice_df['2d'].values.tolist()
macro_human_dice_3d = macro_human_dice_df['3d'].values.tolist()
macro_rat_dice_2d = macro_rat_dice_df['2d'].values.tolist()
macro_rat_dice_3d = macro_rat_dice_df['3d'].values.tolist()
micro_human_dice_2d = micro_human_dice_df['2d'].values.tolist()
micro_human_dice_3d = micro_human_dice_df['3d'].values.tolist()
micro_rat_dice_2d = micro_rat_dice_df['2d'].values.tolist()
micro_rat_dice_3d = micro_rat_dice_df['3d'].values.tolist()

#%% combined both human and rat as one list

macro_dice_2d = macro_human_dice_2d + macro_rat_dice_2d
macro_dice_3d = macro_human_dice_3d + macro_rat_dice_3d
micro_dice_2d = micro_human_dice_2d + micro_rat_dice_2d
micro_dice_3d = micro_human_dice_3d + micro_rat_dice_3d

#%% save boxplot as pdf
plt.boxplot([macro_dice_2d, macro_dice_3d])
plt.savefig('imagesTr/macro_dice.pdf')
plt.boxplot([micro_dice_2d, micro_dice_3d])
plt.savefig('imagesTr/micro_dice.pdf')

#%%
print("macro 2d mean: ", np.mean(macro_dice_2d))
print("macro 3d mean: ", np.mean(macro_dice_3d))
print("micro 2d mean: ", np.mean(micro_dice_2d))
print("micro 3d mean: ", np.mean(micro_dice_3d))
#standard deviation
print("macro 2d std: ", np.std(macro_dice_2d))
print("macro 3d std: ", np.std(macro_dice_3d))
print("micro 2d std: ", np.std(micro_dice_2d))
print("micro 3d std: ", np.std(micro_dice_3d))


# %%
gt_img = Path("imagesTr/gt_Human07_000.nii.gz")
pred_img = Path("imagesTr/pred_Human07_000.nii.gz")

# %%
gt_data = nb.load(gt_img).get_fdata()
pred_data = nb.load(pred_img).get_fdata()

# print unique values
print(np.unique(gt_data))
print(np.unique(pred_data))

# %%
macro_dc, micro_dc = calculate_dice(gt_img, pred_img)
print(macro_dc, micro_dc)
# print dtype of macro_dc and micro_dc
print(type(macro_dc), type(micro_dc))
# torch sensor to normal float
macro_dc = macro_dc.item()
micro_dc = micro_dc.item()
print(type(macro_dc), type(micro_dc))
# print dice
print(macro_dc, micro_dc)

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

# %%print leng of human and rat dice
print(len(human_dice))
print(len(rat_dice))
