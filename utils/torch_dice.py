# %%
import torch
from torchmetrics import Dice
from torchmetrics.functional import stat_scores
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math

# %%
preds = torch.tensor([2, 0, 2, 1])
target = torch.tensor([1, 1, 2, 0])

# %%
dice = Dice(average='macro', num_classes=3)
dice(preds, target)

# %% micro dice score
dice = Dice(average='micro')
dice(preds, target)

# %%
preds = torch.tensor([0, 1, 0, 1])
target = torch.tensor([1, 1, 0, 1])

# %%
dice = Dice(average='macro', num_classes=2)
dice(preds, target)

# %% micro dice score
dice = Dice(average='micro')
dice(preds, target)

# %%
stat_scores(preds, target, reduce='macro', num_classes=2)
# %%
stat_scores(preds, target, reduce='macro', num_classes=1, multiclass=False)

# %%
gt_data = nib.load("chamba_tasks/data/gt_Human07.nii.gz").get_fdata()
pred_data = nib.load("chamba_tasks/data/pred_Human07.nii.gz").get_fdata()

# %%
gt_data = gt_data.flatten()
pred_data = pred_data.flatten()

# %%
print(np.unique(gt_data))
print(np.unique(pred_data))

# %% float to int
gt_data = gt_data.astype(int)
pred_data = pred_data.astype(int)

# %% calculate dice
dice = Dice(average='macro', num_classes=2)
dice(torch.from_numpy(pred_data), torch.from_numpy(gt_data))

# %% micro dice score
dice = Dice(average='micro')
dice(torch.from_numpy(pred_data), torch.from_numpy(gt_data))

#%% weighted dice using torch.dice
dice = Dice(average='weighted', num_classes=2)
dice(torch.from_numpy(pred_data), torch.from_numpy(gt_data))

# %% calculate dice using intersection over union
def dc(p, g):
    dice = np.sum(p[g == 1]) * 2.0 / (np.sum(p) + np.sum(g))
    return dice


dc(pred_data, gt_data)




#%%
for i in Path("results").glob("*"):
    print(i.name)


#%% load csv file
dice_df = pd.read_csv("results/601_human.csv", index_col=0)

#%% if nan, replace with 0
if dice_df.isnull().values.any():
    dice_df = dice_df.fillna(0)

#%%create 4*3 figure with 12 subplots of boxplots
fig, axs = plt.subplots(4, 3, figsize=(15, 15))
fig.suptitle('Boxplots of Dice Scores for 601 Human Data', fontsize=16)
axs[0, 0].boxplot(dice_df["mcc"])
axs[0, 0].set_title("mcc")
#write median value on the boxplot
axs[0, 0].text(1, 0.5, f"{dice_df['mcc'].median():.3f}", fontsize=12)
axs[0, 1].boxplot(dice_df["accuracy"])
axs[0, 1].set_title("accuracy")
axs[0, 1].text(1, 0.5, f"{dice_df['accuracy'].median():.3f}", fontsize=12)
axs[0, 2].boxplot(dice_df["precision"])
axs[0, 2].set_title("precision")
axs[0, 2].text(1, 0.5, f"{dice_df['precision'].median():.3f}", fontsize=12)
axs[1, 0].boxplot(dice_df["sensitivity"])
axs[1, 0].set_title("sensitivity")
axs[1, 0].text(1, 0.5, f"{dice_df['sensitivity'].median():.3f}", fontsize=12)
axs[1, 1].boxplot(dice_df["specificity"])
axs[1, 1].set_title("specificity")
axs[1, 1].text(1, 0.5, f"{dice_df['specificity'].median():.3f}", fontsize=12)
axs[1, 2].boxplot(dice_df["false_Discovery_Rate"])
axs[1, 2].set_title("false_discovery_rate")
axs[1, 2].text(1, 0.5, f"{dice_df['false_Discovery_Rate'].median():.3f}", fontsize=12)
axs[2, 0].boxplot(dice_df["false_Positive_Rate"])
axs[2, 0].set_title("false_positive_rate")
axs[2, 0].text(1, 0.5, f"{dice_df['false_Positive_Rate'].median():.3f}", fontsize=12)
axs[2, 1].boxplot(dice_df["positive_predictive_value"])
axs[2, 1].set_title("positive_predictive_value")
axs[2, 1].text(1, 0.5, f"{dice_df['positive_predictive_value'].median():.3f}", fontsize=12)
axs[2, 2].boxplot(dice_df["negative_predictive_value"])
axs[2, 2].set_title("negative_predictive_value")
axs[2, 2].text(1, 0.5, f"{dice_df['negative_predictive_value'].median():.3f}", fontsize=12)
axs[3, 0].boxplot(dice_df["dice"])
axs[3, 0].set_title("dice")
axs[3, 0].text(1, 0.5, f"{dice_df['dice'].median():.3f}", fontsize=12)
axs[3, 1].boxplot(dice_df["wspec"])
axs[3, 1].set_title("wspec")
axs[3, 1].text(1, 0.5, f"{dice_df['wspec'].median():.3f}", fontsize=12)
plt.xlim(0, 1)
plt.show()


#%% import csv file
df_605 = pd.read_csv("results/605_rat.csv", index_col=0)

#%% if nan, replace with 0
if df_605.isnull().values.any():
    df_605 = df_605.fillna(0)


#%% create 3*3 figure with 9 subplots of boxplots given below
# dice, accuracy, mcc,
# sensitivity, positive_predictive_value, false_Discovery_Rate
# negative_predictive_value, specificity, false_Positive_Rate

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs[0, 0].boxplot(dice_df["dice"])
axs[0, 0].ylabel("dice")
#put median value on the boxplot in right corner
axs[0, 0].text(1, 0.5, f"{dice_df['dice'].median():.3f}", fontsize=12)
axs[0, 1].boxplot(dice_df["accuracy"])
axs[0, 1].ylabel("accuracy")
axs[0, 1].text(1, 0.5, f"{dice_df['accuracy'].median():.3f}", fontsize=12)
axs[0, 2].boxplot(dice_df["mcc"])
axs[0, 2].ylabel("mcc")
axs[0, 2].text(1, 0.5, f"{dice_df['mcc'].median():.3f}", fontsize=12)
axs[1, 0].boxplot(dice_df["sensitivity"])
axs[1, 0].ylabel("sensitivity")
axs[1, 0].text(1, 0.5, f"{dice_df['sensitivity'].median():.3f}", fontsize=12)
axs[1, 1].boxplot(dice_df["positive_predictive_value"])
axs[1, 1].ylabel("positive_predictive_value")
axs[1, 1].text(1, 0.5, f"{dice_df['positive_predictive_value'].median():.3f}", fontsize=12)
axs[1, 2].boxplot(dice_df["false_Discovery_Rate"])
axs[1, 2].ylabel("false_Discovery_Rate")
axs[1, 2].text(1, 0.5, f"{dice_df['false_Discovery_Rate'].median():.3f}", fontsize=12)
axs[2, 0].boxplot(dice_df["negative_predictive_value"])
axs[2, 0].ylabel("negative_predictive_value")
axs[2, 0].text(1, 0.5, f"{dice_df['negative_predictive_value'].median():.3f}", fontsize=12)
axs[2, 1].boxplot(dice_df["specificity"])
axs[2, 1].ylabel("specificity")
axs[2, 1].text(1, 0.5, f"{dice_df['specificity'].median():.3f}", fontsize=12)
axs[2, 2].boxplot(dice_df["false_Positive_Rate"])
axs[2, 2].ylabel("false_Positive_Rate")
axs[2, 2].text(1, 0.5, f"{dice_df['false_Positive_Rate'].median():.3f}", fontsize=12)
plt.xlim(0, 1, 0.2)
plt.show()

#%% create 3*3 figure with 9 subplots of boxplots given below with ylalbel as column name
# dice, accuracy, mcc,
# sensitivity, positive_predictive_value, false_Discovery_Rate
# negative_predictive_value, specificity, false_Positive_Rate

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs[0, 0].boxplot(dice_df["dice"])
axs[0, 0].set_ylabel("dice")
#put median value on the boxplot in right corner
axs[0, 0].text(1, 0.5, f"{dice_df['dice'].median():.3f}", fontsize=12)
axs[0, 1].boxplot(dice_df["accuracy"])
axs[0, 1].set_ylabel("accuracy")
axs[0, 1].text(1, 0.5, f"{dice_df['accuracy'].median():.3f}", fontsize=12)
axs[0, 2].boxplot(dice_df["mcc"])
axs[0, 2].set_ylabel("mcc")
axs[0, 2].text(1, 0.5, f"{dice_df['mcc'].median():.3f}", fontsize=12)
axs[1, 0].boxplot(dice_df["sensitivity"])
axs[1, 0].set_ylabel("sensitivity")
axs[1, 0].text(1, 0.5, f"{dice_df['sensitivity'].median():.3f}", fontsize=12)
axs[1, 1].boxplot(dice_df["positive_predictive_value"])
axs[1, 1].set_ylabel("positive_predictive_value")
axs[1, 1].text(1, 0.5, f"{dice_df['positive_predictive_value'].median():.3f}", fontsize=12)
axs[1, 2].boxplot(dice_df["false_Discovery_Rate"])
axs[1, 2].set_ylabel("false_Discovery_Rate")
axs[1, 2].text(1, 0.5, f"{dice_df['false_Discovery_Rate'].median():.3f}", fontsize=12)
axs[2, 0].boxplot(dice_df["negative_predictive_value"])
axs[2, 0].set_ylabel("negative_predictive_value")
axs[2, 1].boxplot(dice_df["specificity"])
axs[2, 1].set_ylabel("specificity")
axs[2, 2].boxplot(dice_df["false_Positive_Rate"])
axs[2, 2].set_ylabel("false_Positive_Rate")
plt.xlim(0, 1, 0.2)
plt.show()


