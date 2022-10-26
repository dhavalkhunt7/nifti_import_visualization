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
