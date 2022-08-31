import numpy as np
from  pathlib import Path
import nibabel as nb

#%% craete a function to calculate dice score


def dc(pred_data, gt_data):
    """
    calculate dice score
    """
    temp_pred = np.zeros_like(pred_data)
    temp_gt = np.zeros_like(gt_data)

    temp_pred[pred_data == 1] = 1
    temp_gt[gt_data == 1] = 1
    dice = 2 * np.sum(temp_pred * temp_gt) / (np.sum(temp_pred) + np.sum(temp_gt))
    return dice


#%% import data of segmematation_results_adrian/
database = Path("../nnUNet_raw_data_base/nnUNet_raw_data/segmentation_results_adrian")

gt_dir = database / "Task620" / "gt_files"
pred_dir = database / "Task620" / "3d_fullres"
list_dc = []

for i in pred_dir.glob("*.nii"):

    for j in gt_dir.glob("*.nii.gz"):
         if j.name.replace(".nii.gz", ".nii") == i.name:
             gt_data = nb.load(j).get_fdata()
             gt_array = gt_data.flatten()

             pred_data = nb.load(i).get_fdata()
             pred_array = pred_data.flatten()

             dice = dc(pred_array, gt_array)
             list_dc.append(dice)
             print(dice)


#%%
np.mean(list_dc)

#%%
np.std(list_dc)

#%% find the no of data in the list
len(list_dc)




