# %% managing imports
from pathlib import Path
import nibabel as nib
import numpy as np
import os

import patchify

# %% import data from Documents/data/adrian_data
input_folder = Path("../../../Documents/data/adrian_data/Human_labelled/Human07")
reconstructed_folder = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task646_combined_patch/labels_reconstructed")

# output_folder = "../nnUNet_raw_data_base/nnUNet_raw_data/Task630_Patch_2d_Rat24h/"


# %%
for i in input_folder.glob("GroundTrouth.nii"):
    # load the image
    img = nib.load(i)
    # get the data
    img_data = np.array(img.dataobj)
    # print unique values
    print(np.unique(img_data))

    # load the image from the reconstructed folder
    img_reconstructed = nib.load(reconstructed_folder / "pred_Human07.nii.gz")
    # get the data
    img_reconstructed_data = np.array(img_reconstructed.dataobj)
    # print unique values
    print(np.unique(img_reconstructed_data))

#%% get the shape of the image and reconstructed image
img_shape = img_data.shape
print(img_shape)
img_reconstructed_shape = img_reconstructed_data.shape
print(img_reconstructed_shape)

# %% print unique value from both images
print(np.unique(img_data))
print(np.unique(img_reconstructed_data))

#%% reshape the image
img_data_reshaped = img_data[:120, :135, :120]
print(img_data_reshaped.shape)

#%% change the nan values to 0 in img_data_reshaped
img_data_reshaped[np.isnan(img_data_reshaped)] = 0


#%% count the unique values in total image and reshaped image
print(np.unique(img_data_reshaped))
print(np.unique(img_reconstructed_data))
# print total of unique values in both images
print(len(np.unique(img_data_reshaped)))
print(len(np.unique(img_reconstructed_data)))

#%% count howmany 0 and 1 in the image and reconstructed image
print(np.count_nonzero(img_data_reshaped == 0))
print(np.count_nonzero(img_data_reshaped == 1))
print(np.count_nonzero(img_reconstructed_data == 0))
print(np.count_nonzero(img_reconstructed_data == 1))

#%%
print(np.array_equal(img_reconstructed, img_data_reshaped))


#%%
print(calculate_dice(img_data_reshaped, img_reconstructed_data))

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

# %% trail run


for i in input_folder.glob("*.nii"):
    if "Masked_ADC" in i.name:
        adc_file = i
        # import the adc file using nibabel
        adc_img = nib.load(adc_file)
        adc_data = np.array(adc_img.dataobj)
        # create patches using patchify
        patches_1 = patchify.patchify(adc_data, (45, 45, 45), step=15)

# %%
# print the shape of patches_1
print(patches_1.shape)
