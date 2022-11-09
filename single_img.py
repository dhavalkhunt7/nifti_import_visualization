# %% managing imports
from pathlib import Path
import nibabel as nib
import numpy as np
import os

import patchify as patchify

# %%
data = nib.load("abcd/ADC.nii")
data = np.array(data.dataobj)
print(data.shape)


# %%
# check if there is any nan value in the image
def check_nan(input_file):
    if np.isnan(input_file).any():
        input_file[np.isnan(input_file)] = 0
    return input_file


data = check_nan(data)
# %%
# change shape from 77*51*24 to 77*51*30 by adding zeros
data_1 = np.pad(data, ((0, 0), (0, 0), (0, 6)), 'constant', constant_values=0)
print(data.shape)

# %% check if array is same or not
np.equal(data, data_1[:, :, 0:24]).all()


# %% write a function to create patches and append them to a list
def create_patches(img):
    # create patches of 45*45*45 with stride 15
    patches_1 = patchify.patchify(img, (30, 30, 30), step=15)
    final_patch = patches_1.reshape(-1, 30, 30, 30)

    # extract the patches in 3d shape and save it in a list
    list_patches = []
    for z in range(final_patch.shape[0]):
        list_patches.append(final_patch[z, :, :, :])
    return list_patches


# %%
patches = create_patches(data)
print(len(patches))
