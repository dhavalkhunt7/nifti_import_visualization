#%% import .nii file from input
import pathlib as Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import unfoldNd
from patchify import patchify

#%%
data_folder = 'input/Masked_ADC.nii'

img = nib.load(data_folder)

#%%
data = np.array(img.dataobj)
print(data.shape)
print(data.dtype)
type(data)


#%%
patches_1 = patchify(data, (31, 31, 31), step=1)
print(patches_1.shape)

#%%
patches = patchify(data, (31, 31, 31))
print(patches.shape)
#%%
patches_15 = patchify(data, (31, 31, 31), step=15)
print(patches_15.shape)

#%%
patches_2 = patchify(data, (31, 31, 31), step=2)
print(patches_2.shape)


#%%
a = plt.imread('input/a.jpeg')
print(a.shape)
print(a.dtype)
print(type(a))

a_final = a[:, :, 0]
print(a_final.shape)

#%%
plt.imshow(a_final, cmap='gray')
plt.show()

#%%
patches = patchify(data, (31, 31, 31), step=31)


#%%