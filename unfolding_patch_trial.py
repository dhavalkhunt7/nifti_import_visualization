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


#%% plot the patches_15 using subplot
fig, ax = plt.subplots(5, 6, figsize=(10, 10))
for i in range(5):
    for j in range(6):
        ax[i, j].imshow(patches_15[1, i, j, 0, :, :], cmap='gray')
        ax[i, j].axis('off')
plt.show()


#%%
patches_2 = patchify(data, (31, 31, 31), step=2)
print(patches_2.shape)


#%%
a = plt.imread('input/abc.jpeg')
print(a.shape)
print(a.dtype)
print(type(a))

a_final = a[:, :, 0]
print(a_final.shape)

#%%
plt.imshow(a_final)
plt.show()

#%%
patches_1 = patchify(a_final, (100, 100), step=100)
print(patches_1.shape)

#%%
patches_2 = patchify(a_final, (100, 100), step=50)
print(patches_2.shape)

#%%

#%% plot the patches  USING SUBPLOT
fig, ax = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        ax[i, j].imshow(patches_1[i, j, :, :], cmap='gray')
        ax[i, j].axis('off')
plt.show()

#%% plot the patches  USING SUBPLOT
fig, ax = plt.subplots(7, 7)
for i in range(7):
    for j in range(7):
        ax[i, j].imshow(patches_2[i, j, :, :], cmap='gray')
        ax[i, j].axis('off')
plt.show()


#%%
pt_1 = patchify(data, (31, 31, 31), step=31)
print(pt_1.shape)

#%% extract all the patches from the pt_1
pt_1_list = []
for i in range(pt_1.shape[0]):
    for j in range(pt_1.shape[1]):
        for k in range(pt_1.shape[2]):
            pt_1_list.append(pt_1[i, j, k, :, :, :])
print(len(pt_1_list))


#%% save the patches from the pt_1_list as nii file
for i in range(len(pt_1_list)):
    img = nib.Nifti1Image(pt_1_list[i], np.eye(4))
    nib.save(img, 'output/list_patches/patch_{}.nii'.format(i))

#%%
