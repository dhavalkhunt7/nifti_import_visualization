# %%
from pathlib import Path

# import cv2
import nibabel as nib
import numpy as np
import patchify as patchify
import skimage
from matplotlib import pyplot as plt

# %%
print("hellp")

# %% input christine_theranostics_data_folder


database_input = Path("input/Masked_ADC.nii")

# %%
img = nib.load(database_input)
print(img.shape)

# %%

base_folder = 'input/abc.jpeg'

normal_image = skimage.io.imread(base_folder)
print(normal_image.shape)
print(normal_image.dtype)
print(type(normal_image))

# %% plt
normal_image_final = normal_image[:, :, 1]
plt.imshow(normal_image_final)
plt.show()

print(normal_image_final.shape)

# %% write a function to create patches from 3d image with patch size 31x31x31 and overlap 15


data_input = np.array(img.dataobj)
data_h, data_w, data_d = data_input.shape
print(data_h, data_w, data_d)

patch_size = 31
overlap = 15

patch_list = []
for data_h in range(0, data_input.shape[0], patch_size - overlap):
    for data_w in range(0, data_input.shape[1], patch_size - overlap):
        for data_d in range(0, data_input.shape[2], patch_size - overlap):
            patch = data_input[data_h:data_h + patch_size, data_w:data_w + patch_size, data_d:data_d + patch_size]
            if patch.shape == (patch_size, patch_size, patch_size):
                print(patch.shape)
                print(patch.dtype)
                print(type(patch))
                print(patch)

                patch_list.append(patch)

# %%
print(len(patch_list))

# %% code 1
patch_1 = data_input[0:31, 0:31, 0:31]
patch_2 = data_input[0:31, 0:31, 15:46]


# %% create patch function based on  code 1
def create_patches(img, patch_size, overlap):
    patch_list = []
    for data_h in range(0, img.shape[0], patch_size - overlap):
        for data_w in range(0, img.shape[1], patch_size - overlap):
            for data_d in range(0, img.shape[2], patch_size - overlap):
                patch = img[data_h:data_h + patch_size, data_w:data_w + patch_size, data_d:data_d + patch_size]
                if patch.shape == (patch_size, patch_size, patch_size):
                    patch_list.append(patch)
    return patch_list

#%% write same function for 2d image
def create_patches_2d(img, patch_size, overlap):
    patch_list = []
    for data_h in range(0, img.shape[0], patch_size - overlap):
        for data_w in range(0, img.shape[1], patch_size - overlap):
            patch = img[data_h:data_h + patch_size, data_w:data_w + patch_size]
            if patch.shape == (patch_size, patch_size):
                patch_list.append(patch)
    return patch_list


# %%
patch_list_1 = create_patches(data_input, 45, 20)

#%%
print(len(patch_list_1))

#%% export patches to niii.gz
for i in range(len(patch_list_1)):
    patch = patch_list_1[i]
    patch = nib.Nifti1Image(patch, np.eye(4))
    nib.save(patch, 'output/list_patches/patch_{}.nii'.format(i))


#%%
patch_list_2 = create_patches_2d(normal_image_final, 150, 20)
print(len(patch_list_2))

#%% plot patches using subplot
fig, ax = plt.subplots(3,3, figsize=(10, 10))
for i in range(3):
    for j in range(3):
        ax[i, j].imshow(patch_list_2[i*3+j], cmap='gray')
plt.show()

#%% resize data_input to 93x93x93
data_input_resized = skimage.transform.resize(data_input, (93, 93, 93), anti_aliasing=True)
print(data_input_resized.shape)

#%% create patches from resized data_input
patch_list_3 = create_patches(data_input_resized, 31, 0)
print(len(patch_list_3))

#%% plot patches using subplot from resized data_input with using 2d image
fig, ax = plt.subplots(3,3, figsize=(10, 10))