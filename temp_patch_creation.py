#%% import .nii file from input
import pathlib as Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

#%%
data_folder = 'input/Masked_ADC.nii'

img = nib.load(data_folder)

#%%
data = np.array(img.dataobj)
print(data.shape)
print(data.dtype)
type(data)
#%%
single_slice = data[:,:,74]
print(single_slice.shape)
# plot the slice
plt.imshow(single_slice, cmap='gray')
plt.show()


#%% perfect method
#%% create patches from single slice of same size 31x31 with overlap of 15 using for loop
patch_size = 31
overlap = 15
patch_list = []
for i in range(0, single_slice.shape[0], patch_size-overlap):
    for j in range(0, single_slice.shape[1], patch_size-overlap):
        patch = single_slice[i:i+patch_size, j:j+patch_size]
        if patch.shape == (patch_size, patch_size):
            patch_list.append(patch)
        # patch_list.append(patch)
print(len(patch_list))

#%% 3d to 2d slices and patches done....
#%% repeat the same for all slices .....
patch_list_whole_img = []
for k in range(data.shape[2]):
    single_slice = data[:,:,k]
    for i in range(0, single_slice.shape[0], patch_size-overlap):
        for j in range(0, single_slice.shape[1], patch_size-overlap):
            patch = single_slice[i:i+patch_size, j:j+patch_size]
            if patch.shape == (patch_size, patch_size):
                patch_list_whole_img.append(patch)
            # patch_list.append(patch)
print(len(patch_list_whole_img))


#%% create a function for the same
def create_patches(img, patch_size, overlap):
    patch_list_whole_img = []
    for k in range(img.shape[2]):
        single_slice = img[:,:,k]
        for i in range(0, single_slice.shape[0], patch_size-overlap):
            for j in range(0, single_slice.shape[1], patch_size-overlap):
                patch = single_slice[i:i+patch_size, j:j+patch_size]
                if patch.shape == (patch_size, patch_size):
                    patch_list_whole_img.append(patch)
                # patch_list.append(patch)
    return patch_list_whole_img

#%% save the patches in a folder as .nii.gz
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def save_patches(patch_list_whole_img, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(len(patch_list_whole_img)):
        patch = patch_list_whole_img[i]
        img = nib.Nifti1Image(patch, np.eye(4))
        nib.save(img, output_folder + '/patch_' + str(i) + '.nii.gz')
        print('patch_' + str(i) + '.nii.gz saved')

#%%
save_patches(patch_list_whole_img, 'output/patches')

#%%
print(len(patch_list_whole_img))
#%%check the shape of the patches if they are 31x31
for i in range(len(patch_list_whole_img)):
    if patch_list_whole_img[i].shape != (patch_size, patch_size):
        print('error')
    else:
        print('perfect')

#%% print shape of all the patches
for i in range(len(patch_list)):
    print(patch_list[i].shape)
#%% plot all the patches usinf subplot
fig, ax = plt.subplots(4, 4, figsize=(10, 10))
for i in range(4):
    for j in range(4):
        ax[i, j].imshow(patch_list[i*4+j], cmap='gray')
        ax[i, j].axis('off')
plt.show()

#%%
single_slice.shape[0]


#%% import masked_T2.nii file from input

t2_folder = 'input/Masked_T2.nii'

t2_img = nib.load(t2_folder)
t2_data = np.array(t2_img.dataobj)
print(t2_data.shape)
