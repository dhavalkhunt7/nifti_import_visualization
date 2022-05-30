from pathlib import Path

import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns


# %%
import numpy as np

img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/images_tr_converted/BRATS_1020_0002.nii.gz")
img1 = nib.load(img_dir)

#%%

t1_data = img1.get_fdata()
t1_data.shape

#%%
masked_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/prediction_files_for_dc"
                   "/BRATS_1020.nii.gz")
masked_img  = nib.load(masked_path)

masked_img.get_fdata().shape

#%%
plt.figure(figsize=(8, 8))
# plt.imshow(masked_img.get_fdata()[:, :, 78], 'jet', interpolation='none')
plt.imshow(img1.get_fdata()[:, :, 78], interpolation='none')
plt.show()






#%% another way to

path = "../nnUNet_raw_data_base/nnUNet_raw_data/Task501_BrainTumour/imagesTs/BRATS_500.nii.gz"
mask_path = "../nnUNet_raw_data_base/nnUNet_raw_data/Result_502/BRATS_500.nii.gz"

pre_img = nib.load(path)
masked_img = nib.load(mask_path)

#%%
print(type(pre_img))
pre_img_hdr = pre_img.header
print(pre_img_hdr)
pre_img_hdr.keys()

#%%
print(type(masked_img))
masked_hdr = masked_img.header
print(masked_hdr)
masked_hdr.keys()

#%%
print(pre_img.shape)
print(pre_img.header.get_zooms())
print(pre_img.header.get_xyzt_units())

#%%
print(masked_img.shape)
print(masked_img.header.get_zooms())
print(masked_img.header.get_xyzt_units())

#%%
img_data = pre_img.get_fdata()
print(img_data.shape)
type(img_data)

print(np.min(img_data))
print(np.max(img_data))

print(np.argmax(img_data, axis=2))

#%%
masked_data = masked_img.get_fdata()
print(masked_data.shape)
type(masked_data)

print(np.min(masked_data))
print(np.max(masked_data))

#%%
print(img_data[118:121, 118:121, 75:78,0])

#%%
print(masked_data[134, 126, 49])
print(masked_data[134,:,:])






#%% new way

from nilearn import plotting,image

path = "../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/images_tr/BRATS_020.nii.gz"
mask_path = "../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/prediction_files_for_dc/BRATS_1020.nii.gz"

first_volume = image.index_img(path,1)


plotting.plot_roi(mask_path, bg_img=first_volume, cmap='Paired', colorbar=True)
plt.show()


#%%
print(mask_path)

masked_img = image.load_img(mask_path)
masked_data = image.get_data(masked_img)
h_masked_data = masked_img.header
print(h_masked_data)

plotting.plot_img(mask_path)
plt.show()

#%%
plotting.plot_roi(masked_img, first_volume)
plt.show()
