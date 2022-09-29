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


#%%
img = nib.load(database_input)
print(img.shape)


#%%


#%%

base_folder  = 'input/a.jpeg'

normal_image = skimage.io.imread(base_folder)
print(normal_image.shape)
print(normal_image.dtype)
print(type(normal_image))

#%% plt
normal_image_final = normal_image[:, :, 1]
plt.imshow(normal_image_final)
plt.show()
