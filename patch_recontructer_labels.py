# %% managing all the imports here only
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import patchify as patchify
import matplotlib.pyplot as plt

# %% get data from imagesTr
dataset_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task645_Patch_2d_Rat24h/results")
output_path = "../nnUNet_raw_data_base/nnUNet_raw_data/Task645_Patch_2d_Rat24h/rat_reconstructed"


# %% create a function to process all the data using dictionary
def process_all_data(file):
    main_name = i.name.split("_")[0]
    # if main name doesnot exist in dictionary then create a new key
    if main_name not in data_dict:
        data_dict[main_name] = {}
        print("dict created: %s" % main_name)

    get_patch = i.name.split("_")[1].split(".")[0]
    # if get_patch_no has one character then add 0 as prefix
    if len(get_patch) == 1:
        get_patch = "0" + get_patch
    # add the patch number to the dictionary
    data_dict[main_name][get_patch] = i
    # print patch added in main_name dictionary
    print("patch added in %s dictionary: %s" % (main_name, get_patch))


# %%
# save the data in a dictionary
data_dict = {}
for i in dataset_path.glob("*.nii.gz"):
    process_all_data(i)

# %% dict to dataframe
df = pd.DataFrame(data_dict)


# %% create a function for upper code
def get_data_from_nifti(file_list):
    new_list = []
    for element in file_list:
        e_img = nib.load(element)
        e_data = np.array(e_img.dataobj)
        new_list.append(e_data)
    return new_list


# %% create a function for whole process
def reconstruct_patches(patches_list,):
    patches = np.array(patches_list)
    patches_reshaped = patches.reshape(4, 4, 6, 45, 45, 45)
    reconstructed_data = patchify.unpatchify(patches_reshaped, (90, 90, 120))
    return reconstructed_data


# %% access all the colums of dataframe and get the data
for i in df.columns:
    # get the data from dataframe
    name = i
    df_data = df[i].tolist()
    # get the data from nifti files
    data_list = get_data_from_nifti(df_data)
    # reconstruct patches
    inverse_patch = reconstruct_patches(data_list)

    # save inverse_patch as nifti file
    img = nib.Nifti1Image(inverse_patch, np.eye(4))
    nib.save(img, output_path + "/" + name + ".nii.gz")
    print("saved: %s" % name)

# %%
original_adc = nib.load('input/Masked_ADC.nii')
original_adc_data = np.array(original_adc.dataobj)

modified_original_adc_data = original_adc_data[0:90, 0:90, 0:120]
modified_original_adc_data[np.isnan(modified_original_adc_data)] = 0

# %%check equality of reconstructed_adc and modified_original_adc_data
print(np.array_equal(reconstructed_adc, modified_original_adc_data))
