# %% managing imports
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# %% import data from Documents/data/adrian_data
input_folder = Path("../../../Documents/data/adrian_data/Rats24h")

# %%
for i in input_folder.glob("*"):
    new_dir = i
    new_dir_name = i.name.replace("-24h", "")
    # print(new_dir_name)

    for j in new_dir.glob("*.nii"):
        print(j)


# %% function to create patches
def create_patches(input_image_array):
    patch_size = 31
    overlap = 15
    patch_list_whole_img = []
    for k in range(input_image_array.shape[2]):
        single_slice = input_image_array[:, :, k]
        for i in range(0, single_slice.shape[0], patch_size - overlap):
            for j in range(0, single_slice.shape[1], patch_size - overlap):
                patch = single_slice[i:i + patch_size, j:j + patch_size]
                if patch.shape == (patch_size, patch_size):
                    patch_list_whole_img.append(patch)
                # patch_list.append(patch)
    return patch_list_whole_img


# %% save the patches in a folder as .nii.gz
def save_patches(patch_list_whole_img, output_folder, type, count):
    # if type is t2 or adc then output_folder is output_folder/imagesTs
    if count < 37:
        if type == "t2" or type == "adc":
            output_folder = output_folder + "/imagesTr"
        elif type == "seg":
            output_folder = output_folder + "/labelsTr"
    else:
        if type == "t2" or type == "adc":
            output_folder = output_folder + "/imagesTs"
        elif type == "seg":
            output_folder = output_folder + "/labelsTs"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for element in range(len(patch_list_whole_img)):
        patch = patch_list_whole_img[element]
        nii_file = nib.Nifti1Image(patch, np.eye(4))
        if type == "t2":
            new_name = new_dir_name + '_0001_' + str(element) + '.nii.gz'
        elif type == "adc":
            new_name = new_dir_name + '_0000_' + str(element) + '.nii.gz'
        elif type == "seg":
            new_name = new_dir_name + '_' + str(element) + '.nii.gz'
        nib.save(nii_file, output_folder + '/' + new_name)
    print(new_dir_name + ' saved')


# %% create a function to create patches and save them
def create_and_save_patches(input_file, output_folder, type_file, count):
    load_nifti = nib.load(input_file)
    input_image_array = np.array(load_nifti.dataobj)

    patch_size = 31
    overlap = 15
    patch_list_whole_img = []
    for k in range(input_image_array.shape[2]):
        single_slice = input_image_array[:, :, k]
        for i in range(0, single_slice.shape[0], patch_size - overlap):
            for j in range(0, single_slice.shape[1], patch_size - overlap):
                patch = single_slice[i:i + patch_size, j:j + patch_size]
                if patch.shape == (patch_size, patch_size):
                    patch_list_whole_img.append(patch)
                # patch_list.append(patch)

    list_patches = patch_list_whole_img

    if count < 37:
        if type_file == "t2" or type_file == "adc":
            output_folder = output_folder + "/imagesTr"
        elif type_file == "seg":
            output_folder = output_folder + "/labelsTr"
    else:
        if type_file == "t2" or type_file == "adc":
            output_folder = output_folder + "/imagesTs"
        elif type_file == "seg":
            output_folder = output_folder + "/labelsTs"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for element in range(len(patch_list_whole_img)):
        patch = patch_list_whole_img[element]
        nii_file = nib.Nifti1Image(patch, np.eye(4))
        if type_file == "t2":
            new_name = new_dir_name + '_0001_' + str(element) + '.nii.gz'
        elif type_file == "adc":
            new_name = new_dir_name + '_0000_' + str(element) + '.nii.gz'
        elif type_file == "seg":
            new_name = new_dir_name + '_' + str(element) + '.nii.gz'
        nib.save(nii_file, output_folder + '/' + new_name)
    print(new_dir_name + ' saved')


# %%
folder = input_folder / "Rat081-24h"
output_fold = "patch_data/"

for i in folder.glob("*.nii"):
    if "Masked_ADC" in i.name:
        adc_file = i
        create_and_save_patches(adc_file, output_fold, "adc", 1)

    elif "Masked_T2" in i.name:
        t2_file = i
        create_and_save_patches(t2_file, output_fold, "t2", 1)

    elif "GroundTruth24h" in i.name:
        gt_file = i
        create_and_save_patches(gt_file, output_fold, "seg", 1)


#%%
output = Path("patch_data")
for i in output.glob("*"):
    print(i)
