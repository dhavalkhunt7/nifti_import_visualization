# %% managing imports
from pathlib import Path
import nibabel as nib
import numpy as np
import os

from patchify import patchify

# %% import data from Documents/data/adrian_data
input_folder = Path("../../../Documents/data/adrian_data/Rats24h")
output_folder = "../nnUNet_raw_data_base/nnUNet_raw_data/Task630_Patch_2d_Rat24h/"


# %% write a function to create patches and append them to a list
def create_patches(input_image_array):
    patches_15 = patchify(input_image_array, (31, 31, 31), step=15)
    # print(patches_15.shape)

    patches_15_list = []
    for i in range(patches_15.shape[0]):
        for j in range(patches_15.shape[1]):
            for k in range(patches_15.shape[2]):
                patch = patches_15[i, j, k, :, :, :]
                patches_15_list.append(patch)
    # print(len(patches_15_list))

    return patches_15_list


# %% save the patches in a folder as .nii.gz
def save_patches(patch_list_whole_img, destination_folder, file_type, counter):
    # if type is t2 or adc then output_folder is output_folder/imagesTs
    if counter < 37:
        if file_type == "t2" or file_type == "adc":
            destination_folder = destination_folder + "/imagesTr"
        elif file_type == "seg":
            destination_folder = destination_folder + "/labelsTr"
    else:
        if file_type == "t2" or file_type == "adc":
            destination_folder = destination_folder + "/imagesTs"
        elif file_type == "seg":
            destination_folder = destination_folder + "/labelsTs"

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for element in range(len(patch_list_whole_img)):
        patch = patch_list_whole_img[element]
        nii_file = nib.Nifti1Image(patch, np.eye(4))
        if file_type == "t2":
            new_name = new_dir_name + '_' + str(element) + '_0001' + '.nii.gz'
        elif file_type == "adc":
            new_name = new_dir_name + '_' + str(element) + '_0000' + '.nii.gz'
            # new_name = new_dir_name + '_0000_' + str(element) + '.nii.gz'
        elif file_type == "seg":
            new_name = new_dir_name + '_' + str(element) + '.nii.gz'
            # new_name = new_dir_name + '_' + str(element) + '.nii.gz'
        nib.save(nii_file, destination_folder + '/' + new_name)
    print(new_dir_name + " " + file_type + ' saved')


# %% create a function to create patches and save them
def create_and_save_patches(input_file, output, type_file, tally):
    load_nifti = nib.load(input_file)
    input_image_array = np.array(load_nifti.dataobj)

    list_patches = create_patches(input_image_array)
    save_patches(list_patches, output, type_file, tally)


# %% trail run
folder = input_folder / "Rat081-24h"
output_fold = "patch_data/"
new_dir_name = "Rat081"
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

# %% perfect run
count = 1
for i in input_folder.glob("*"):
    new_dir = i
    new_dir_name = i.name.replace("-24h", "")
    # print(new_dir_name)
    print(count)

    for j in new_dir.glob("*.nii"):
        if "Masked_ADC" in j.name:
            adc_file = j
            create_and_save_patches(adc_file, output_folder, "adc", count)

        elif "Masked_T2" in j.name:
            t2_file = j
            create_and_save_patches(t2_file, output_folder, "t2", count)

        elif "GroundTruth24h" in j.name:
            gt_file = j
            create_and_save_patches(gt_file, output_folder, "seg", count)

    count += 1
