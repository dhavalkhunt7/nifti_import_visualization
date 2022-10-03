# %%
import patchify as patchify
from pathlib import Path
import nibabel as nib
import numpy as np
import os

# %%
input_folder = Path("../../../Documents/data/adrian_data/Rats24h")
output_folder = "../nnUNet_raw_data_base/nnUNet_raw_data/Task630_Patch_3d_Rat24h"


# %%
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


# %%
def create_and_save_patches(input_file, output, type_file, tally):
    load_nifti = nib.load(input_file)
    input_image_array = np.array(load_nifti.dataobj)

    # if input_image_array has nan values then change it to 0
    if np.isnan(input_image_array).any():
        input_image_array[np.isnan(input_image_array)] = 0

    print(np.isnan(input_image_array).any())

    patches = patchify.patchify(input_image_array, (31, 31, 31), step=15)
    final_patch = patches.reshape(-1, 31, 31, 31)
    # extract the patches in 3d shape and save it in a list
    list_patches = []
    for i in range(final_patch.shape[0]):
        list_patches.append(final_patch[i, :, :, :])
    # print(patches.shape)

    # list_patches = create_patches(input_image_array)
    save_patches(list_patches, output, type_file, tally)


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
