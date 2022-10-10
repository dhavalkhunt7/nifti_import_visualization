# %% managing imports
from pathlib import Path
import nibabel as nib
import numpy as np
import os

import patchify as patchify

# %% import data from Documents/data/adrian_data
input_Human_folder = Path("../../../Documents/data/adrian_data/Human_labelled")
input_rat_folder = Path("../../../Documents/data/adrian_data/Christine_data_Rat24h_devided")
rat_trainings_folder = input_rat_folder / "christine_control_data"
rat_test_folder = input_rat_folder / "christine_therapy_data"

output_folder = "../nnUNet_raw_data_base/nnUNet_raw_data/Task646_combined_patch/"


# %% write a function to create patches and append them to a list
def create_patches(img):
    # create patches of 45*45*45 with stride 15
    patches_1 = patchify.patchify(img, (45, 45, 45), step=15)
    final_patch = patches_1.reshape(-1, 45, 45, 45)

    # extract the patches in 3d shape and save it in a list
    list_patches = []
    for z in range(final_patch.shape[0]):
        list_patches.append(final_patch[z, :, :, :])
    return list_patches


# %% save the patches in a folder as .nii.gz  for human data
def save_patches(patch_list_whole_img, destination_folder, file_type, counter):
    # if type is t2 or adc then output_folder is output_folder/imagesTs
    if counter < 34:
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
        # length of element is 1 then add 00 before the element, if length is 2 then add 0 before the element else
        # add nothing
        if len(str(element)) == 1:
            element_no = "00" + str(element)
        elif len(str(element)) == 2:
            element_no = "0" + str(element)
        else:
            element_no = str(element)

        if file_type == "t2":
            file_name = new_dir_name + '_' + element_no + '_0001' + '.nii.gz'
        elif file_type == "adc":
            file_name = new_dir_name + '_' + element_no + '_0000' + '.nii.gz'
        elif file_type == "seg":
            file_name = new_dir_name + '_' + element_no + '.nii.gz'
        nib.save(nii_file, destination_folder + '/' + file_name)
    print(new_dir_name + " " + file_type + ' saved')


# %%
def whole_process(input_file, output, type_file, tally):
    nifti_file = nib.load(input_file)
    image_array = np.array(nifti_file.dataobj)

    # if input_image_array has nan values then change it to 0
    if np.isnan(image_array).any():
        image_array[np.isnan(image_array)] = 0

    print(np.isnan(image_array).any())

    list_patches = create_patches(image_array)
    save_patches(list_patches, output, type_file, tally)


# %% trail run of
count = 0
for j in input_Human_folder.glob("*.nii"):
    if "Masked_ADC" in j.name:
        adc_file = j
        print(adc_file.name)
        whole_process(adc_file, output_folder, "adc", count)

    elif "T2_norm" in j.name:
        t2_file = j
        whole_process(t2_file, output_folder, "t2", count)

    elif "GroundTrouth" in j.name:
        gt_file = j
        whole_process(gt_file, output_folder, "seg", count)

# %% perfect run for human data
count = 1
for i in input_Human_folder.glob("*"):
    new_dir = i
    # new_dir_name = i.name.replace("-24h", "")
    new_dir_name = i.name
    print(new_dir_name)
    print(count)

    for j in new_dir.glob("*.nii"):
        if "Masked_ADC" in j.name:
            adc_file = j
            print(adc_file.name)
            whole_process(adc_file, output_folder, "adc", count)

        elif "T2_norm" in j.name:
            t2_file = j
            print(t2_file.name)
            whole_process(t2_file, output_folder, "t2", count)

        elif "GroundTrouth" in j.name:
            gt_file = j
            print(gt_file.name)
            whole_process(gt_file, output_folder, "seg", count)

    count += 1


# %% for rat data training and testing folder
def saving_patches(patch_list_whole_img, destination_folder, file_type, folder_type):
    # if type is t2 or adc then output_folder is output_folder/imagesTs
    if folder_type == "train":
        if file_type == "t2" or file_type == "adc":
            destination_folder = destination_folder + "/imagesTr"
        elif file_type == "seg":
            destination_folder = destination_folder + "/labelsTr"
    elif folder_type == "test":
        if file_type == "t2" or file_type == "adc":
            destination_folder = destination_folder + "/imagesTs"
        elif file_type == "seg":
            destination_folder = destination_folder + "/labelsTs"

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for element in range(len(patch_list_whole_img)):
        patch = patch_list_whole_img[element]
        nii_file = nib.Nifti1Image(patch, np.eye(4))
        # length of element is 1 then add 00 before the element, if length is 2 then add 0 before the element else
        # add nothing
        if len(str(element)) == 1:
            element_no = "00" + str(element)
        elif len(str(element)) == 2:
            element_no = "0" + str(element)
        else:
            element_no = str(element)

        if file_type == "t2":
            new_name = new_dir_name + '_' + element_no + '_0001' + '.nii.gz'
        elif file_type == "adc":
            new_name = new_dir_name + '_' + element_no + '_0000' + '.nii.gz'
            # new_name = new_dir_name + '_0000_' + str(element) + '.nii.gz'
        elif file_type == "seg":
            new_name = new_dir_name + '_' + element_no + '.nii.gz'
            # new_name = new_dir_name + '_' + str(element) + '.nii.gz'
        nib.save(nii_file, destination_folder + '/' + new_name)
    print(new_dir_name + " " + file_type + ' saved')


# %% handling whole whole_process for rat data
def whole_process_rat(input_file, output, type_file, folder_type):
    nifti_file = nib.load(input_file)
    image_array = np.array(nifti_file.dataobj)

    # if input_image_array has nan values then change it to 0
    if np.isnan(image_array).any():
        image_array[np.isnan(image_array)] = 0

    print(np.isnan(image_array).any())

    list_patches = create_patches(image_array)
    saving_patches(list_patches, output, type_file, folder_type)


# %% perfect run for rat training data and testing data

for i in rat_trainings_folder.glob("*"):
    new_dir = i
    print(new_dir.name)
    new_dir_name = i.name.replace("-24h", "")
    print(new_dir_name)
    type_folder = "train"

    for j in new_dir.glob("*.nii.gz"):
        if "Masked_ADC" in j.name:
            adc_file = j
            print(adc_file.name)
            whole_process_rat(adc_file, output_folder, "adc", type_folder)

        elif "Masked_T2" in j.name:
            t2_file = j
            print(t2_file.name)
            whole_process_rat(t2_file, output_folder, "t2", type_folder)

        elif "GroundTruth24h" in j.name:
            gt_file = j
            print(gt_file.name)
            whole_process_rat(gt_file, output_folder, "seg", type_folder)


#%%
for i in rat_test_folder.glob("*"):
    new_dir = i
    print(new_dir.name)
    new_dir_name = i.name.replace("-24h", "")
    print(new_dir_name)
    type_folder = "test"

    for j in new_dir.glob("*.nii.gz"):
        if "Masked_ADC" in j.name:
            adc_file = j
            print(adc_file.name)
            whole_process_rat(adc_file, output_folder, "adc", type_folder)

        elif "Masked_T2" in j.name:
            t2_file = j
            print(t2_file.name)
            whole_process_rat(t2_file, output_folder, "t2", type_folder)

        elif "GroundTruth24h" in j.name:
            gt_file = j
            print(gt_file.name)
            whole_process_rat(gt_file, output_folder, "seg", type_folder)

