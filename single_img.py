# %% managing imports
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import os

import patchify as patchify

# %% import data from Documents/data/adrian_data

input_folder = "../nnUNet_raw_data_base/nnUNet_raw_data/Task647_patch/main_files"
upadated_folder = "../nnUNet_raw_data_base/nnUNet_raw_data/Task647_patch/filtered_files"
output_folder = "../nnUNet_raw_data_base/nnUNet_raw_data/Task647_patch/images_trail"

#%% craete folder if not exists
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(folder_name + " created")
    else:
        print(folder_name + " already exists")


#%%

human_folder = upadated_folder + "/human"
mcao_60_folder = upadated_folder + "/mcao_60"
mcao_100_folder = upadated_folder + "/mcao_100"

#%%
for i in Path(human_folder).iterdir():
    print(i)

#%%
# copy all the data with folders of folder from one folder to another folder

def copy_files(source, destination):
    shutil.copytree(source, destination, dirs_exist_ok=True)

#%%
copy_files(input_folder + "/Human_labelled" ,human_folder)

#%%
copy_files(input_folder + "/Theranostics" ,mcao_100_folder)

#%%
copy_files(input_folder + "/christine_control_data" ,mcao_60_folder)

#%%
for i in Path(input_folder).glob("*"):
    print(i)

#%% delete folfder if exists
def delete_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        print(folder_name + " deleted")
    else:
        print(folder_name + " does not exists")

delete_folder(mcao_60_folder)
# delete_folder(mcao_100_folder)

#%% create folder if not exists
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(folder_name + " created")
    else:
        print(folder_name + " already exists")

create_folder(human_folder)

#%%
for i in Path(input_folder).glob('*'):
    if i.name == "Human_labelled":
        print(i.name)
        dir = i
        # copy all the files from dir i to human folder
        for file in dir.iterdir():
            shutil.copy(file, human_folder)

    elif i.name == "Theranostics":
        print(i.name)
    elif i.name == "christine_therapy_data":
        print(i.name)
    elif i.name == "christine_control_data":
        print(i.name)



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
    if counter < 24:
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


#%% creeate folder if not exist
folder_name = "../nnUNet_raw_data_base/nnUNet_raw_data/Task647_patch/"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# print(output_folder)

#%%
output_folder = '../nnUNet_raw_data_base/nnUNet_raw_data/Task647_patch_trail/images_trail'

copy_files(output_folder, folder_name)