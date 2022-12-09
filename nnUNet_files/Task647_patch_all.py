# %% managing imports
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import os

import patchify as patchify

# %% import data from Documents/data/adrian_data]

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

#%%
data_path = Path("input/data")

#%% create function to make patches
def create_patches_diff_data(img):
# create patches of 30*30*30 with stride 15
    patches_1 = patchify.patchify(img, (30, 30, 30), step=15)
    final_patch = patches_1.reshape(-1, 30, 30, 30)

    # extract the patches in 3d shape and save it in a list
    list_patches = []
    for z in range(final_patch.shape[0]):
        list_patches.append(final_patch[z, :, :, :])
    return list_patches


#%% create cut function to preprocess data
def cut_data(data, type):
    if type == "60":
        data_1 = data[:90, :90, :]
    elif type == "100":
        data_1 = np.zeros((75, 45, 30))
        # ADD 60 data to 0s array
        data_1[:, :, :24] = data[:75, :45, :]
    elif type == "human":
        data_1 = data[:120, :135, :120]
    else:
        print("wrong type")

    # remove nan values if any
    if np.isnan(data_1).any():
        data_1[np.isnan(data_1)] = 0

    return data_1

#%% create function to save patches
def save_patches(patch_list, destination_folder, file_type, counter, new_dir_name, type_data):
    if type_data == "human":
        counter_threshold = int(48*0.7)
    elif type_data == "60":
        counter_threshold = int(53*0.7)
    elif type_data == "100":
        counter_threshold = int(35*0.7)

    # if type is t2 or adc then output_folder is output_folder/imagesTs
    if counter < counter_threshold:
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
    for element in range(len(patch_list)):
        patch = patch_list[element]
        nii_file = nib.Nifti1Image(patch, np.eye(4))
        # length of element is 1 then add 00 before the element, if length is 2 then add 0 before the element else
        # add nothing
        if len(str(element)) == 1:
            element_no = "00" + str(element)
        elif len(str(element)) == 2:
            element_no = "0" + str(element)
        else:
            element_no = str(element)

        # element name has .nii or .nii.gz then remove it
        if ".nii" in new_dir_name:
            new_dir_name = new_dir_name.replace(".nii", "")
        elif ".nii.gz" in new_dir_name:
            new_dir_name = new_dir_name.replace(".nii.gz", "")

        if file_type == "t2":
            file_name = new_dir_name + '_' + element_no + '_0001' + '.nii.gz'
        elif file_type == "adc":
            file_name = new_dir_name + '_' + element_no + '_0000' + '.nii.gz'
        elif file_type == "seg":
            file_name = new_dir_name + '_' + element_no + '.nii.gz'
        nib.save(nii_file, destination_folder + '/' + file_name)
    print(new_dir_name + " " + file_type + ' saved')

#%%
for i in data_path.glob("*"):
    print(i.name)

    img = nib.load(i)
    data = img.get_fdata()
    print(data.shape)

    # cut data based on folder name
    data_new = cut_data(data, i.name.split(".")[0])
    print(data_new.shape)

    # create patches
    list_patches = create_patches_diff_data(data_new)
    print(len(list_patches))
    save_patches(list_patches, output_folder, "adc", 1, i.name)
    print("....")

#%%
def whole_process(input_file, output, type_file, tally, type_data):
    nifti_file = nib.load(input_file)
    image_array = np.array(nifti_file.dataobj)

    # preprocess data
    image_array = cut_data(image_array, type_data)

    list_patches = create_patches_diff_data(image_array)

    if type_data == "human":
        # get parent dir name
        parent_dir = os.path.basename(os.path.dirname(input_file))
        print(parent_dir)
        save_patches(list_patches, output, type_file, tally, parent_dir, type_data)
    elif type_data == "60":
        #get parent dir name
        parent_dir = os.path.basename(os.path.dirname(input_file))
        parent_dir = parent_dir.split("-")[0]
        print(parent_dir)

        save_patches(list_patches, output, type_file, tally, parent_dir, type_data)
    elif type_data == "100":
        # get parent dir name
        parent_dir = os.path.basename(os.path.dirname(input_file))
        print(parent_dir)

        save_patches(list_patches, output, type_file, tally, parent_dir, type_data)


#%% human folder
count = 0
for i in Path(upadated_folder + "/human").glob("*"):
    print(i.name)
    new_dir = i
    for j in new_dir.glob("*"):
        if "Masked_ADC" in j.name:
            adc_file = j
            print(adc_file.name)
            whole_process(adc_file, output_folder, "adc", count, "human")
            # whole_process(adc_file, output_folder, "adc", count, "human")

        elif "T2_norm" in j.name:
            t2_file = j
            print(t2_file.name)
            whole_process(t2_file, output_folder, "t2", count, "human")
            # whole_process(t2_file, output_folder, "t2", count)

        elif "GroundTrouth" in j.name:
            gt_file = j
            print(gt_file.name)
            whole_process(gt_file, output_folder, "seg", count, "human")
    count += 1

#%% 60 folder
count = 0
for i in Path(upadated_folder + "/mcao_60").glob("*"):
    print(i.name)
    new_dir = i
    for j in new_dir.glob("*"):
        if "Masked_ADC" in j.name:
            adc_file = j
            print(adc_file.name)
            whole_process(adc_file, output_folder, "adc", count, "60")
            # whole_process(adc_file, output_folder, "adc", count, "human")

        elif "Masked_T2" in j.name:
            t2_file = j
            print(t2_file.name)
            whole_process(t2_file, output_folder, "t2", count, "60")
            # whole_process(t2_file, output_folder, "t2", count)

        elif "GroundTruth24h" in j.name:
            gt_file = j
            print(gt_file.name)
            whole_process(gt_file, output_folder, "seg", count, "60")
    count += 1

#%% 100 folder
count = 0
for i in Path(upadated_folder + "/mcao_100").glob("*"):
    print(i.name)
    new_dir = i
    for j in new_dir.glob("*"):
        if "ADC" in j.name:
            adc_file = j
            print(adc_file.name)
            whole_process(adc_file, output_folder, "adc", count, "100")
            # whole_process(adc_file, output_folder, "adc", count, "human")

        elif "T2W_c" in j.name:
            t2_file = j
            print(t2_file.name)
            whole_process(t2_file, output_folder, "t2", count, "100")
            # whole_process(t2_file, output_folder, "t2", count)

        elif "Voi24" in j.name:
            gt_file = j
            print(gt_file.name)
            whole_process(gt_file, output_folder, "seg", count, "100")
    count += 1


#%% delete folder with files if exists
def delete_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        print(folder_name + " deleted")
    else:
        print(folder_name + " does not exists")

delete_folder(output_folder)

#%% create folder
create_folder(output_folder)


