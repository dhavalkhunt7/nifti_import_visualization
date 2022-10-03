#%%
import os
from pathlib import Path

import nibabel as nib
import numpy as np

#%%
input_database = "../../Documents/WSIC/data WSIC/Rats24h"
output_database = "../../Documents/WSIC/data WSIC/Trail_task"


#%%


#%% get and save the main nifti files
def get_and_save_nifti(input_file, destination_folder, file_type, counter):

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

    load_nifti = nib.load(input_file)

    if file_type == "seg":
        seg_data = load_nifti.get_fdata()
        seg_data[seg_data != 0] = 0
        nii_file = nib.Nifti1Image(seg_data, np.eye(4))
    else:
        nii_file = input_file

    if file_type == "t2":
        new_name = new_dir_name + '_0001' + '.nii.gz'
    elif file_type == "adc":
        new_name = new_dir_name + '_0000' + '.nii.gz'
    elif file_type == "seg":
        new_name = new_dir_name + '.nii.gz'
        # new_name = new_dir_name + '_' + str(element) + '.nii.gz'

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    nib.save(nii_file, destination_folder + '/' + new_name)
    print(new_dir_name + " " + file_type + ' saved')


#%%import data using Pathlib
input_folder = Path(input_database)
count = 0
for file in input_folder.iterdir():
    new_dir = file
    new_dir_name = new_dir.name.replace("-24h", "")

    for i in new_dir.glob("*.nii"):
        if i.name == "Masked_ADC.nii":
            get_and_save_nifti(i, output_database, "adc", count)
        elif i.name == "Masked_T2.nii":
            get_and_save_nifti(i, output_database, "t2", count)
        elif i.name == "GroundTruth24h.nii":
            get_and_save_nifti(i, output_database, "seg", count)
    count += 1
