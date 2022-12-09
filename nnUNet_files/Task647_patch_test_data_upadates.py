# %% managing imports
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import os

import patchify as patchify

# %% import data from Documents/data/adrian_data]

testing_folder = "../nnUNet_raw_data_base/nnUNet_raw_data/Task650_patch/imagesTs"
labels_folder = "../nnUNet_raw_data_base/nnUNet_raw_data/Task650_patch/labelsTs"

#%%
# count = 0
for i in Path(testing_folder).glob("*_0000.nii.gz"):
    # load nii.gz file
    nii_file = nib.load(i).get_fdata()

    # if array is np.zeros then dont count it
    if np.count_nonzero(nii_file) == 0:
        print(i.name  + " zero array")
        adc_path = str(i)
        t2_path = adc_path.replace("0000", "0001")
        labels_path = adc_path.replace("imagesTs", "labelsTs").replace("_0000", "")
        print(adc_path)
        print(t2_path)
        print(labels_path)
        if os.path.exists(adc_path):
            os.remove(adc_path)
            print("adc File exists")
        # for t2 file and labels file
        if os.path.exists(t2_path):
            os.remove(t2_path)
            print("t2 File exists")
        if os.path.exists(labels_path):
            os.remove(labels_path)
            print("labels File exists")
        continue


#%%
for i in Path(labels_folder).glob("*"):
    # load nii.gz file
    nii_file = nib.load(i).get_fdata()

    # if array is np.zeros then dont count it
    if np.count_nonzero(nii_file) == 0:
        print(i.name  + " zero array")
        print(i)
        labels_path = str(i)
        adc_path = labels_path.replace("labelsTs", "imagesTs").replace(".nii", "_0000.nii")
        t2_path = adc_path.replace("0000", "0001")
        print(adc_path)
        print(t2_path)
        print(labels_path)
        if os.path.exists(adc_path):
            os.remove(adc_path)
            print("adc File exists")
        # for t2 file and labels file
        if os.path.exists(t2_path):
            os.remove(t2_path)
            print("t2 File exists")
        if os.path.exists(labels_path):
            os.remove(labels_path)
            print("labels File exists")
        continue
