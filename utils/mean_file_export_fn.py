# %%
import nibabel as nib
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

# %%
database = Path("../../../Documents/data/adrian_data/devided")


# %% create a function to extract all the data from img and add in to the list
def extract_data(img):
    # load nii data
    data = nib.load(img).get_fdata()
    # if img_data has nan values then replace them with 0
    # if np.isnan(data).any():
    #     data[np.isnan(data)] = 0
    # return the data
    return data


# %% create a function for the upper code
def extract_data_from_folder(folder):
    # create a list for all the data
    list_adc = []
    list_t2 = []
    list_gt = []
    for i in folder.glob("*"):
        # print(i.name)
        new_dir = i
        for j in new_dir.glob("*.nii"):
            # print(j.name)
            if j.name == "Masked_ADC.nii":
                # print(j.name)
                # extract the data
                img_data = extract_data(j)
                # if data has more than 1 unique values then add it to the list
                if len(np.unique(img_data)) > 1:
                    list_adc.append(img_data)
            elif j.name == "Masked_T2.nii":
                # print(j.name)
                # extract the data
                img_data = extract_data(j)
                # if data has more than 1 unique values then add it to the list
                if len(np.unique(img_data)) > 1:
                    list_t2.append(img_data)
            elif j.name == "GroundTruth24h.nii" or j.name == "Voi_1w.nii" or j.name == "Voi_72h.nii" or j.name == \
                    "Voi_1m.nii":
                # print(j.name)
                # extract the data
                img_data = extract_data(j)
                # print unique values
                # if data has more than 1 unique values then add it to the list
                if len(np.unique(img_data)) > 1:
                    list_gt.append(img_data)
    return list_adc, list_t2, list_gt


# %% create a function for code 1 to 4
def create_final_data(task):
    # extract the data from the therapy folder
    therapy_adc, therapy_t2, therapy_gt = extract_data_from_folder(task / "therapy")
    # extract the data from the control folder
    control_adc, control_t2, control_gt = extract_data_from_folder(task / "control")

    # create a new folder for the final data
    final_data = database / "final_data" / task.name
    # if the folder does not exist then create it
    if not final_data.exists():
        final_data.mkdir(parents=True)

    # save the final data for therapy and control
    nib.save(nib.Nifti1Image(np.mean(therapy_adc, axis=0), np.eye(4)), final_data / "final_adc_therapy.nii.gz")
    nib.save(nib.Nifti1Image(np.mean(therapy_t2, axis=0), np.eye(4)), final_data / "final_t2_therapy.nii.gz")
    nib.save(nib.Nifti1Image(np.mean(therapy_gt, axis=0), np.eye(4)), final_data / "final_gt_therapy.nii.gz")

    nib.save(nib.Nifti1Image(np.mean(control_adc, axis=0), np.eye(4)), final_data / "final_adc_control.nii.gz")
    nib.save(nib.Nifti1Image(np.mean(control_t2, axis=0), np.eye(4)), final_data / "final_t2_control.nii.gz")
    nib.save(nib.Nifti1Image(np.mean(control_gt, axis=0), np.eye(4)), final_data / "final_gt_control.nii.gz")


# %%
create_final_data(database / "Rats24h")

# %%
create_final_data(database / "Rats72h")

# %%
create_final_data(database / "Rats1w")

# %%
create_final_data(database / "Rats1m")

