#%%
from pathlib import Path
import nibabel as nb
import numpy as np

#%%
database = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task505_BrainTumour/PETMRI")
results = database / "result_2d"


#%%
#load nii_file
img = nb.load("for_sd/17.nii.gz").get_fdata()
print(img.shape)
gt_img = nb.load("for_sd/gt_17.nii.gz").get_fdata()
print(gt_img.shape)

#%%
#print unique values
print(np.unique(img))
print(np.unique(gt_img))
