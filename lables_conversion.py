#%%
import nibabel as nb
import numpy as np
from pathlib import Path

#%%
img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task501_BrainTumour/")
train_dir = img_dir / "imagesTr"
test_dir = img_dir / "imagesTs"

for i in train_dir.glob("*.nii.gz"):
    #print(i)
    data = nb.load(i).get_fdata()
    for mod in range(4):
        # print(mod)
        f = data[:,:,:,mod]
        f_img = nb.Nifti1Image(f, np.eye(4))
        new_name = str(i).replace('.nii.gz','_').replace('Task501_BrainTumour','converted_data') + f'{str(mod).zfill(4)}.nii.gz'
        print(new_name)
        nb.save(f_img, new_name)

#%%

for i in test_dir.glob("*.nii.gz"):
    print(i)
    data = nb.load(i).get_fdata()
    for mod in range(4):
        # print(mod)
        f = data[:,:,:,mod]
        f_img = nb.Nifti1Image(f, np.eye(4))
        new_name = str(i).replace('.nii.gz','_').replace('Task501_BrainTumour','converted_data') + f'{str(mod).zfill(4)}.nii.gz'
        print(new_name)
        nb.save(f_img, new_name)

    # for mod in range(4):
    #     f = data[:,:,:,i]
    #     new_name = i / f'{str(mod).zfill(4)}.nii.gz'
    #     nb.save(f, "../nnUNet_raw_data_base/nnUNet_raw_data/converted_data/imagesTs/" + new_name)



#%% for brats 2019 data validation miccai

img_dir = Path("../nifti_")
train_dir = img_dir / "imagesTr"
test_dir = img_dir / "imagesTs"

for i in train_dir.glob("*.nii.gz"):
    #print(i)
    data = nb.load(i).get_fdata()
    for mod in range(4):
        # print(mod)
        f = data[:,:,:,mod]
        f_img = nb.Nifti1Image(f, np.eye(4))
        new_name = str(i).replace('.nii.gz','_').replace('Task501_BrainTumour','converted_data') + f'{str(mod).zfill(4)}.nii.gz'
        print(new_name)

        nb.save(f_img, new_name)


#%% for brats 2019 hgg


#%% for brats 2019 lgg