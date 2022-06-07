#%%
import nibabel as nb
import numpy as np
from pathlib import Path

#%% converting from 484 to 340 training data -- 144 will be added as test data without t2
img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task501_BrainTumour/")
train_dir = img_dir / "imagesTr"

for i in train_dir.glob("*.nii.gz"):
    # print(i)
    data = nb.load(i).get_fdata()
    img_name = i.name
    img_name = img_name.replace('BRATS_','').replace('.nii.gz','')
    for mod in range(4):
        # print(type(mod))
        f = data[:,:,:,mod]
        f_img = nb.Nifti1Image(f, np.eye(4))
        if int(img_name) < 341:
            new_name = str(i).replace('.nii.gz', '_').replace('Task501_BrainTumour', 'Task505_BrainTumour') + \
                       f'{str(mod).zfill(4)}.nii.gz'
            # print(new_name)
            if mod < 3:
                nb.save(f_img, new_name)
                print(new_name)
        else:
            new_name = str(i).replace('.nii.gz', '_').replace('Task501_BrainTumour', 'Task505_BrainTumour').\
                           replace('imagesTr', 'imagesTs') + f'{str(mod).zfill(4)}.nii.gz'
            # print(new_name)
            if mod < 3:
                nb.save(f_img, new_name)
                print(new_name)

#%% labels rearrange for 505
img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task501_BrainTumour/")
labels_dir = img_dir / "labelsTr"

for i in labels_dir.glob("*.nii.gz"):
    img_name = i.name
    lbl_name = img_name.replace('BRATS_', '').replace('.nii.gz', '')

    k_img = nb.load(i)

    if int(lbl_name) < 341:
        new_name = str(i).replace('Task501_BrainTumour', 'Task505_BrainTumour')
        print(new_name)
        nb.save(k_img, new_name)
    else:
        new_name = str(i).replace('Task501_BrainTumour', 'Task505_BrainTumour').replace('labelsTr', 'labelsTs')
        print(new_name)
        nb.save(k_img, new_name)


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
        new_name = str(i).replace('.nii.gz', '_').replace('Task501_BrainTumour', 'Task505_BrainTumour') + f'{str(mod).zfill(4)}.nii.gz'
        print(new_name)
        # nb.save(f_img, new_name)

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

img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/")
test_dir = img_dir / "MICCAI_Brats_2019_Data_VAlidation_pre_for_test/imagesTs"

for i in test_dir.glob("*"):
    print(i)
    # data = nb.load(i).get_fdata()
    # for mod in range(4):
    #     # print(mod)
    #     f = data[:,:,:,mod]
    #     f_img = nb.Nifti1Image(f, np.eye(4))
    #     new_name = str(i).replace('.nii.gz','_').replace('Task501_BrainTumour','converted_data') + f'{str(mod).zfill(4)}.nii.gz'
    #     print(new_name)
    #
    #     nb.save(f_img, new_name)


#%% for brats 2019 hgg


#%% for brats 2019 lgg




#%% training data will bne used for testing
img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing")
train_dir = img_dir / "images_tr"
labels = img_dir / "labels"

for i in train_dir.glob("*.nii.gz"):
    # print(i)
    data = nb.load(i).get_fdata()
    for mod in range(4):
        # print(mod)
        f = data[:,:,:,mod]
        f_img = nb.Nifti1Image(f, np.eye(4))

        new_name = str(i).replace('.nii.gz','_').replace('images_tr','images_tr_converted') + f'{str(mod).zfill(4)}.nii.gz'
        # print(new_name)
        new_name = new_name.replace('BRATS_','BRATS_1')
        # print(new_name)
        nb.save(f_img, new_name)

# for i in labels.glob("*.nii.gz"):
#     k_img = nb.load(i)
#     new_name = str(i).replace('BRATS_', 'BRATS_1')
#     # print(new_name)
#     nb.save(k_img, new_name)