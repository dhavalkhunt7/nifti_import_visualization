from pathlib import Path
import nibabel as nb

# %% MICCAI_Brats_2019_Data_VAlidation_pre_for_test
img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/new_Task001_BrainTumour/")
test_dir = img_dir / "imagesTs"
output_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/new_Task001_BrainTumour/imagesTs_converted/")

a = 900
for i in test_dir.glob("*"):
    new_dir = test_dir / i.name
    a += 1
    for k in new_dir.glob("*.nii.gz"):
        k_img = nb.load(k)
        img_name = k.name
        # print(img_name)
        find_lable_string = img_name.split("_1_", 1)
        lable = str(find_lable_string[1]).replace(".nii.gz", "")
        # print(lable)
        if (lable == 'flair'):
            new_name = "BRATS_" + str(a) + "_0000.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
            # print(str(new_dir) + "/" + new_name)
        elif (lable == 't1'):
            new_name = "BRATS_" + str(a) + "_0001.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        elif (lable == 't1ce'):
            new_name = "BRATS_" + str(a) + "_0002.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        elif (lable == 't2'):
            new_name = "BRATS_" + str(a) + "_0003.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        else:
            print("error found.....")

# %% Brats_2019 traing data --------HGG-------------for--testing0------------
img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/")
test_dir = img_dir / "Brats_2019_tr_ts/HGG/"
output_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Brats_2019_tr_ts/HGG_converted")

a = 920
for i in test_dir.glob("*"):
    # print(i)
    new_dir = test_dir / i.name
    a += 1
    for k in new_dir.glob("*.nii.gz"):
        # print(k)
        k_img = nb.load(k)
        img_name = k.name
        # print(img_name)
        find_lable_string = img_name.split("_1_", 1)
        lable = str(find_lable_string[1]).replace(".nii.gz", "")
        # print(lable)
        if (lable == 'flair'):
            new_name = "BRATS_" + str(a) + "_0000.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
            # print(str(new_dir) + "/" + new_name)
        elif (lable == 't1'):
            new_name = "BRATS_" + str(a) + "_0001.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        elif (lable == 't1ce'):
            new_name = "BRATS_" + str(a) + "_0002.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        elif (lable == 't2'):
            new_name = "BRATS_" + str(a) + "_0003.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)

# %% Brats_2019 traing data --------LGG-------------for--testing0------------
img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/")
test_dir = img_dir / "Brats_2019_tr_ts/LGG/"
output_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Brats_2019_tr_ts/LGG_converted")

a = 940
for i in test_dir.glob("*"):
    # print(i)
    new_dir = test_dir / i.name
    a += 1
    for k in new_dir.glob("*.nii.gz"):
        # print(k)
        k_img = nb.load(k)
        img_name = k.name
        # print(img_name)
        find_lable_string = img_name.split("_1_", 1)
        lable = str(find_lable_string[1]).replace(".nii.gz", "")
        # print(lable)
        if (lable == 'flair'):
            new_name = "BRATS_" + str(a) + "_0000.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
            # print(str(new_dir) + "/" + new_name)
        elif (lable == 't1'):
            new_name = "BRATS_" + str(a) + "_0001.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        elif (lable == 't1ce'):
            new_name = "BRATS_" + str(a) + "_0002.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        elif (lable == 't2'):
            new_name = "BRATS_" + str(a) + "_0003.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)

# %% Brats_2020 traing data -------------for--testing0------------
img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Brats_2020_tr_dv_ts/")
test_dir = img_dir / "MICCAI_Brats2020_trD_Ts"
output_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Brats_2020_tr_dv_ts/MICCAI_Brats2020_trD_Ts_converted")

a = 960
for i in test_dir.glob("*"):
    # print(i.name)
    task_id = i.name.replace("BraTS20_Training_", "")
    # print(task_id)
    new_dir = test_dir / i.name
    a += 1

    for k in new_dir.glob("*.nii.gz"):
        # print(k)
        k_img = nb.load(k)
        img_name = k.name
        # print(img_name)
        split_str = "_" + task_id + "_"
        find_lable_string = img_name.split(split_str, 1)
        lable = str(find_lable_string[1]).replace(".nii.gz", "")
        # print(lable)

        if (lable == 'flair'):
            new_name = "BRATS_" + str(a) + "_0000.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
            # print(str(new_dir) + "/" + new_name)
        elif (lable == 't1'):
            new_name = "BRATS_" + str(a) + "_0001.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        elif (lable == 't1ce'):
            new_name = "BRATS_" + str(a) + "_0002.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        elif (lable == 't2'):
            new_name = "BRATS_" + str(a) + "_0003.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)

# %% Brats_2020 validation data -------------for--testing0------------
img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Brats_2020_tr_dv_ts/")
test_dir = img_dir / "MICCAI_Brats2020_vd_Ts"
output_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Brats_2020_tr_dv_ts/MICCAI_Brats2020_vd_Ts_converted")

a = 980
for i in test_dir.glob("*"):
    # print(i.name)
    task_id = i.name.replace("BraTS20_Validation_", "")
    print(task_id)
    new_dir = test_dir / i.name
    a += 1

    for k in new_dir.glob("*.nii.gz"):
        # print(k)
        k_img = nb.load(k)
        img_name = k.name
        # print(img_name)
        split_str = "_" + task_id + "_"
        find_lable_string = img_name.split(split_str, 1)
        lable = str(find_lable_string[1]).replace(".nii.gz", "")
        # print(lable)

        if (lable == 'flair'):
            new_name = "BRATS_" + str(a) + "_0000.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
            # print(str(new_dir) + "/" + new_name)
        elif (lable == 't1'):
            new_name = "BRATS_" + str(a) + "_0001.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        elif (lable == 't1ce'):
            new_name = "BRATS_" + str(a) + "_0002.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
        elif (lable == 't2'):
            new_name = "BRATS_" + str(a) + "_0003.nii.gz"
            nb.save(k_img, str(new_dir) + "/" + new_name)
            nb.save(k_img, str(output_path) + "/" + new_name)
