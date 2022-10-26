# %% imports
from pathlib import Path
import nibabel as nb

# %%
database = Path("../../../Documents/data/PETMRI_Glioma/")
# set ouput path in Task502_BrainTumour
output_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task510_BrainTumour/")


# %% function to analyze and save nii files
def nii_analyze_save(data, output_db, counter):
    k_img = nb.load(data)
    label = k.name.replace(".nii", "")

    if counter < 32:
        img_dir = "imagesTr"
        label_dir = "labelsTr"
    else:
        img_dir = "imagesTs"
        label_dir = "labelsTs"

    if label == "T1_mprage":
        img_name = f"{new_dir.name}_0001.nii.gz"
        # save t1 file
        t1_file = output_db / img_dir / img_name
        nb.save(k_img, t1_file)
    elif label == "T1_mprageKM":
        img_name = f"{new_dir.name}_0002.nii.gz"
        # save t1w file
        t1w_file = output_db / img_dir / img_name
        nb.save(k_img, t1w_file)
    elif label == "T2":
        img_name = f"{new_dir.name}_0003.nii.gz"
        # save t2 file
        t2_file = output_db / img_dir / img_name
        nb.save(k_img, t2_file)
    elif label == "Darkfluid":
        img_name = f"{new_dir.name}_0000.nii.gz"
        # save flair file
        flair_file = output_db / img_dir / img_name
        nb.save(k_img, flair_file)
    elif label == "GT":
        img_name = f"{new_dir.name}.nii.gz"
        # save gt file
        gt_file = output_db / label_dir / img_name
        nb.save(k_img, gt_file)


# %%
count = 0
for i in database.glob("*"):
    new_dir = i
    for k in new_dir.glob("*.nii"):
        nii_analyze_save(k, output_path, count)
    count += 1

# %%
# if path doesn't exist, create it
path = output_path / "imagesTr"
path.mkdir(parents=True, exist_ok=True)
path = output_path / "labelsTr"
path.mkdir(parents=True, exist_ok=True)
path = output_path / "imagesTs"
path.mkdir(parents=True, exist_ok=True)
path = output_path / "labelsTs"
path.mkdir(parents=True, exist_ok=True)
