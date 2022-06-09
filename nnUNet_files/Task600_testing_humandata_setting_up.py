# %%  used for testing
from pathlib import Path
import nibabel as nb
import numpy as np

# %% data import paths
img_dir = Path("../../../Documents/data/Adrian_chamba_strokes_data/Human")

#%%
output_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task600_rat/human_data_testing_set")
output_testing_dir = output_dir / "imagesTs"
output_labels_dir = output_dir / "all_others_ts"
#

#%%

for i in img_dir.glob("*"):
    new_dir = img_dir / i.name
    # print(i.name)

    for j in new_dir.glob("*"):
        # print(j.name)
        img = nb.load(j)

        if j.name == "Masked_ADC.nii":
            # print(j.name)
            new_name = i.name + "_0000.nii.gz"
            nb.save(img, output_testing_dir / new_name)
            print(output_testing_dir / new_name)

        elif j.name == "T2_norm.nii":
            new_name = i.name + "_0001.nii.gz"
            nb.save(img, output_testing_dir / new_name)
            print(output_testing_dir / new_name)

        elif j.name == "GroundTrouth.nii":
            new_name = i.name + "_GroundTrouth.nii.gz"
            nb.save(img, output_labels_dir / new_name)
            print(output_labels_dir / new_name)

        else:
            new_name = i.name + "_" + j.name + ".gz"
            nb.save(img, output_labels_dir / new_name)
            print(output_labels_dir / new_name)


# %%
for i in img_dir.glob("*"):
    new_name = i.name.split("-24h")[0]
    new_dir = img_dir / i.name
    # print(new_dir)

    for k in new_dir.glob("*"):
        # print(k.name)

        if (k.name == "Masked_ADC.nii"):
            img = nb.load(k)
            label_name = new_name + "_0000.nii.gz"
            print(label_name)
            nb.save(img, output_training_dir / label_name)

        if (k.name == "Masked_T2.nii"):
            img = nb.load(k)
            label_name = new_name + "_0001.nii.gz"
            print(label_name)
            nb.save(img, output_training_dir / label_name)

        if (k.name == "GroundTruth24h.nii"):
            img = nb.load(k)
            label_name = new_name + ".nii.gz"
            print(label_name)
            nb.save(img, output_labels_tr / label_name)

# %% testing parrt
output_testing_dir = output_dir / "images_ts"
output_all_testing_dir = output_dir / "all_other_images_connected_ts"

count = 0
for i in img_dir.glob("*"):
    new_name = i.name.split("-24h")[0]
    new_dir = img_dir / i.name
    # print(new_dir)
    new_name = new_name[3:]
    # print(new_name)  # for giving new index for testing data

    for k in new_dir.glob("*"):

        if (count < 20):
            if (k.name == "Masked_ADC.nii"):
                img = nb.load(k)
                label_name = "Rat0" + new_name + "_0000.nii.gz"
                print(label_name)
                nb.save(img, output_testing_dir / label_name)

            elif(k.name == "Masked_T2.nii"):
                img = nb.load(k)
                label_name = "Rat0" + new_name + "_0001.nii.gz"
                print(label_name)
                nb.save(img, output_testing_dir / label_name)

            else:
                img = nb.load(k)
                label_name = "Rat0" + new_name + "_" + k.name + ".gz"
                print(label_name)
                nb.save(img, output_all_testing_dir / label_name)
    count += 1

