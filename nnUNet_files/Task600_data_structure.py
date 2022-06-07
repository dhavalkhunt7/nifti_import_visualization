# %% training data will bne used for testing
from pathlib import Path
import nibabel as nb
import numpy as np

# %% data import paths
img_dir = Path("../../../Documents/WSIC/data WSIC/Rats24h")
# train_dir = img_dir / "images_tr"
# labels = img_dir / "labels"

output_dir = Path("../../Documents/WSIC/data WSIC/Task600")
output_training_dir = output_dir / "images_tr"
output_labels_tr = output_dir / "labels_tr"

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

