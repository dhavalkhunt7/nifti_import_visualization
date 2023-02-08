# %% training data will bne used for testing
import os
from pathlib import Path
import nibabel as nb
import numpy as np

#%%
img_dir =Path("../../../Documents/data/Adrian_chamba_strokes_data/Human")

output_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task601_human")
output_training_dir = output_dir / "imagesTr"
output_labels_tr = output_dir / "labelsTr"
output_labels_ts = output_dir / "labelsTs"
output_testing_dir = output_dir / "imagesTs"
output_testing_all = output_dir / "all_about_testing"

#%%
for i in output_testing_all.glob("*"):
    print(i.name)
    new_dir = i
    for j in new_dir.glob("*"):
        print(j.name)
        if j.name == "GroundTrouth.nii.gz":
            #save it to labelsTs folder
            img = nb.load(j)
            label_name = i.name + "1.nii.gz"
            nb.save(img, output_labels_ts / label_name)

#%%
for i in img_dir.glob("*"):
    # print(i.name)

    if i.name == "MLSTROKEPAT4TP1":
        print("sorry...")
    else:
        # print(i.name)
        new_dir = img_dir / i.name

        for j in new_dir.glob("*"):
            # print(j.name)
            img = nb.load(j)

            if j.name == "Masked_ADC.nii":
                label_name = i.name + "_0000.nii.gz"
                nb.save(img, output_training_dir / label_name)
                # print(i.name)

            elif j.name == "T2_norm.nii":
                label_name = i.name + "_0001.nii.gz"
                nb.save(img, output_training_dir / label_name)

            elif j.name == "GroundTrouth.nii":
                label_name = i.name + ".nii.gz"
                nb.save(img, output_labels_tr / label_name)
                print(label_name)

            else:
                label_name = i.name + "_" + j.name + ".gz"
                # nb.save(img, output_testing / label_name)
                # print(label_name)

#%% for testing
count = 0
for i in img_dir.glob("*"):
    # print(i.name)

    if i.name == "MLSTROKEPAT4TP1":
        print("sorry...")
    else:
        new_dir = img_dir / i.name


        if count < 15:
            directory = os.path.join(output_testing_all, i.name)
            os.mkdir(directory)
            for j in new_dir.glob("*"):
                img = nb.load(j)

                if j.name == "Masked_ADC.nii":
                    label_name = i.name + "1_0000.nii.gz"
                    nb.save(img, output_testing_dir / label_name)
                    print(i.name)

                elif j.name == "T2_norm.nii":
                    label_name = i.name + "1_0001.nii.gz"
                    nb.save(img, output_testing_dir / label_name)

                # elif j.name == "GroundTrouth.nii":
                #     label_name = i.name + ".nii.gz"
                #     nb.save(img, output_labels_tr / label_name)
                #     print(label_name)

                else:
                    label_name = j.name + ".gz"
                    nb.save(img, output_testing_all / i.name / label_name)
                    print(label_name)
        count+=1


