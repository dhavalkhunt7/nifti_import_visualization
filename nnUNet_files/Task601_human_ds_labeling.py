# %% training data will bne used for testing
from pathlib import Path
import nibabel as nb
import numpy as np

#%%
img_dir =Path("../../../Documents/data/Adrian_chamba_strokes_data/Human")

output_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task601_human")
output_training_dir = output_dir / "imagesTr"
output_labels_tr = output_dir / "labelsTr"
output_testing_dir = output_dir / "imagesTs"
output_testing = output_dir /"all_other_ts"

#%%
for i in img_dir.glob("*"):
    # print(i.name)
    new_dir = img_dir / i.name

    for j in new_dir.glob("*"):
        # print(j.name)

        if j.name == "Masked_ADC.nii":
            print()
        elif j.name == "T2.norm.nii":
            print
        elif j.name == "GroundTrouth.nii":
            print
        else:
            print()


