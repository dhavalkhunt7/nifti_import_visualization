#%%
import pathlib as Path
import nibabel as nb

#%% load data using pathlib

target_base = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task605_rat")

gt_dir = target_base / "labelsTs"
pred_dir = target_base / "resultTs"

#%%
for i in target_base.glob("*"):
    print(i.name)
