# %%
import nibabel as nib
import numpy as np
from pathlib import Path
import copy
import matplotlib.pyplot as plt
from utilities.jpm_related_functions import *


#%%
dataset_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task610_rat/testing/jpm_&_mean_files/")



#%% 24h
task_name = "Task1/24h_all"
data_path = dataset_path / task_name


for i in data_path.iterdir():
    print(i.name)

unet_file = data_path / "final_seg_nnunet.nii.gz"
gmm_file = data_path / "final_seg_gmm.nii.gz"
t2_file = data_path / "final_t2.nii.gz"
gt_file = data_path / "final_gt.nii.gz"


#%% load data
unet_data = nib.load(unet_file).get_fdata()
gmm_data = nib.load(gmm_file).get_fdata()
t2_data = nib.load(t2_file).get_fdata()
gt_data = nib.load(gt_file).get_fdata()

unet_data_rot = np.rot90(unet_data, 3)
gmm_data_rot = np.rot90(gmm_data, 3)
t2_data_rot = np.rot90(t2_data, 3)
gt_data_rot = np.rot90(gt_data, 3)


#%% plot using plot_subplots
plt_t2_gt = plot_subplots(t2_data_rot, gt_data_rot, 1, 6)
plt_t2_unet = plot_subplots(t2_data_rot, unet_data_rot, 1, 6)
plt_t2_gmm = plot_subplots(t2_data_rot, gmm_data_rot, 1, 6)

# %% save the figure
plt_t2_gt.savefig(str(data_path) + "/t2_gt.pdf", bbox_inches="tight", dpi=300)
plt_t2_unet.savefig(str(data_path) + "/t2_unet.pdf", bbox_inches="tight", dpi=300)
plt_t2_gmm.savefig(str(data_path) + "/t2_gmm.pdf", bbox_inches="tight", dpi=300)


#%% 1w gt
task_name = "Task1/1w_gt"
data_path = dataset_path / task_name


for i in data_path.iterdir():
    print(i.name)

# unet_file = data_path / "final_seg_nnunet.nii.gz"
# gmm_file = data_path / "final_seg_gmm.nii.gz"
t2_file = data_path / "final_t2.nii.gz"
gt_file = data_path / "final_gt.nii.gz"


#%% load data
# unet_data = nib.load(unet_file).get_fdata()
# gmm_data = nib.load(gmm_file).get_fdata()
t2_data = nib.load(t2_file).get_fdata()
gt_data = nib.load(gt_file).get_fdata()

# unet_data_rot = np.rot90(unet_data, 3)
# gmm_data_rot = np.rot90(gmm_data, 3)
t2_data_rot = np.rot90(t2_data, 3)
gt_data_rot = np.rot90(gt_data, 3)


#%% plot using plot_subplots
plt_t2_gt = plot_subplots(t2_data_rot, gt_data_rot, 1, 6)
# plt_t2_unet = plot_subplots(t2_data_rot, unet_data_rot, 1, 6)
# plt_t2_gmm = plot_subplots(t2_data_rot, gmm_data_rot, 1, 6)

# %% save the figure
plt_t2_gt.savefig(str(data_path) + "/t2_gt.pdf", bbox_inches="tight", dpi=300)
# plt_t2_unet.savefig(str(data_path) + "/t2_unet.pdf", bbox_inches="tight", dpi=300)
# plt_t2_gmm.savefig(str(data_path) + "/t2_gmm.pdf", bbox_inches="tight", dpi=300)


#----------------- Task 4-----------------
#%% 24h
task_name = "Task4/control"
data_path = dataset_path / task_name


for i in data_path.iterdir():
    print(i.name)

unet_file = data_path / "final_seg_unet.nii.gz"
gmm_file = data_path / "final_seg.nii.gz"
t2_file = data_path / "final_t2.nii.gz"
gt_file = data_path / "final_gt.nii.gz"


#%% load data
unet_data = nib.load(unet_file).get_fdata()
gmm_data = nib.load(gmm_file).get_fdata()
t2_data = nib.load(t2_file).get_fdata()
gt_data = nib.load(gt_file).get_fdata()

unet_data_rot = np.rot90(unet_data, 3)
gmm_data_rot = np.rot90(gmm_data, 3)
t2_data_rot = np.rot90(t2_data, 3)
gt_data_rot = np.rot90(gt_data, 3)


#%% plot using plot_subplots
plt_t2_gt = plot_subplots(t2_data_rot, gt_data_rot, 1, 6)
plt_t2_unet = plot_subplots(t2_data_rot, unet_data_rot, 1, 6)
plt_t2_gmm = plot_subplots(t2_data_rot, gmm_data_rot, 1, 6)

# %% save the figure
plt_t2_gt.savefig(str(data_path) + "/t2_gt.pdf", bbox_inches="tight", dpi=300)
plt_t2_unet.savefig(str(data_path) + "/t2_unet.pdf", bbox_inches="tight", dpi=300)
plt_t2_gmm.savefig(str(data_path) + "/t2_gmm.pdf", bbox_inches="tight", dpi=300)



#%% therapy
task_name = "Task4/therapy"
data_path = dataset_path / task_name


for i in data_path.iterdir():
    print(i.name)

unet_file = data_path / "final_seg_unet.nii.gz"
gmm_file = data_path / "final_seg.nii.gz"
t2_file = data_path / "final_t2.nii.gz"
gt_file = data_path / "final_gt.nii.gz"


#%% load data
unet_data = nib.load(unet_file).get_fdata()
gmm_data = nib.load(gmm_file).get_fdata()
t2_data = nib.load(t2_file).get_fdata()
gt_data = nib.load(gt_file).get_fdata()

unet_data_rot = np.rot90(unet_data, 3)
gmm_data_rot = np.rot90(gmm_data, 3)
t2_data_rot = np.rot90(t2_data, 3)
gt_data_rot = np.rot90(gt_data, 3)


#%% plot using plot_subplots
plt_t2_gt = plot_subplots(t2_data_rot, gt_data_rot, 1, 6)
plt_t2_unet = plot_subplots(t2_data_rot, unet_data_rot, 1, 6)
plt_t2_gmm = plot_subplots(t2_data_rot, gmm_data_rot, 1, 6)

# %% save the figure
plt_t2_gt.savefig(str(data_path) + "/t2_gt.pdf", bbox_inches="tight", dpi=300)
plt_t2_unet.savefig(str(data_path) + "/t2_unet.pdf", bbox_inches="tight", dpi=300)
plt_t2_gmm.savefig(str(data_path) + "/t2_gmm.pdf", bbox_inches="tight", dpi=300)






#%% 72h
task_name = "72h"
data_path = dataset_path / task_name

unet_file = data_path / "final_seg_unet.nii.gz"
gmm_file = data_path / "final_seg.nii.gz"
t2_file = data_path / "final_t2.nii.gz"
gt_file = data_path / "final_gt.nii.gz"


#%% load data
unet_data = nib.load(unet_file).get_fdata()
gmm_data = nib.load(gmm_file).get_fdata()
t2_data = nib.load(t2_file).get_fdata()
gt_data = nib.load(gt_file).get_fdata()


unet_data_rot = np.rot90(unet_data, 3)
gmm_data_rot = np.rot90(gmm_data, 3)
t2_data_rot = np.rot90(t2_data, 3)
gt_data_rot = np.rot90(gt_data, 3)


#%% plot using plot_subplots
plt_t2_gt = plot_subplots(t2_data_rot, gt_data_rot, 1, 8)
plt_t2_unet = plot_subplots(t2_data_rot, unet_data_rot, 1, 8)
plt_t2_gmm = plot_subplots(t2_data_rot, gmm_data_rot, 1, 8)

# %% save the figure
plt_t2_gt.savefig(str(data_path) + "/t2_gt_old.pdf", bbox_inches="tight", dpi=300)
plt_t2_unet.savefig(str(data_path) + "/t2_unet.pdf", bbox_inches="tight", dpi=300)
plt_t2_gmm.savefig(str(data_path) + "/t2_gmm.pdf", bbox_inches="tight", dpi=300)

#%% 1w
task_name = "1w"
data_path = dataset_path / task_name

unet_file = data_path / "final_seg_unet.nii.gz"
gmm_file = data_path / "final_seg.nii.gz"
t2_file = data_path / "final_t2.nii.gz"
gt_file = data_path / "final_gt.nii.gz"


#%% load data
unet_data = nib.load(unet_file).get_fdata()
gmm_data = nib.load(gmm_file).get_fdata()
t2_data = nib.load(t2_file).get_fdata()
gt_data = nib.load(gt_file).get_fdata()


unet_data_rot = np.rot90(unet_data, 3)
gmm_data_rot = np.rot90(gmm_data, 3)
t2_data_rot = np.rot90(t2_data, 3)
gt_data_rot = np.rot90(gt_data, 3)


#%% plot using plot_subplots
plt_t2_gt = plot_subplots(t2_data_rot, gt_data_rot, 1, 8)
plt_t2_unet = plot_subplots(t2_data_rot, unet_data_rot, 1, 8)
plt_t2_gmm = plot_subplots(t2_data_rot, gmm_data_rot, 1, 8)

# %% save the figure
plt_t2_gt.savefig(str(data_path) + "/t2_gt_old.pdf", bbox_inches="tight", dpi=300)
print("done")
plt_t2_unet.savefig(str(data_path) + "/t2_unet.pdf", bbox_inches="tight", dpi=300)
print("done")
plt_t2_gmm.savefig(str(data_path) + "/t2_gmm.pdf", bbox_inches="tight", dpi=300)
print("done")

#%% 1m
task_name = "1m"
data_path = dataset_path / task_name

unet_file = data_path / "final_seg_unet.nii.gz"
gmm_file = data_path / "final_seg.nii.gz"
t2_file = data_path / "final_t2.nii.gz"
gt_file = data_path / "final_gt.nii.gz"

#%% load data
unet_data = nib.load(unet_file).get_fdata()
gmm_data = nib.load(gmm_file).get_fdata()
t2_data = nib.load(t2_file).get_fdata()
gt_data = nib.load(gt_file).get_fdata()

unet_data_rot = np.rot90(unet_data, 3)
gmm_data_rot = np.rot90(gmm_data, 3)
t2_data_rot = np.rot90(t2_data, 3)
gt_data_rot = np.rot90(gt_data, 3)

#%% plot using plot_subplots
plt_t2_gt = plot_subplots(t2_data_rot, gt_data_rot, 1, 8)
plt_t2_unet = plot_subplots(t2_data_rot, unet_data_rot, 1, 8)
plt_t2_gmm = plot_subplots(t2_data_rot, gmm_data_rot, 1, 8)

# %% save the figure
plt_t2_gt.savefig(str(data_path) + "/t2_gt_old.pdf", bbox_inches="tight", dpi=300)
plt_t2_unet.savefig(str(data_path) + "/t2_unet.pdf", bbox_inches="tight", dpi=300)
plt_t2_gmm.savefig(str(data_path) + "/t2_gmm.pdf", bbox_inches="tight", dpi=300)





#%%
for i in dataset_path.iterdir():
    print(i.name)
