# %%
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import copy
import numpy as np

# %%
database = Path("../../../Documents/data/adrian_data/devided/final_data")


# %% create a function for the upper code
def get_data_jpm_plotting(task, type):
    for i in task.glob("*"):
        if type == "therapy":
            # get therapy data by name of file
            if i.name == "final_t2_therapy.nii.gz":
                t2_data = nib.load(i).get_fdata()
            elif i.name == "final_adc_therapy.nii.gz":
                adc_data = nib.load(i).get_fdata()
            elif i.name == "final_gt_therapy.nii.gz":
                gt_data = nib.load(i).get_fdata()
        elif type == "control":
            # get control data by name of file
            if i.name == "final_t2_control.nii.gz":
                t2_data = nib.load(i).get_fdata()
            elif i.name == "final_adc_control.nii.gz":
                adc_data = nib.load(i).get_fdata()
            elif i.name == "final_gt_control.nii.gz":
                gt_data = nib.load(i).get_fdata()

    # rotate data for plotting by 270°
    t2_rot = np.rot90(t2_data, 3)
    adc_rot = np.rot90(adc_data, 3)
    gt_rot = np.rot90(gt_data, 3)

    # return data
    return t2_rot, adc_rot, gt_rot


# %% plot the t2 therapy data
# plt.imshow(therapy_t2[:, :, 78], cmap='gray')
# plt.imshow(therapy_gt[:, :, 78], cmap='hot', alpha=0.7)
# # save the figure
# # plt.savefig("therapy_t2_alpha7.pdf", bbox_inches='tight', dpi=300)
# plt.show()


# for other colo options see: https://gist.github.com/endolith/74275dc8fa2bb9a78266


# %% create a function plot subplots of the image and the masks

def plot_subplots(image, mask_img, n_rows, n_cols):
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(150, 50))

    n = 0
    slice_no = 60
    for _ in range(n_rows):
        for _ in range(n_cols):
            ax[n].imshow(image[:, :, slice_no], cmap='gray')
            ax[n].imshow(mask_img[:, :, slice_no], cmap='hot', alpha=0.7)
            ax[n].set_xticks([])
            ax[n].set_yticks([])
            # hider border of subplot
            ax[n].set_frame_on(False)
            ax[n].set_title('Slice {}'.format(slice_no), color='r', fontsize=5)
            n += 1
            slice_no += 5
    fig.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    return fig


# %% create a function plot subplots of the images only

def plot_subplots_single_modality(image, n_rows, n_cols):
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(150, 50))

    n = 0
    slice_no = 60
    for _ in range(n_rows):
        for _ in range(n_cols):
            ax[n].imshow(image[:, :, slice_no], cmap='gray')
            ax[n].set_xticks([])
            ax[n].set_yticks([])
            # hider border of subplot
            ax[n].set_frame_on(False)
            ax[n].set_title('Slice {}'.format(slice_no), color='r', fontsize=5)
            n += 1
            slice_no += 5
    fig.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    return fig


# %%
task = database / "Rats1m"

# %%
therapy_t2, therapy_adc, therapy_gt = get_data_jpm_plotting(task, "therapy")

# %%
# if the data hads nan values, the nan values are replaced with 0
if np.isnan(therapy_t2).any():
    therapy_t2[np.isnan(therapy_t2)] = 0
if np.isnan(therapy_adc).any():
    therapy_adc[np.isnan(therapy_adc)] = 0
if np.isnan(therapy_gt).any():
    therapy_gt[np.isnan(therapy_gt)] = 0


# %% plot the subplots of the t2 therapy data and the mask data same for the adc data
plt_t2_gt = plot_subplots(therapy_t2, therapy_gt, 1, 7)
plt_adc_gt = plot_subplots(therapy_adc, therapy_gt, 1, 7)

# %% folder to save the plots
plot_path = "abcd/" + task.name
# craete a folder if it does not exist
if not Path(plot_path).exists():
    Path(plot_path).mkdir()

# %% save the figure
plt_t2_gt.savefig(plot_path + "/therapy_t2_gt.pdf", bbox_inches="tight", dpi=300)
plt_adc_gt.savefig(plot_path + "/therapy_adc_gt.pdf", bbox_inches="tight", dpi=300)


# %% plot only adc and ony t2
plt_t2 = plot_subplots_single_modality(therapy_t2, 1, 7)
plt_adc = plot_subplots_single_modality(therapy_adc, 1, 7)

# %% save the figure
plt_t2.savefig(plot_path + "/therapy_t2.pdf", bbox_inches="tight", dpi=300)
plt_adc.savefig(plot_path + "/therapy_adc.pdf", bbox_inches="tight", dpi=300)

#%% re-run the code for the control data
control_t2, control_adc, control_gt = get_data_jpm_plotting(task, "control")

# %%
# if the data hads nan values, the nan values are replaced with 0
if np.isnan(control_t2).any():
    control_t2[np.isnan(control_t2)] = 0
if np.isnan(control_adc).any():
    control_adc[np.isnan(control_adc)] = 0
if np.isnan(control_gt).any():
    control_gt[np.isnan(control_gt)] = 0

# %% plot the subplots of the t2 therapy data and the mask data same for the adc data
plt_t2_gt = plot_subplots(control_t2, control_gt, 1, 7)
plt_adc_gt = plot_subplots(control_adc, control_gt, 1, 7)

# %% save the figure
plt_t2_gt.savefig(plot_path + "/control_t2_gt.pdf", bbox_inches="tight", dpi=300)
plt_adc_gt.savefig(plot_path + "/control_adc_gt.pdf", bbox_inches="tight", dpi=300)

# %% plot only adc and ony t2
plt_t2 = plot_subplots_single_modality(control_t2, 1, 7)
plt_adc = plot_subplots_single_modality(control_adc, 1, 7)

# %% save the figure
plt_t2.savefig(plot_path + "/control_t2.pdf", bbox_inches="tight", dpi=300)
plt_adc.savefig(plot_path + "/control_adc.pdf", bbox_inches="tight", dpi=300)
