# %%
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import copy
import numpy as np

# %%
database = Path("../../../Documents/data/adrian_data/devided/final_data")

# %%
task = database / "Rats24h"


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


# %%
therapy_t2, therapy_adc, therapy_gt = get_data_jpm_plotting(task, "therapy")

#%%
# if the data hads nan values, the nan values are replaced with 0
if np.isnan(therapy_t2).any():
    therapy_t2[np.isnan(therapy_t2)] = 0
if np.isnan(therapy_adc).any():
    therapy_adc[np.isnan(therapy_adc)] = 0
if np.isnan(therapy_gt).any():
    therapy_gt[np.isnan(therapy_gt)] = 0

# %% plot the t2 therapy data
plt.imshow(therapy_t2[:, :, 78], cmap='gray')
plt.imshow(therapy_gt[:, :, 78], cmap='Oranges', alpha=0.7)
# save the figure
# plt.savefig("therapy_t2_alpha7.pdf", bbox_inches='tight', dpi=300)
plt.show()

# for other colo options see: https://gist.github.com/endolith/74275dc8fa2bb9a78266


# %% create a function plot subplots of the image and the masks

def plot_subplots(image, mask_img, n_rows, n_cols):
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(150, 50))

    my_cmap = copy.copy(plt.cm.get_cmap('gray'))  # get a copy of the gray color map
    my_cmap.set_bad(alpha=0)

    n = 0
    slice = 60
    for _ in range(n_rows):
        for _ in range(n_cols):
            # ax[n].imshow(image[:, :, slice])
            ax[n].imshow(image[:, :, slice], cmap='gray')
            ax[n].imshow(mask_img[:, :, slice], cmap='Greens', alpha=0.7)
            # ax[n].imshow(mask_img[:, :, slice], cmap='Purples', alpha=0.7)
            ax[n].set_xticks([])
            ax[n].set_yticks([])
            # hider border of subplot
            ax[n].set_frame_on(False)
            ax[n].set_title('Slice {}'.format(slice), color='r', fontsize=5)
            n += 1
            slice += 5
    fig.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    return fig


# %%
plt_5 = plot_subplots(img_data_rot, segmentation_data_610_rot, 1, 7)

# %% save the figure
plt_5.savefig("output1/625_segmentation.pdf", bbox_inches="tight", dpi=300)
