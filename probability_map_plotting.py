#%%
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
# from segmentation_mask_overlay import overlay_masks
import copy
import numpy as np

#%%
database = Path("output")

for i in database.glob("*.nii.gz"):
    print(i)


#%% print current working directory
import os
print(os.getcwd())


#%%
img_path = database / "final_img_625.nii.gz"
img_data = nib.load(img_path).get_fdata()

gt_path = database / "final_gt_625.nii.gz"
gt_data = nib.load(gt_path).get_fdata()

segmentation_path_615 = database / "final_segmentation_625.nii.gz"
segmentation_data_610 = nib.load(segmentation_path_615).get_fdata()

# segmentation_path_620 = database / "final_segmentation_620.nii.gz"
# segmentation_data_620 = nib.load(segmentation_path_620).get_fdata()


#%%
print(gt_data.shape)

#%% rotate data
img_data_rot = np.rot90(img_data, 1)
gt_data_rot = np.rot90(gt_data, 1)
segmentation_data_610_rot = np.rot90(segmentation_data_610, 1)

#%% create a function plot subplots of the image and the masks

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
            #hider border of subplot
            ax[n].set_frame_on(False)
            ax[n].set_title('Slice {}'.format(slice), color='r', fontsize=5)
            n += 1
            slice += 5
    fig.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    return fig

#%%
plt_5 = plot_subplots(img_data_rot, segmentation_data_610_rot, 1, 7)

#%% save the figure
plt_5.savefig("output1/625_segmentation.pdf", bbox_inches="tight", dpi=300)
