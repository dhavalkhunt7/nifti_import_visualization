from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing")

# %%
original_data = img_dir / "images_tr_converted"
labels = img_dir / "labels"
predicted_dir = img_dir / "prediction_files_for_dc"

# %%

img = nib.load(original_data / "BRATS_1020_0002.nii.gz")
epi_img_data = img.get_fdata()
epi_img_data.shape

# %%
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    # fig.tight_layout()
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, origin="lower")
        axes[i].axis('off')

slice_0 = epi_img_data[:, 75 , :]
slice_1 = epi_img_data[:, 110, :]
slice_2 = epi_img_data[:, 125, :]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


#%% labels
label_img = nib.load(labels / "BRATS_1020.nii.gz")
lbl_img_data = img.get_fdata()
lbl_img_data.shape

#%% mask
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    # fig.tight_layout()
    for i, slice in enumerate(slices):
        str_0 = "background_slice_" + str(i)

        axes[i].imshow(str_0.T, origin="lower")
        axes[i].imshow(slice.T,  origin="lower")
        axes[i].axis('off')
#
# slice_0 = lbl_img_data[:, 75 , :]
# slice_1 = lbl_img_data[:, 110, :]
# slice_2 = lbl_img_data[:, 125, :]

background_slice_0 = epi_img_data[:, 75 , :]
background_slice_1 = epi_img_data[:, 110, :]
background_slice_2 = epi_img_data[:, 125, :]

show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()









#%%

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="Greys_r", origin="lower")


slice_0 = epi_img_data[60, : , :]
slice_1 = epi_img_data[100, :, :]
slice_2 = epi_img_data[115, :, :]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
plt.show()

