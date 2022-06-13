# %%
from pathlib import Path
import nibabel as nib
from pyparsing import col
from skimage import color, segmentation, img_as_float
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# %% skimage input
# main_img_dir_ = "../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/" \
#                 "images_tr_converted/BRATS_1020_0002.nii.gz"
# lbl_img_dir_ = "../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/" \
#                "labels/BRATS_1020.nii.gz"
#
# imageInput = sitk.ReadImage(main_img_dir_)
# labelInput = sitk.ReadImage(lbl_img_dir_)

# %%
# pathlib inputs
from skimage.color import gray2rgb, rgb2gray
from skimage.color.colorlabel import _match_label_with_color, _rgb_vector, DEFAULT_COLORS
from skimage.measure import label

img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing")

original_data = img_dir / "images_tr_converted"
labels = img_dir / "labels"
predicted_dir = img_dir / "prediction_files_for_dc"

# %% t2 img data

img = nib.load(original_data / "BRATS_1020_0002.nii.gz")
t2_main_img_data = img.get_fdata()
type(t2_main_img_data)

# %%
img = nib.load(labels / "BRATS_1020.nii.gz")
label_img_data = img.get_fdata()
type(label_img_data)
label_img_data.shape


# %% convert into 3 slices
# define methopd for that
def extract_slices(img, list_name, name):
    list_name = []
    slices_list = [75, 110, 125]  # for x and y
    slices_z = [38, 50, 70]
    img_xslices = [img[s, :, :] for s in slices_list]
    img_yslices = [img[:, s, :] for s in slices_list]
    img_zslices = [img[:, :, s] for s in slices_z]

    # x slices
    for s in slices_list:
        img_slices = img[s, :, :]
        new_name = str(name) + "_x_slices_" + str(slices_list.index(s) + 1)
        new_name = img_slices
        list_name.append(new_name)
        # list_name.append(new_name)

    # y slices
    for s in slices_list:
        img_slices = img[s, :, :]
        new_name = str(name) + "_y_slices_" + str(slices_list.index(s) + 1)
        new_name = img_slices
        list_name.append(new_name)
        # print(new_name)

    # x slices
    for s in slices_z:
        img_slices = img[s, :, :]
        new_name = str(name) + "_z_slices_" + str(slices_z.index(s) + 1)
        new_name = img_slices
        list_name.append(new_name)
        # print(new_name)

    return list_name


# %% main img data
t_slices = extract_slices(t2_main_img_data, "t_slices", "img1")
len(t_slices)

# %%
t_slices[0].ndim

# %% label data
label_slices = extract_slices(label_img_data, "label_slices", "label_img")
len(label_slices)

# %%
plt.imshow(t_slices[0].T, origin="lower")
plt.imshow(color.label2rgb(label_slices[0].T), origin="lower", alpha=0.5)
plt.savefig("plotting4.png")

# %%
N = 3
colours = cm.get_cmap('viridis', N)  # Change the string from 'viridis' to whatever you want from the above link
cmap = colours(np.linspace(0, 1, N))  # Obtain RGB colour map
cmap[0, -1] = 0  # Set alpha for label 0 to be 0
cmap[1:, -1] = 0.3
cmap[2:, -1] = 0.3
cmap[3:, -1] = 0.3

# %%
np.max(cmap)


# %% for normalization od array ... for now not important

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

scaled_x = NormalizeData(label_slices[0].flatten())

# %%
output = cmap[scaled_x.astype(int).flatten()]
output

# %%
R, C = label_slices[0].shape[:2]
output_array = output.reshape((R, C, -1))

output_array

# %%
fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.imshow(t_slices[0], cmap='gray')
ax.imshow(output_array)
fig.show()





#%% Label 2 rgb plotting

# colours = color.get_cmap('viridis', N)  # Change the string from 'viridis' to whatever you want from the above link

#%%
labels1 = segmentation.slic(label_slices[0],  n_segments = 4)
labels1
#%%
plt.imshow(t_slices[0])
plt.imshow(color.label2rgb(label_slices[0], kind='overlay', bg_label=0))
plt.show()
# plt.savefig("plotting3.png")



