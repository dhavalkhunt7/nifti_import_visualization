# %%
from pathlib import Path

# import cv2
import nibabel as nib
import numpy as np
import patchify as patchify

# %%
print("hellp")

# %% input christine_theranostics_data_folder


database_input = Path("input/Masked_ADC.nii")

# for i in database_input.glob("*.nii"):
#     print(i)


img = nib.load(database_input)

a = np.array(img.dataobj)
print(a.shape)
print(type(a))

# %% import nii as numpy array
#
# control_data = nib.load("input/Masked_ADC.nii").get_fdata()
# print(type(control_data))
# print(control_data.shape)

# %% craete list of 2d slices from 3d data
control_data_list = []
for i in range(a.shape[2]):
    control_data_list.append(a[:, :, i])

# %% array to normal img
print(type(control_data_list[0]))


# %% length of patch list


# with stride

# %% creeate a function to split 2d image array into overlapping tiles
def split_image(image, patch_size, stride):
    patches = patchify.patchify(image, patch_size, step=stride)
    return patches


# %%
list_array = split_image(control_data_list[0], (10, 10), 5)
print(list_array.shape)
print(type(list_array))
# %% plot the first patch
import matplotlib.pyplot as plt

plt.imshow(a[:, :, 74])
plt.show()

# %% split a[:,:,74] into tiles using np.split


height = 10
width = 10
overlap = 5
y_step = height - overlap
x_step = width - overlap
A = a[:, :, 74]
B = [A[y:y + height, x:x + width] for y in range(0, A.shape[0], y_step) for x in range(0, A.shape[1], x_step)]

# %% print the shape of the all the tiles of list
for i in range(len(B)):
    print(B[i].shape)
# print(len(B))
# %% plot the first patch
# print(B[0])
plt.imshow(B[6])
plt.show()

# %%
print(a[:, :, 74].shape)

# %%


img = a[:, :, 74]
img_h, img_w = img.shape
split_width = 10
split_height = 10

list_of_tiles = []


def start_points(size, split_size, overlap=5):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


X_points = start_points(img_w, split_width, 0.5)
Y_points = start_points(img_h, split_height, 0.5)

count = 0
name = 'splitted'
frmt = 'jpeg'

for i in Y_points:
    for j in X_points:
        split = img[i:i + split_height, j:j + split_width]
        list_of_tiles.append(split)
        count += 1

# %%prnt sizer of all patches
for i in range(len(list_of_tiles)):
    print(list_of_tiles[i].shape)
print(len(list_of_tiles))


# %%Here is function, that splits image with overlapping from all sides. On the borders it will be filled with zeros.
#
# What it essentially does: it creates a bigger image with zero padding, and then extract patches of size window_
# size+2*margin with strides of window_size. (You may want to adjust it to your needs)


def split(img, window_size, margin):
    sh = list(img.shape)
    sh[0], sh[1] = sh[0] + margin * 2, sh[1] + margin * 2
    img_ = np.zeros(shape=sh)
    img_[margin:-margin, margin:-margin] = img

    stride = window_size
    step = window_size + 2 * margin

    nrows, ncols = img.shape[0] // window_size, img.shape[1] // window_size
    splitted = []
    for i in range(nrows):
        for j in range(ncols):
            h_start = j * stride
            v_start = i * stride
            cropped = img_[v_start:v_start + step, h_start:h_start + step]
            splitted.append(cropped)
    return splitted


# %%
split_tiles = split(a[:, :, 74], 10, 5)
#%%
for i in range(len(split_tiles)):
    print(split_tiles[i].shape)
print(len(split_tiles))

#%% plot split_tiles
import matplotlib.pyplot as plt

for i in range(len(split_tiles)):
    plt.imshow(split_tiles[i])
    plt.show()