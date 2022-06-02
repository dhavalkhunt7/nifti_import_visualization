# %%
from __future__ import print_function

from pathlib import Path

import SimpleITK as sitk
import nibabel as nb
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

main_img_dir_ = "../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/" \
                "images_tr_converted/BRATS_1020_0002.nii.gz"
lbl_img_dir_ = "../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/" \
               "labels/BRATS_1020.nii.gz"

imageInput = sitk.ReadImage(main_img_dir_)
labelInput = sitk.ReadImage(lbl_img_dir_)


# plt.imshow(imageInput)
# plt.show()

# %%

def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3, 4):
            nda = nda[nda.shape[0] // 2, :, :]

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3, 4):
            # raise Runtime("Unable to show 3D-vector Image")
            print("Unable to show 3D-vector Image")

        # take a z-slice
        nda = nda[nda.shape[0] // 2, :, :, :]

    ysize = nda.shape[0]
    xsize = nda.shape[1]

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

    t = ax.imshow(nda, extent=extent, interpolation=None)

    if nda.ndim == 2:
        t.set_cmap("gray")

    if (title):
        plt.title(title)
    # plt.axis('off')
    plt.savefig("plotting2.png")


# %%
# myshow(sitk.LabelToRGB(labelInput))

labelInput.CopyInformation(imageInput)
# %% not working because of 3d file.. perfectly works if data is 2d format
# myshow(sitk.LabelOverlay(imageInput, sitk.LabelContour(labelInput), 1.0))

# %%
arrayInput = sitk.GetArrayFromImage(imageInput)
arrayspacing = imageInput.GetSpacing()

if arrayInput.ndim == 3:
    print(arrayInput.ndim)

# %%
img_T1 = imageInput
img_labels = labelInput
# %%
myshow(img_T1)
# myshow(img_labels)

# %%

size = img_T1.GetSize()
myshow(img_T1[:, size[1] // 2, :])
print(size)

# %%
slices = [img_T1[size[0] // 2, :, :], img_T1[:, size[1] // 2, :], img_T1[:, :, size[2] // 2]]
myshow(sitk.Tile(slices, [3, 1]), dpi=20)

# %%
nslices = 5
slices = [img_T1[:, :, s] for s in range(0, size[2], size[0] // (nslices + 1))]
myshow(sitk.Tile(slices, [1, 0]))


# %%
def myshow3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    size = img.GetSize()
    img_xslices = [img[s, :, :] for s in xslices]
    img_yslices = [img[:, s, :] for s in yslices]
    img_zslices = [img[:, :, s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    img_null = sitk.Image([0, 0], img.GetPixelIDValue(), img.GetNumberOfComponentsPerPixel())

    img_slices = []
    d = 0

    if len(img_xslices):
        img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
        d += 1

    if len(img_yslices):
        img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
        d += 1

    if len(img_zslices):
        img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
        d += 1

    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen, d])
        # TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0, img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
            img = sitk.Compose(img_comps)

    myshow(img, title, margin, dpi)


# %%
myshow3d(img_T1, yslices=range(50, size[1] - 50, 20), zslices=range(50, size[2] - 50, 20), dpi=800)

# %%
myshow3d(img_T1, yslices=range(50, size[1] - 50, 30), zslices=range(50, size[2] - 50, 20), dpi=30)

# %%
myshow3d(sitk.LabelToRGB(img_labels), yslices=range(50, size[1] - 50, 20), zslices=range(50, size[2] - 50, 20), dpi=30)

# %%
myshow3d(sitk.LabelOverlay(img_T1, sitk.LabelContour(img_labels)), yslices=range(50, size[1] - 50, 20),
         zslices=range(50, size[2] - 50, 20), dpi=30)
# %%
print(img_labels.CopyInformation(img_T1))


# %%
def plot3d(img, xslices=[], yslices=[], zslices=[]):
    size = img.GetSize()
    print(size)
    img_xslices = [img[s, :, :] for s in xslices]
    img_yslices = [img[:, s, :] for s in yslices]
    img_zslices = [img[:, :, s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    img_null = sitk.Image([0, 0], img.GetPixelIDValue(), img.GetNumberOfComponentsPerPixel())

    img_slices = []
    d = 0

    if len(img_xslices):
        img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
        d += 1

    if len(img_yslices):
        img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
        d += 1

    if len(img_zslices):
        img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
        d += 1

    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen, d])
        # TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0, img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
            img = sitk.Compose(img_comps)

    plotfig(img)


# %%


def plotfig(img):
    size = img.GetSize()
    print(size)

    slices_list = [75, 110, 125]  # for x and y
    slices_z = [38, 50, 70]
    img_xslices = [img[s, :, :] for s in slices_list]
    img_yslices = [img[:, s, :] for s in slices_list]
    img_zslices = [img[:, :, s] for s in slices_z]

    # maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    print(img_zslices[2].ndim)

    fig, axis = plt.subplots(1, 3, figsize=(11, 6))

    # plt.figure()
    # for i in range(3):
    #     axis[i].imshow(img_yslices[i], origin="lower", interpolation='none')
    #     axis[i].axis('off')
    # j += 1
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    #

    # t = plt.imshow(nda, interpolation=None)
    # plt.savefig("plotting2.png")


plotfig(img_T1)

# %%

import numpy as np

img = img_T1
slices_list = [75, 110, 125]  # for x and y
slices_z = [38, 50, 70]
img_xslices = [img[s, :, :] for s in slices_list]
img_yslices = [img[:, s, :] for s in slices_list]
img_zslices = [img[:, :, s] for s in slices_z]

print(np.array(img_yslices[1]).ndim)


def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False


# valid_imshow_data(img_yslices[1])

# %%
img_T2 = img_T1
img_T2.SetPixel(0, 0, 0, 1)
print(img_T2.GetPixel(0, 0, 0))

# %%
plot3d(sitk.LabelOverlay(img_T1, sitk.LabelContour(img_labels)), yslices=range(50, size[1] - 50, 20),
       zslices=range(50, size[2] - 50, 20))


#%%

# %%
1

# %%

def plotfig(img):
    # size = img.GetSize()
    # print(size)

    slices_list = [75, 110, 125]  # for x and y
    slices_z = [38, 50, 70]
    img_xslices = [img[s, :, :] for s in slices_list]
    img_yslices = [img[:, s, :] for s in slices_list]
    img_zslices = [img[:, :, s] for s in slices_z]

    # maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    print(img_zslices[2].ndim)

    for i in img_xslices:
        # print(i)
        img = sitk.ReadArray(i)
    # plt.figure()
    # for i in range(3):
    #     plt.imshow(img_yslices[i], origin="lower", interpolation='none')
    #     plt.axis('off')
    #     j += 1
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    #

    # t = plt.imshow(nda, interpolation=None)
    # plt.savefig("plotting2.png")


plotfig(epi_img_data)
