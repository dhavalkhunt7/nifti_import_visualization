#%%
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
# from segmentation_mask_overlay import overlay_masks
import copy
import numpy as np


#%%


database = Path("../nnUNet_raw_data_base/nnUNet_raw_data/")

main_name  = "Task610_rat"
img_dir = database / main_name / "imagesTs"
# label_dir = database / Task_name / "resultTs"


img_path = img_dir / "Rat086_0001.nii.gz"
img_data = nib.load(img_path).get_fdata()

#%%
label_db = database / "segmentation_results_adrian/final_segmentation_Task6**"
Task_name = "Task625"

label_dir = label_db / Task_name / "2d_best"


#%%
label_data = nib.load(label_dir / "Rat086.nii").get_fdata()

print(label_data.shape)

#%% rotate img data
img_data_rot = np.rot90(img_data, 1)
label_data_rot = np.rot90(label_data, 1)

print(type(label_data_rot))

#%%
# #%%plot 10 splits of the image
# s=5
# f, axarr = plt.subplots(1,10)
# for i in range(10):
#     axarr[1,i].imshow(img_data_rot[:, :, s], cmap='gray')
#     # axarr[i,0].imshow(label_data_rot[:, :, s], cmap='jet', alpha=0.5)
#     s+=12
# plt.show()

# #%% plot an image using matplotlib
# plt.imshow(img_data_rot[:, :, 78], cmap='gray_r')
# plt.imshow(label_data_rot[:, :, 78], cmap='gray', alpha=0.5)
# plt.show()


# #%%
# fig, ax = plt.subplots(1, 7, figsize=(150, 50))
#
# n = 0
# slice = 40
# for _ in range(7):
#     ax[n].imshow(img_data_rot[:, :, slice], 'gray')
#     ax[n].set_xticks([])
#     ax[n].set_yticks([])
#     ax[n].set_title('Slice {}'.format(slice), color='r', fontsize=5)
#     n += 1
#     slice += 10
#
# fig.subplots_adjust(wspace=0, hspace=0)
# # plt.show()


#%%
# [Example] Load image. If you are sure of you masks
# image = Image.open("cat.jpg").convert("L")
# image = img_data_rot


# [Example] Mimic list of masks
# masks = []
#
# overlay_masks(image[0], [label_data_rot[0].astype(bool)], mask_alpha=1)
#
# image = image.transpose((2, 0, 1))
# label_data_rot = label_data_rot.transpose((2, 0, 1))
#
# masks.append(label_data_rot)
#
# # Laminate your image!
# fig = [overlay_masks(i, [m], mask_alpha=1, mpl_colormap="Accent") for i,m in zip(image, label_data_rot.astype(bool))]
#
# # Do with that image whatever you want to do.
# fig[80].savefig("at_masked.png", bbox_inches="tight", dpi=300)
#
#
#


#%%
# fig, ax = plt.subplots(1, 7)
#
# image = img_data_rot
#
# masks = []
#
# overlay_masks(image[0], [label_data_rot[0].astype(bool)], mask_alpha=1)
#
# image = image.transpose((2, 0, 1))
# label_data_rot = label_data_rot.transpose((2, 0, 1))
#
# masks.append(label_data_rot)
#
#
# n = 0
# slice = 40
# for _ in range(7):
#     fig = [overlay_masks(i, [m], mask_alpha=1, mpl_colormap="Accent") for i, m in
#            zip(image, label_data_rot.astype(bool))]
#     ax[n].imshow(fig[slice])
#     # ax[n].imshow(img_data_rot[:, :, slice], 'gray')
#     ax[n].set_xticks([])
#     ax[n].set_yticks([])
#     ax[n].set_title('Slice {}'.format(slice), color='r', fontsize=5)
#     n += 1
#     slice += 10
#
# fig.subplots_adjust(wspace=0, hspace=0)
# # plt.show()
#
# plt.savefig("1_masked.png", bbox_inches="tight", dpi=300)




#%%
# image = img_data_rot
#
#
# # [Example] Mimic list of masks
# masks = []
#
# overlay_masks(image[0], [label_data_rot[0].astype(bool)], mask_alpha=1)
#
# image = image.transpose((2, 0, 1))
# label_data_rot = label_data_rot.transpose((2, 0, 1))
#
# masks.append(label_data_rot)
#
# # Laminate your image!
# fig = [overlay_masks(i, [m], mask_alpha=1, mpl_colormap="Accent") for i,m in zip(image, label_data_rot.astype(bool))]

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
plt_5 = plot_subplots(img_data_rot, label_data_rot, 1, 7)

#%% save the figure
plt_5.savefig("segmentation_plot/625_2d_segmentation_7.pdf", bbox_inches="tight", dpi=300)


#%%
def show_image():

    my_cmap = copy.copy(plt.cm.get_cmap('gray'))  # get a copy of the gray color map
    my_cmap.set_bad(alpha=0)

    plt.imshow(img_data_rot[:, :, 90], cmap='gray')
    plt.imshow(label_data_rot[:, :, 90], cmap='Greens', alpha=0.7)
    # plt.colorbar()
    #hidw border of plot
    plt.axis('off')
    # plt.set_visible(False)
    return plt

#%%
plt1 = show_image()

#%%
plt1.savefig("222.png", bbox_inches="tight", dpi=300)
plt1.show()