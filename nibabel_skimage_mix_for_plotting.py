# %%
from pathlib import Path
import nibabel as nb
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

img_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing")
original_data = img_dir / "images_tr_converted"
labels = img_dir / "labels"
predicted_dir = img_dir / "prediction_files_for_dc"

# %%

img = nb.load(original_data / "BRATS_1020_0002.nii.gz")
epi_img_data = img.get_fdata()
epi_img_data.shape

label_img = nb.load(labels / "BRATS_1020.nii.gz")
lbl_img_data = label_img.get_fdata()
lbl_img_data.shape

# %%

img = epi_img_data
slices_list = [75, 110, 125]  # for x and y
slices_z = [38, 50, 70]
img_xslices = [img[s, :, :] for s in slices_list]
img_yslices = [img[:, s, :] for s in slices_list]
img_zslices = [img[:, :, s] for s in slices_z]

# %%
store_dir = Path("nifti_store/xlabels")

a = 1
for i in img_xslices:
    f_img = nb.Nifti1Image(i, np.eye(4))
    new_name = str(a) + "_x_slices.nii.gz"
    # print(f_img)
    nb.save(f_img, str(store_dir) + "/" + new_name)
    a += 1

#%%
main_img_dir_ = "nifti_store/xlabels/"
list_0 = []

for i in range(3):
    img_dir = main_img_dir_ + str(i +1) + "_x_slices.nii.gz"
    image = sitk.ReadImage(img_dir)
    list_0.append(image)

#%%
img_T1 = list_0[0]
img_T1.GetSize()

#%%
def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayFromImage(img)


    t = plt.imshow(nda)


    # plt.axis('off')
    plt.show()
    # plt.savefig("plotting2.png")

myshow(img_T1)
#%%
# print(img_T1)
nda = sitk.GetArrayFromImage(img_T1)
nda.min()
plt.imshow(nda)
plt.show()




















