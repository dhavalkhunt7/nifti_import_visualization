import nibabel as nb
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from nilearn import plotting, image

# directy path
# from lables_conversion import new_name


test_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task501_BrainTumour/imagesTs")
predicted_dir = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Result_502")
plot_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Plot_results_502/")
# dict = {1: test_dir("*.nii.gz"), 2: predicted_dir("*.nii.gz")}


# %%
for i in test_dir.glob("*.nii.gz"):
    test_img = i.name
    print(test_img)
    for j in predicted_dir.glob("*.nii.gz"):
        pre_img = j.name
        if pre_img == test_img:
            first_volume = image.index_img(test_img, 1)

            plotting.plot_roi(pre_img, bg_img=first_volume, cmap='Paired', colorbar=True)

            new_name = str(pre_img).replace("nii.gz", "png")
            plt.savefig(new_name, bbox_inches='tight', path=plot_path)
            print(new_name + " saved ")
