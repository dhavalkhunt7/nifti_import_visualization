#%%
from pathlib import Path
import numpy as np
import nibabel as nib

from utilities.dice import calc_DSC_Enhanced, calc_DSC_Sets, calc_DSC_CM
from utilities.mism import calc_MISm
from utilities.accuracy import calc_Accuracy_CM

#%%

data_path = Path("../../nnUNet_raw_data_base/nnUNet_raw_data/Task645_Patch_2d_Rat24h/Human_Patch_testing")
labels_rat = data_path / "labels_reconstructed"
pred_rat = data_path / "result_reconstructed"


#%%
# dictionary to store the results
dice_dict = {}
for i in labels_rat.glob("*"):
    print(i.name)
    name = i.name

    label_data = nib.load(i).get_fdata()
    pred_data = nib.load(pred_rat / name).get_fdata()

    # flatten data
    label_data = label_data.flatten()
    pred_data = pred_data.flatten()

    # calculate dice
    normal_dice = calc_DSC_Sets(label_data, pred_data, c=1)
    print("normal_dice: ", normal_dice)
    mism_dice = calc_MISm(label_data, pred_data, c=1)
    print("mism_dice: ", mism_dice)
    # add the results to the dictionary with
    dice_dict[name] = {"normal_dice": normal_dice, "mism_dice": mism_dice}


#%% dice to DataFrame
import pandas as pd
dice_df = pd.DataFrame.from_dict(dice_dict, orient="index")

#%% export to csv
dice_df.to_csv("dice_human_645.csv")

#%% print current working directory
import os
print(os.getcwd())


# #%%
#
# data_path = Path("../../nnUNet_raw_data_base/nnUNet_raw_data/Task646_combined_patch/")
# labels_rat = data_path / "labels_reconstructed"
# pred_rat_2d = data_path / "result_2d_reconstructed"
# pred_rat_3d = data_path / "result_3d_reconstructed"
#
# #%%
# # dictionary to store the results
# human_dict = {}
# rat_dict = {}
# for i in labels_rat.glob("*"):
#     print(i.name)
#     name = i.name
#
#     label_data = nib.load(i).get_fdata()
#     pred_data = nib.load(pred_rat_2d / name).get_fdata()
#
#     # flatten data
#     label_data = label_data.flatten()
#     pred_data = pred_data.flatten()
#
#     # calculate dice
#     normal_dice = calc_DSC_Sets(label_data, pred_data, c=1)
#     print("normal_dice: ", normal_dice)
#     mism_dice = calc_MISm(label_data, pred_data, c=1)
#     print("mism_dice: ", mism_dice)
#     # if name starts with rat then add to rat dictionary
#     if name.startswith("Rat"):
#         # print("rat")
#         rat_dict[name] ={"normal_dice": normal_dice, "mism_dice": mism_dice}
#     # if name starts with human then add to human dictionary
#     elif name.startswith("Human"):
#         # print("human")
#         human_dict[name] ={"normal_dice": normal_dice, "mism_dice": mism_dice}
#
# #%% dice to DataFrame
# import pandas as pd
# human_df = pd.DataFrame(human_dict)
# rat_df = pd.DataFrame(rat_dict)
#
# #%% export to csv
# human_df.to_csv("human_dice_646.csv")
# rat_df.to_csv("rat_dice_646.csv")
#
# #%% print current working directory
# import os
# print(os.getcwd())