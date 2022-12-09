#%%
from pathlib import Path
import numpy as np
import nibabel as nib

from utilities.dice import calc_DSC_Enhanced, calc_DSC_Sets, calc_DSC_CM
from utilities.mism import calc_MISm
from utilities.accuracy import calc_Accuracy_CM

#%%

data_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task601_human")
labels_human = data_path / "all_about_testing"
pred_human = data_path / "result_3d"

#%%
dice_dict = {}
for i in labels_human.glob("*"):
    print(i.name)
    new_dir = i
    for j in new_dir.glob("GroundTrouth.nii.gz"):
        print(j.name)

        label_data = nib.load(j).get_fdata()
        name = i.name + "1.nii.gz"
        pred_data = nib.load(pred_human / name).get_fdata()
        print(label_data.shape)
        print(pred_data.shape)

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
dice_df.to_csv("dice_human_601.csv")

#%% print current working directory
import os
print(os.getcwd())


#%% import csv file
import pandas as pd
dice_df = pd.read_csv("dice_human_601.csv", index_col=0)


#%% mean and std of dice and normalized dice
print("mean of dice: ", dice_df["normal_dice"].mean())
print("std of dice: ", dice_df["normal_dice"].std())
print("mean of normalized dice: ", dice_df["mism_dice"].mean())
print("std of normalized dice: ", dice_df["mism_dice"].std())

#%% box plot normal dice vs normalized dice side by side
import matplotlib.pyplot as plt

dice_df.boxplot(column=["normal_dice", "mism_dice"], grid=False)
plt.show()
# save the figure to a file


#%% box plot df.boxplot(column=["normal_dice", "mism_dice"], grid=False) with range of y axis from 0.5 to 1
import matplotlib.pyplot as plt

dice_df.boxplot(column=["normal_dice", "mism_dice"], grid=False)
plt.ylim(0.5,)
plt.figsize(10,10)
plt.show()

#%%
plt.savefig("all_for_dgn/boxplot_human_601.png")


