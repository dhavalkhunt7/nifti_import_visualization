#%%
from pathlib import Path
import pandas as pd
from utilities.confusionMatrix_dependent_functions import *


#%% print current working directory
import os
print(os.getcwd())

#%% 705

data_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data")

#%% 705
rs_path = data_path / "Task705_sampling_threshold"
dict_705 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

for i in segmentation_path.glob("*.nii.gz"):
    print(i.name)

#%%
calc_stats(gt_path, segmentation_path, dict_705)

#%% dict to df 705
df_705 = pd.DataFrame.from_dict(dict_705, orient='index')

#%%  mean of df_705 dice
np.mean(df_705['dice'])


#%% save to csv 705
df_705.to_csv(str(rs_path) + "/705.csv")


#%% 706
rs_path = data_path / "Task706_sampling_threshold"
dict_706 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

#%%
calc_stats(segmentation_path, gt_path, dict_706)

#%% dict to df 706
df_706 = pd.DataFrame.from_dict(dict_706, orient='index')

#%% save to csv 706
df_706.to_csv(str(rs_path) + "/706.csv")

#%% 707
rs_path = data_path / "Task707_sampling_threshold"
dict_707 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

#%%
calc_stats(segmentation_path, gt_path, dict_707)

#%% dict to df 707
df_707 = pd.DataFrame.from_dict(dict_707, orient='index')

#%% save to csv 707
df_707.to_csv(str(rs_path) + "/707.csv")

#%% 708
rs_path = data_path / "Task708_sampling_threshold"
dict_708 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

#%%
calc_stats(segmentation_path, gt_path, dict_708)

#%% dict to df 708
df_708 = pd.DataFrame.from_dict(dict_708, orient='index')

#%% save to csv 708
df_708.to_csv(str(rs_path) + "/708.csv")

#%% 709
rs_path = data_path / "Task709_sampling_threshold"
dict_709 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

#%%
calc_stats(segmentation_path, gt_path, dict_709)

#%% dict to df 709
df_709 = pd.DataFrame.from_dict(dict_709, orient='index')

#%% save to csv 709
df_709.to_csv(str(rs_path) + "/709.csv")

#%% 710
rs_path = data_path / "Task710_sampling_threshold"
dict_710 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

#%%
calc_stats(segmentation_path, gt_path, dict_710)

#%% dict to df 710
df_710 = pd.DataFrame.from_dict(dict_710, orient='index')

#%% save to csv 710
df_710.to_csv(str(rs_path) + "/710.csv")

#%% 711
rs_path = data_path / "Task711_sampling_threshold"
dict_711 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_711)

#%% dict to df 711 & save to csv 711
df_711 = pd.DataFrame.from_dict(dict_711, orient='index')

df_711.to_csv(str(rs_path) + "/711.csv")

#%% 712
rs_path = data_path / "Task712_sampling_threshold"
dict_712 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_712)

#%% dict to df 712 & save to csv 712
df_712 = pd.DataFrame.from_dict(dict_712, orient='index')

df_712.to_csv(str(rs_path) + "/712.csv")

#%% 713
rs_path = data_path / "Task713_sampling_threshold"
dict_713 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_713)

#%% dict to df 713 & save to csv 713
df_713 = pd.DataFrame.from_dict(dict_713, orient='index')

#%%
df_713.to_csv(str(rs_path) + "/713.csv")

#%%
np.mean(df_713['dice'])

#%% 714
rs_path = data_path / "Task714_sampling_threshold"
dict_714 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_714)

#%% dict to df 714 & save to csv 714
df_714 = pd.DataFrame.from_dict(dict_714, orient='index')

df_714.to_csv(str(rs_path) + "/714.csv")

#%% 715
rs_path = data_path / "Task715_sampling_threshold"
dict_715 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_715)

#%% dict to df 715 & save to csv 715
df_715 = pd.DataFrame.from_dict(dict_715, orient='index')

#%%
np.mean(df_715['dice'])

#%%
df_715.to_csv(str(rs_path) + "/715.csv")

#%% 716
rs_path = data_path / "Task716_sampling_threshold"
dict_716 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_716)

#%% dict to df 716 & save to csv 716
df_716 = pd.DataFrame.from_dict(dict_716, orient='index')

df_716.to_csv(str(rs_path) + "/716.csv")

#%%
np.mean(df_716['dice'])

#%% 717
rs_path = data_path / "Task717_sampling_threshold"
dict_717 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_717)

#%% dict to df 717 & save to csv 717
df_717 = pd.DataFrame.from_dict(dict_717, orient='index')

df_717.to_csv(str(rs_path) + "/717.csv")

#%%
np.mean(df_717['dice'])

#%% 801
rs_path = data_path / "Task801_control_training"
dict_801 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_801)

#%% dict to df 801 & save to csv 801
df_801 = pd.DataFrame.from_dict(dict_801, orient='index')

#%%

df_801.to_csv(str(rs_path) + "/801.csv")

#%% 802
rs_path = data_path / "Task802_therapy_training"
dict_802 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_802)

#%% dict to df 802 & save to csv 802
df_802 = pd.DataFrame.from_dict(dict_802, orient='index')

df_802.to_csv(str(rs_path) + "/802.csv")


#%% GMM stats 24h
data_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data/")


rs_path = data_path / "Task610_rat/testing/gmm_files/24h"
dict_gmm_24h = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

for i in segmentation_path.glob("*"):
    print(i)

calc_stats(segmentation_path, gt_path, dict_gmm_24h)

#%% dict to df gmm 24h & save to csv gmm 24h
df_gmm_24h = pd.DataFrame.from_dict(dict_gmm_24h, orient='index')

#%% remove .nii from index
df_gmm_24h.index = df_gmm_24h.index.str.replace('.nii', '')


#%%
dict = {}
for i in df_gmm_24h.index:
    print(i)
    str = i
    #if str is in index_list, add to dict
    if str in index_list:
        dict[i] = df_gmm_24h.loc[i]

#%% dict to df
df_gmm_same_testing = pd.DataFrame.from_dict(dict, orient='index')
#%%
df_gmm_same_testing.to_csv("../nnUNet_raw_data_base/nnUNet_raw_data/Task610_rat/testing/gmm_files/24h/gmm_24h_with_same_testing_data.csv")

#%% GMM stats 72h
rs_path = data_path / "Task610_rat/testing/gmm_files/72h"
dict_gmm_72h = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_gmm_72h)

#%% dict to df gmm 72h & save to csv gmm 72h
df_gmm_72h = pd.DataFrame.from_dict(dict_gmm_72h, orient='index')

df_gmm_72h.to_csv(str(rs_path) + "/gmm_72h.csv")

#%% GMM stats 1w
rs_path = data_path / "Task610_rat/testing/gmm_files/1w"
dict_gmm_1w = {}

segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_gmm_1w)

#%% dict to df gmm 1w & save to csv gmm 1w
df_gmm_1w = pd.DataFrame.from_dict(dict_gmm_1w, orient='index')

df_gmm_1w.to_csv(str(rs_path) + "/gmm_1w.csv")

#%% GMM stats 1m
rs_path = data_path / "Task610_rat/testing/gmm_files/1m"
dict_gmm_1m = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_gmm_1m)

#%% dict to df gmm 1m & save to csv gmm 1m
df_gmm_1m = pd.DataFrame.from_dict(dict_gmm_1m, orient='index')

df_gmm_1m.to_csv(str(rs_path) + "/gmm_1m.csv")

#%%
dict_717 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_717)

#%% dict to df 717 & save to csv 717
df_717 = pd.DataFrame.from_dict(dict_717, orient='index')

df_717.to_csv(str(rs_path) + "/717.csv")

#%%
np.mean(df_717['dice'])


#%% 725
rs_path = data_path / "Task725_sampling_threshold"
dict_725 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_725)

#%% dict to df 725 & save to csv 725
df_725 = pd.DataFrame.from_dict(dict_725, orient='index')

df_725.to_csv(str(rs_path) + "/725.csv")

#%%
np.mean(df_725['dice'])

#%% 720
rs_path = data_path / "Task720_sampling_threshold"
dict_720 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_720)

#%% dict to df 720 & save to csv 720
df_720 = pd.DataFrame.from_dict(dict_720, orient='index')

df_720.to_csv(str(rs_path) + "/720.csv")

#%% 651_rat
rs_path = data_path / "Task651_rat"

for i in rs_path.iterdir():
    print(i)
dict_651 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_651)

#%% dict to df 651 & save to csv 651
df_651 = pd.DataFrame.from_dict(dict_651, orient='index')

df_651.to_csv(str(rs_path) + "/651_twersky.csv")

#%%
np.mean(df_651['tversky'])

#%% 650_patch
rs_path = data_path / "Task650_patch"
dict_650 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"

calc_stats(segmentation_path, gt_path, dict_650)

#%% dict to df 650 & save to csv 650
df_650 = pd.DataFrame.from_dict(dict_650, orient='index')

df_650.to_csv(str(rs_path) + "/650.csv")

#%%
np.mean(df_650['dice'])

#%% 651_patch
rs_path = data_path / "Task651_patch"
dict_651 = {}
segmentation_path = rs_path / "result"
gt_path = rs_path / "labelsTs"
