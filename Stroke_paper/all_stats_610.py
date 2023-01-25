#%%
from utilities.confusion_matrix import calc_ConfusionMatrix
from utilities.confusionMatrix_dependent_functions import *
import numpy as np
import shutil
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
from utilities.confusionMatrix_dependent_functions import *
import os


#%% 605 24h

data_path = Path("../../nnUNet_raw_data_base/nnUNet_raw_data/Task610_rat")

#%%
dict_24h = {}
segmentation_path = data_path / "resultTs"
gt_path = data_path / "labelsTs"

#%%
calc_stats(segmentation_path, gt_path, dict_24h)

#%% dict to df
df_24h = pd.DataFrame.from_dict(dict_24h, orient='index')

#%% save to csv
df_24h.to_csv(str(data_path) + "/24h.csv")

#%% 605 24h 3d
segmentation_path = data_path / "resultTs_3d"
gt_path = data_path / "labelsTs"

#%%
dict_24h_3d = {}
calc_stats(segmentation_path, gt_path, dict_24h_3d)

#%% dict to df
df_24h_3d = pd.DataFrame.from_dict(dict_24h_3d, orient='index')

df_24h_3d.to_csv(str(data_path) + "/24h_3d.csv")

#%%
# ------------------------------605---------------------------------#
data_path = Path("../../nnUNet_raw_data_base/nnUNet_raw_data/Task610_rat/testing")
data_72h = data_path / "72h"
data_1w = data_path / "1w"
data_1m = data_path / "1m"


#%%
dict_72h = {}
segmentation_path = data_72h / "result"
ground_truth_path = data_72h / "labelsTs"
calc_stats(ground_truth_path, segmentation_path, dict_72h)

#%% dict to df
df_72h = pd.DataFrame.from_dict(dict_72h, orient='index')
# save df to csv

df_72h.to_csv(str(data_72h) +"/2d_72h.csv")

#%% 72h 3d
dict_72h_3d = {}
segmentation_path = data_72h / "result_3d"
ground_truth_path = data_72h / "labelsTs"
calc_stats(ground_truth_path, segmentation_path, dict_72h_3d)


#%% dict to df
df_72h_3d = pd.DataFrame.from_dict(dict_72h_3d, orient='index')

# save df to csv
df_72h_3d.to_csv(str(data_72h) +"/3d_72h.csv")


#%% 1w 2d
dict_1w = {}
segmentation_path = data_1w / "result"
ground_truth_path = data_1w / "labelsTs"
calc_stats(ground_truth_path, segmentation_path, dict_1w)

#%% dict to df
df_1w = pd.DataFrame.from_dict(dict_1w, orient='index')

# save df to csv
df_1w.to_csv(str(data_1w) +"/2d_1w.csv")

#%% 1w 3d
dict_1w_3d = {}
segmentation_path = data_1w / "result_3d"
ground_truth_path = data_1w / "labelsTs"
calc_stats(ground_truth_path, segmentation_path, dict_1w_3d)

#%% dict to df
df_1w_3d = pd.DataFrame.from_dict(dict_1w_3d, orient='index')

# save df to csv
df_1w_3d.to_csv(str(data_1w) +"/3d_1w.csv")


#%% 1m 2d
dict_1m = {}
segmentation_path = data_1m / "result"
ground_truth_path = data_1m / "labelsTs"
calc_stats(ground_truth_path, segmentation_path, dict_1m)

#%% dict to df
df_1m = pd.DataFrame.from_dict(dict_1m, orient='index')

# save df to csv
df_1m.to_csv(str(data_1m) +"/2d_1m.csv")

#%% 1m 3d
dict_1m_3d = {}
segmentation_path = data_1m / "result_3d"
ground_truth_path = data_1m / "labelsTs"
calc_stats(ground_truth_path, segmentation_path, dict_1m_3d)

#%% dict to df
df_1m_3d = pd.DataFrame.from_dict(dict_1m_3d, orient='index')

# save df to csv
df_1m_3d.to_csv(str(data_1m) +"/3d_1m.csv")


#%%
data_path = Path("../../../../Documents/data/adrian_data/Data_Paper_12092022")
gmm_1w = data_path / "GMM/GMM_1w/Niftis"
gmm_24h = data_path / "GMM/GMM_24h/Niftis"

#%%
dict_gmm_1w = {}
calc_stats3(gmm_1w, "Voi_1w.nii", gmm_24h, "RF_Probmaps1.nii", dict_gmm_1w)

#%% dict to df and save to csv
df_gmm_1w = pd.DataFrame.from_dict(dict_gmm_1w, orient='index')
df_gmm_1w.to_csv(str(data_path) + "/gmm_1w_voi_24h_mask.csv")

#%%
dict_gmm1w = {}
calc_stats2(gmm_1w, "Voi_1w.nii", "RF_Probmaps1.nii", dict_gmm1w)

#%% dict to df
df_gmm1w = pd.DataFrame.from_dict(dict_gmm1w, orient='index')

#%% save to csv
df_gmm1w.to_csv(str(gmm_1w) + "/GMM_1w.csv")

#%%
dict_gmm24h = {}
calc_stats2(gmm_24h, "Voi_24h.nii", "RF_Probmaps1.nii", dict_gmm24h)

#%% dict to df
df_gmm24h = pd.DataFrame.from_dict(dict_gmm24h, orient='index')

#%% save to csv
df_gmm24h.to_csv(str(gmm_24h) + "/GMM_24h.csv")



#%%
segmentation_path = data_path / "resultTs"
gt_path = data_path / "testing/1w" / "labelsTs"

#%%
dict_24h_seg_1w_mask = {}
# calc_stats3(gmm_1w, "Voi_1w.nii", gmm_24h, "RF_Probmaps1.nii", dict_gmm_1w)
for i in segmentation_path.glob("*.nii.gz"):
    print(i.name)
    # file_name = i.name.replace("-1w", "")
    # # print(file_name)
    #
    # if path exists, then continue
    # new_filename = file_name + "-24h"
    if (gt_path / i.name).exists():
        print(str(gt_path / i.name) + "  path exists")
        seg_file_name = i
        gt_file_name = gt_path / i.name

        segmentation = nib.load(str(seg_file_name)).get_fdata()
        ground_truth = nib.load(str(gt_file_name)).get_fdata()
        print(segmentation.shape)
        print(ground_truth.shape)

        # flatten data
        pred_data = segmentation.flatten()
        label_data = ground_truth.flatten()

        name = i.name.replace(".nii.gz", "")
        # calculate all metrics using function calc_all_metrics_CM
        stats = calc_all_metrics_CM(label_data, pred_data)
        dict_24h_seg_1w_mask[name] = {'mcc': stats[0], 'sens': stats[1], 'spec': stats[2], 'prec': stats[3], 'acc': stats[4],
                        'FDR': stats[5], 'FPR': stats[6], 'PPV': stats[7], 'NPV': stats[8], 'dice': stats[9],
                        'wspec': stats[10], 'tversky': stats[11], 'balanced_acc': stats[12]}

#%% dict to df and save to csv
df = pd.DataFrame.from_dict(dict_24h_seg_1w_mask, orient='index')
df.to_csv(str(data_path) + "/24h_pred_1w_gt.csv")