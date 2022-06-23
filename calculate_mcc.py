# %%
from pathlib import Path
import nibabel as nb
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score

# %%
target_base = Path("../nnUNet_raw_data_base/nnUNet_raw_data/Task605_rat")

gt_dir = target_base / "labelsTs"
pred_dir = target_base / "resultTs"

# %%

for i in gt_dir.glob("Rat102.nii.gz"):
    print(i.name)
    gt_img = nb.load(i).get_fdata()

# %%
for j in pred_dir.glob("Rat102.nii.gz"):
    prediction_img = nb.load(j).get_fdata()
    print(j)
# %%
pred_img = prediction_img

# %%
gt_vector = gt_img.flatten()
pred_vector = pred_img.flatten()

# %% calculate mcc using matthews_corrcoef
matthews_corrcoef(gt_vector, pred_vector)


# %%
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return TP, FP, TN, FN


# %% calculate by calling perf_measure function TP, TN, FP, FN
TP, FP, TN, FN = perf_measure(gt_vector, pred_vector)


# %% define function to calculate accuracy, precision, recall, f1score, sensitivity, specificity,
# false omission rate
def calculate_all_terms(TP, FP, TN, FN):
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1score = 2 * (precision * recall) / (precision + recall)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    false_omission_rate = FN / (FN + TN)

    return accuracy, precision, recall, f1score, sensitivity, specificity, false_omission_rate


# %%
accuracy, precision, recall, f1score, sensitivity, specificity, false_omission_rate = calculate_all_terms(TP, FP, TN,
                                                                                                          FN)

#%%
precision

# %% define function to calculate  , Area under Curve
def calculate_auc(gt_vector, pred_vector):
    auc = roc_auc_score(gt_vector, pred_vector)

    return auc
