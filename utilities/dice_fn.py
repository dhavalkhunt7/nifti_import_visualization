import numpy as np



# %%
# def dice(prediction_array, label_array):
#     alpha = 0.1
#     # flatten the array
#     prediction_array = prediction_array.flatten()
#     label_array = label_array.flatten()
#
#     # find TP, FP, TN, FN
#     TP = np.sum(prediction_array[label_array == 1])
#     FP = np.sum(prediction_array[label_array == 0])
#     TN = np.sum(prediction_array[label_array == 0])
#     FN = np.sum(prediction_array[label_array == 1])
#
#     # calculate dice coefficient
#     if len(np.unique(label_array)) == 1:
#         dice = (alpha * TN) / ((1 - alpha) * FP + alpha * TN)
#         return 0
#     else:
#         # calculate dice coefficient
#         dice = (2 * TP) / (2 * TP + FP + FN)
#         return dice
#
#     return dice


