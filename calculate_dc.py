# import numpy as np
#
# k=1
#
# # segmentation
# seg = np.zeros((100,100), dtype='int')
# seg[30:70, 30:70] = k
#
# # ground truth
# gt = np.zeros((100,100), dtype='int')
# gt[30:70, 40:80] = k
#
# dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))
#
# print 'Dice similarity score is {}'.format(dice)
#
#
#
# ####
#
#
#
# import cv2
# import numpy as np
#
# #load images
# y_pred = cv2.imread('predictions/image_001.png')
# y_true = cv2.imread('ground_truth/image_001.png')
#
# # Dice similarity function
# def dice(pred, true, k = 1):
#     intersection = np.sum(pred[true==k]) * 2.0
#     dice = intersection / (np.sum(pred) + np.sum(true))
#     return dice
#
# dice_score = dice(y_pred, y_true, k = 255) #255 in my case, can be 1
# print ("Dice Similarity: {}".format(dice_score))
# In case you want to evaluate with this metric within a deep learning model using tensorflow you can use the following:
#
# def dice_coef(y_true, y_pred):
#     y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
#     y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)
#
#
#
# ##
#
#
#
# import numpy as np
# np.random.seed(0)
# true = np.random.rand(10, 5, 5, 4)>0.5
# pred = np.random.rand(10, 5, 5, 4)>0.5
#
# def single_dice_coef(y_true, y_pred_bin):
#     # shape of y_true and y_pred_bin: (height, width)
#     intersection = np.sum(y_true * y_pred_bin)
#     if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
#         return 1
#     return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))
#
# def mean_dice_coef(y_true, y_pred_bin):
#     # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
#     batch_size = y_true.shape[0]
#     channel_num = y_true.shape[-1]
#     mean_dice_channel = 0.
#     for i in range(batch_size):
#         for j in range(channel_num):
#             channel_dice = single_dice_coef(y_true[i, :, :, j], y_pred_bin[i, :, :, j])
#             mean_dice_channel += channel_dice/(channel_num*batch_size)
#     return mean_dice_channel
#
# def dice_coef2(y_true, y_pred):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     union = np.sum(y_true_f) + np.sum(y_pred_f)
#     if union==0: return 1
#     intersection = np.sum(y_true_f * y_pred_f)
#     return 2. * intersection / union
#
# print(mean_dice_coef(true, pred))
# print(dice_coef2(true, pred))

# 0.4884357140842496
# 0.499001996007984






##----------------------------------------------------
#https://blog.actorsfit.com/a?ID=01600-b957e3c3-de31-4f7d-9f99-cb0a98b34a50

#%%
import nibabel as nib
import scipy.io as io
import os
import numpy as np
import tensorflow as tf
from keras import backend as K

#%%
from nnunet.dataset_conversion.Task040_KiTS import compute_dice_scores


def dice_coefficient(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth)/(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#%%
pred_dir = '../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/prediction_files_for_dc'
true_dir = '../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/labels'
pred_filenames = os.listdir(pred_dir)
pred_filenames.sort(key=lambda x:x[:-2])

#%%
true_filenames = os.listdir(true_dir)
true_filenames.sort(key=lambda x:x[:-8])

#%%

dice_value = np.zeros(10)
temp = []
for f in range(10):
    pred_path = os.path.join(pred_dir, pred_filenames[f])
    img_pred = nib.load(pred_path)
    y_pred = img_pred.get_fdata()
    true_path = os.path.join(true_dir, true_filenames[f])
    img_true = nib.load(true_path)
    y_true = img_true.get_fdata()

    temp.append(dice_coefficient(y_true, y_pred))
    pass


#%%
with tf.Session() as sess:
    dice_value = sess.run(temp)
    pass

mat_path = '../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/DICE_Lac.mat'
io.savemat(mat_path, {'DICE_value': dice_value})


# trying new way
#%%

pred_dir = '../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/prediction_files_for_dc'
true_dir = '../nnUNet_raw_data_base/nnUNet_raw_data/main_training_data_for_testing/labels'


for i in range(10):
    compute_dice_scores()