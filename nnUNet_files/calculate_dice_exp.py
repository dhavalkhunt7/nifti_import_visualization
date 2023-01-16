#%% create empty folder
# folder = "../nnUNet_raw_data_base/nnUNet_raw_data/task615_data_prep/"
# create_folder(folder + "christine_therapy_data")
# create_folder(folder + "christine_therapy_data/small_strokes_data")
# create_folder(folder + "christine_therapy_data/big_strokes_data")
# create_folder(folder + "christine_control_data")
# create_folder(folder + "christine_control_data/small_strokes_data")
# create_folder(folder + "christine_control_data/big_strokes_data")
# create_folder(folder + "theranostics_data")
# create_folder(folder + "theranostics_data/small_strokes_data")
# create_folder(folder + "theranostics_data/big_strokes_data")
#

#%% calculate dice
def calculate_dice(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = np.sum(pred * gt)
    return 2 * intersection / (np.sum(pred) + np.sum(gt))

#%% function to create empty folder if it doesn't exsit
