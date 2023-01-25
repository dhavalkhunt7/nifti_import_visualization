# %%
import nibabel as nib
import numpy as np
from pathlib import Path
import copy
import matplotlib.pyplot as plt

#%%
def extract_data(img):
    # load nii data
    data = nib.load(img).get_fdata()
    # if img_data has nan values then replace them with 0
    if np.isnan(data).any():
        data[np.isnan(data)] = 0
    # return the data
    return data

#%%
def seg_list(seg_path):
    seg_list = []
    for i in seg_path.glob("*.nii.gz"):
        seg_list.append(i.name.split(".nii")[0])
    return seg_list

#%%
def create_dict_for_mean_jpm(data_path):
    dict_final_data = {}
    gt_path = data_path / "labelsTs"
    seg_path = data_path / "result"
    seg_path_3d = data_path / "result_3d"
    t2_path = data_path / "imagesTs"

    print("checking all the paths")
    # print the paths
    print("gt path : " + str(gt_path))
    print("seg path : " + str(seg_path))
    print("seg path 3d : " + str(seg_path_3d))

    for i in seg_path.glob("*.nii.gz"):
        # print(i.name)
        name = i.name.split(".nii")[0]
        print(name)
        dict_final_data[name] = {}

        if (t2_path / i.name.replace(".nii","_0001.nii")).exists():
            # print("t2 exists")
            # extract the data and add it to the dictionary dict_final_data
            dict_final_data[name]["t2"] = extract_data(t2_path / i.name.replace(".nii","_0001.nii"))
        if (gt_path / i.name).exists():
            # print("seg exists")
            # dict_task[name]["gt"] = str(gt_path / i.name)
            # extract the data and add it to the dictionary dict_final_data
            dict_final_data[name]["gt"] = extract_data(gt_path / i.name)
        dict_final_data[name]["seg"] = extract_data(seg_path / i.name)
        # dict_task[name]["seg"] = str(i)
        if (seg_path_3d / i.name).exists():
            # print("seg_3d exists")
            # dict_task[name]["seg_3d"] = str(seg_path_3d / i.name)
            # extract the data and add it to the dictionary dict_final_data
            dict_final_data[name]["seg_3d"] = extract_data(seg_path_3d / i.name)

    print(" added all the extracted data to the dictionary")
    # print the dictionary
    # print(dict_task)

    return dict_final_data


# %% control therapy separate mean files foe jpm
def create_dict_for_mean_jpm_control_therapy(filepath):
    dict_final_data_therapy = {}
    dict_final_data_control = {}

    # get the list of all the control files
    control_path = Path("../../../Documents/data/adrian_data/devided/Rats24h/control")
    control_data_list = []
    for i in control_path.iterdir():
        print(i.name)
        control_data_name = i.name.split("-")[0]
        print(control_data_name)
        # add it to list
        control_data_list.append(control_data_name)

    gt_path = filepath / "labelsTs"
    seg_path = filepath / "result"
    t2_path = filepath / "imagesTs"

    print("checking all the paths")
    # print the paths
    print("gt path : " + str(gt_path))
    print("seg path : " + str(seg_path))

    for i in seg_path.glob("*.nii.gz"):
        # print(i.name)
        name = i.name.split(".nii")[0]
        print(name)

        if name in control_data_list:
            print(i.name  + "  control")
            dict_final_data_control[name] = {}

            if (t2_path / i.name.replace(".nii", "_0001.nii")).exists():
                # print("t2 exists")
                # extract the data and add it to the dictionary dict_final_data
                dict_final_data_control[name]["t2"] = extract_data(t2_path / i.name.replace(".nii", "_0001.nii"))
            if (gt_path / i.name).exists():
                # print("seg exists")
                # dict_task[name]["gt"] = str(gt_path / i.name)
                # extract the data and add it to the dictionary dict_final_data
                dict_final_data_control[name]["gt"] = extract_data(gt_path / i.name)
            dict_final_data_control[name]["seg"] = extract_data(seg_path / i.name)
        else:
            print(i.name + "  therapy")
            dict_final_data_therapy[name] = {}

            if (t2_path / i.name.replace(".nii", "_0001.nii")).exists():
                # print("t2 exists")
                # extract the data and add it to the dictionary dict_final_data
                dict_final_data_therapy[name]["t2"] = extract_data(t2_path / i.name.replace(".nii", "_0001.nii"))
            if (gt_path / i.name).exists():
                # print("seg exists")
                # dict_task[name]["gt"] = str(gt_path / i.name)
                # extract the data and add it to the dictionary dict_final_data
                dict_final_data_therapy[name]["gt"] = extract_data(gt_path / i.name)
            dict_final_data_therapy[name]["seg"] = extract_data(seg_path / i.name)

    print(" added all the extracted data to the dictionary")
    # print the dictionary
    # print(dict_task)

    return dict_final_data_control, dict_final_data_therapy

#%% from list fro gmm
def create_dict_for_mean_jpm_with_list(data_path, seg_list):
    dict_final_data = {}
    gt_path = data_path / "labelsTs"
    seg_path = data_path / "result"
    seg_path_3d = data_path / "result_3d"
    t2_path = data_path / "imagesTs"

    print("checking all the paths")
    # print the paths
    print("gt path : " + str(gt_path))
    print("seg path : " + str(seg_path))
    print("seg path 3d : " + str(seg_path_3d))

    for i in seg_path.glob("*.nii"):
        # print(i.name)
        name = i.name.split(".nii")[0]
        print(name)
        if name in seg_list:
            dict_final_data[name] = {}

            if (t2_path / i.name.replace(".nii","_0001.nii")).exists():
                # print("t2 exists")
                # extract the data and add it to the dictionary dict_final_data
                dict_final_data[name]["t2"] = extract_data(t2_path / i.name.replace(".nii","_0001.nii"))
            if (gt_path / i.name).exists():
                # print("seg exists")
                # dict_task[name]["gt"] = str(gt_path / i.name)
                # extract the data and add it to the dictionary dict_final_data
                dict_final_data[name]["gt"] = extract_data(gt_path / i.name)
            dict_final_data[name]["seg"] = extract_data(seg_path / i.name)
            # dict_task[name]["seg"] = str(i)
            if (seg_path_3d / i.name).exists():
                # print("seg_3d exists")
                # dict_task[name]["seg_3d"] = str(seg_path_3d / i.name)
                # extract the data and add it to the dictionary dict_final_data
                dict_final_data[name]["seg_3d"] = extract_data(seg_path_3d / i.name)
        else:
            print(name  + " not in the list")

    print(" added all the extracted data to the dictionary")
    # print the dictionary
    # print(dict_task)

    return dict_final_data



#%% function for code 1
def create_mean_img_jpm(dict_final_data, jpm_img_path):
    t2_list = []
    gt_list = []
    seg_list = []

    print(" adding the data to the list...")

    # add the data to the list from dict_final_data
    for i in dict_final_data.keys():
        # print(i)
        seg_list.append(dict_final_data[i]["seg"])
        # check if there is nan values in the t2 data and if there is then replace them with 0 and add it to the list,
        # if not add to list directly
        if np.isnan(dict_final_data[i]["t2"]).any():
            dict_final_data[i]["t2"][np.isnan(dict_final_data[i]["t2"])] = 0
            t2_list.append(dict_final_data[i]["t2"])
        else:
            t2_list.append(dict_final_data[i]["t2"])
        gt_list.append(dict_final_data[i]["gt"])
    print("list created")

    print("saving final images for jpm")
    # save the final images
    nib.save(nib.Nifti1Image(np.mean(t2_list, axis=0), np.eye(4)), jpm_img_path / "final_t2.nii.gz")
    nib.save(nib.Nifti1Image(np.mean(gt_list, axis=0), np.eye(4)), jpm_img_path / "final_gt.nii.gz")
    nib.save(nib.Nifti1Image(np.mean(seg_list, axis=0), np.eye(4)), jpm_img_path / "final_seg.nii.gz")

    print("final images saved in folder" + str(jpm_img_path))


# %% create a function plot subplots of the image and the masks

def plot_subplots(image, mask_img, n_rows, n_cols):
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(150, 50))

    n = 0
    slice_no = 45
    for _ in range(n_rows):
        for _ in range(n_cols):
            ax[n].imshow(image[:, :, slice_no], cmap='gray')
            ax[n].imshow(mask_img[:, :, slice_no], cmap='hot', alpha=0.7, vmin=0, vmax=1)
            ax[n].set_xticks([])
            ax[n].set_yticks([])
            # hider border of subplot
            ax[n].set_frame_on(False)
            ax[n].set_title('Slice {}'.format(slice_no), color='r', fontsize=5)
            n += 1
            slice_no += 10
    fig.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    return fig

#%%
# adc_rot = np.flip(adc_rot, axis=1)
# gt_rot = np.flip(gt_rot, axis=1)
# pred_rot = np.flip(pred_rot, axis=1)
#
# #%% plt adc, adc_gt, adc_pred
# plt_adc = plot_subplots_single_modality(adc_rot, 1, 8)
# plt_adc_gt = plot_subplots(adc_rot, gt_rot, 1, 8)
# plt_adc_pred = plot_subplots(adc_rot, pred_rot, 1, 8)
#
#
# # %% folder to save the plots
# plot_path = "new_plots/601"
# # craete a folder if it does not exist
# if not Path(plot_path).exists():
#     Path(plot_path).mkdir()
#
# # %% save the plots
# plt_adc.savefig(plot_path + "/adc.png", bbox_inches='tight', dpi=300)
# plt_adc_gt.savefig(plot_path + "/adc_gt.png", bbox_inches='tight', dpi=300)
# plt_adc_pred.savefig(plot_path + "/adc_pred.png", bbox_inches='tight', dpi=300)
#
#
# # %% save the figure
# plt_t2_gt.savefig(plot_path + "/therapy_t2_gt.pdf", bbox_inches="tight", dpi=300)
# plt_adc_gt.savefig(plot_path + "/therapy_adc_gt.pdf", bbox_inches="tight", dpi=300)
#
# # %% plot only adc and ony t2
# plt_t2 = plot_subplots_single_modality(therapy_t2, 1, 5)
# plt_adc = plot_subplots_single_modality(therapy_adc, 1, 5)
#
# # %% save the figure
# plt_t2.savefig(plot_path + "/therapy_t2.pdf", bbox_inches="tight", dpi=300)
# plt_adc.savefig(plot_path + "/therapy_adc.pdf", bbox_inches="tight", dpi=300)






#%% old functions
def extract_data_from_folder(folder):
    # create a list for all the data
    list_adc = []
    list_t2 = []
    list_gt = []
    for i in folder.glob("*"):
        # print(i.name)
        new_dir = i
        for j in new_dir.glob("*.nii"):
            # print(j.name)
            if j.name == "Masked_ADC.nii":
                # print(j.name)
                # extract the data
                img_data = extract_data(j)
                # if data has more than 1 unique values then add it to the list
                if len(np.unique(img_data)) > 1:
                    list_adc.append(img_data)
            elif j.name == "Masked_T2.nii":
                # print(j.name)
                # extract the data
                img_data = extract_data(j)
                # if data has more than 1 unique values then add it to the list
                if len(np.unique(img_data)) > 1:
                    list_t2.append(img_data)
            elif j.name == "GroundTruth24h.nii" or j.name == "Voi_1w.nii" or j.name == "Voi_72h.nii" or j.name == \
                    "Voi_1m.nii":
                # print(j.name)
                # extract the data
                img_data = extract_data(j)
                # print unique values
                # if data has more than 1 unique values then add it to the list
                if len(np.unique(img_data)) > 1:
                    list_gt.append(img_data)
    return list_adc, list_t2, list_gt



# %% create a function for code 1 to 4
def create_final_data(task, database):
    # extract the data from the therapy folder
    therapy_adc, therapy_t2, therapy_gt = extract_data_from_folder(task / "therapy")
    # extract the data from the control folder
    control_adc, control_t2, control_gt = extract_data_from_folder(task / "control")

    # create a new folder for the final data
    final_data = database / "final_data" / task.name
    # if the folder does not exist then create it
    if not final_data.exists():
        final_data.mkdir(parents=True)

    # save the final data for therapy and control
    nib.save(nib.Nifti1Image(np.mean(therapy_adc, axis=0), np.eye(4)), final_data / "final_adc_therapy.nii.gz")
    nib.save(nib.Nifti1Image(np.mean(therapy_t2, axis=0), np.eye(4)), final_data / "final_t2_therapy.nii.gz")
    nib.save(nib.Nifti1Image(np.mean(therapy_gt, axis=0), np.eye(4)), final_data / "final_gt_therapy.nii.gz")

    nib.save(nib.Nifti1Image(np.mean(control_adc, axis=0), np.eye(4)), final_data / "final_adc_control.nii.gz")
    nib.save(nib.Nifti1Image(np.mean(control_t2, axis=0), np.eye(4)), final_data / "final_t2_control.nii.gz")
    nib.save(nib.Nifti1Image(np.mean(control_gt, axis=0), np.eye(4)), final_data / "final_gt_control.nii.gz")

#%%