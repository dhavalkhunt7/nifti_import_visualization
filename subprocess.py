#%%
import subprocess

#%% create function to execute all the subprocess after one by one is finished
def execute_subprocess_list(subprocess_list):
    for subprocess in subprocess_list:
        subprocess.wait()
        print(subprocess.returncode)
        if subprocess.returncode != 0:
            print("subprocess failed")
            return False
    return True

#%% create subprocess list
subprocess_list = []

#%% create function to give next task to gpu if gpu is available
def give_next_task_to_gpu(gpu_list, subprocess_list):
    for gpu in gpu_list:
        if len(subprocess_list) == 0:
            break
        else:
            subprocess_list.append(subprocess.Popen(gpu, shell=True))
    return subprocess_list

#%% function to manage subprocess for gpus
def manage_subprocess_for_gpus(gpu_list, subprocess_list):
    if len(subprocess_list) == 0:
        subprocess_list = give_next_task_to_gpu(gpu_list, subprocess_list)
    else:
        for subprocess in subprocess_list:
            if subprocess.poll() is not None:
                subprocess_list.remove(subprocess)
                subprocess_list = give_next_task_to_gpu(gpu_list, subprocess_list)
                break
    return subprocess_list

#%% create function to make subprocess to list
def make_subprocess_list(folds, model, task_name):
    folds = [0, 1, 2, 3, 4]
    model = "2d", "3d_fullres"
    task_name = "620", "625"
    subprocess_list.append(subprocess.Popen("nnUnet_train "+ model + " nnUnetTrainerV2 " + task_name + " " + folds + " --npz" , shell=True))
    return subprocess_list


