# %% load txt file
f = open("input/run1.txt", "r")
content = f.read()

# %% splitting the data to get only important parts from  all text
# taking all the important epoch related information from log file
start = '2022-05-04 17:05:29.496988:'
end = '2022-05-07 21:12:16.923839:'
data = (content.split(start))[1].split(end)[0]
# print(data)

# %% saving every individual parts in list (epoch wise)
splited_data = data.split("\n\n")
print(splited_data[0])

# %%
# making empty dictionary
log_dict = {}
for i in range(1000):
    log_dict[i] = {}  # making dic of dict to save all the necessary information
    # individual epoch info that we get, accessing every line one by one to extract important variables and saving it
    # into dict
    list_0 = splited_data[i].split("\n")

    train = list_0[2].split("train loss : ")
    # train_loss = train[1]
    log_dict[i]["train_loss"] = float(train[1])

    valid = list_0[3].split("validation loss: ")
    # validation_loss = valid[1]
    log_dict[i]["validation_loss"] = float(valid[1])

    dice = list_0[4].split("Average global foreground Dice: ")
    average_global_foreground_dice = dice[1]
    dc1 = (average_global_foreground_dice.split("["))[1].split(", ")[0]
    dc2 = (average_global_foreground_dice.split(", "))[1].split(", ")[0]
    dc3 = (average_global_foreground_dice.split(", "))[2].split("]")[0]
    log_dict[i]["dc1"] = float(dc1)
    log_dict[i]["dc2"] = float(dc2)
    log_dict[i]["dc3"] = float(dc3)

    # lr_m = list_0[6].split("lr: ")
    # learning_rate = lr_m[1]
    #
    # time = list_0[7].split("This epoch took ")
    # epoch_time = time[1]

# %%

# print(train_loss)
# print(validation_loss)
# print(average_global_foreground_dice)
# print(dc1)
# print(dc2)
# print(dc3)
# print(learning_rate)
# print(epoch_time)

# %%
# access dict as dataframe(something like Excel file) for plotting
import pandas as pd

df = pd.DataFrame.from_dict(log_dict).T

# %% plotting all variables using matplot
import matplotlib.pyplot as plt

plt.plot(df)
plt.show()

# %% plotly = library for plotting but more efficient compared to matplotlib
import plotly.express as px

# showing only dc variable to plot for 100 epoch
fig = px.line(df.iloc[:100, [2, 3, 4]])
fig.show()
fig.write_image("output/fig.png")

# %%
import plotly.express as px

# plotting all var on 100 epochs
fig = px.line(df.iloc[:100])
fig.show()
fig.write_image("output/all_var_100.png")

# %%
# plot train_loss on 1000 epochs
fig = px.line(df, y="train_loss")
fig.write_image("output/train_loss.png")







# %% load txt file
f = open("../nnUNet_raw_data_base/run505.txt", "r")
content = f.read()

# %% splitting the data to get only important parts from  all text
# taking all the important epoch related information from log file
start = '2022-05-17 14:15:46.621613:'
end = '2022-05-19 08:58:50.912868:'
data = (content.split(start))[1].split(end)[0]
print(data[2])

# %% saving every individual parts in list (epoch wise)
splited_data = data.split("\n\n")
print(splited_data[0])

# %%
# making empty dictionary
log_dict = {}
for i in range(1000):
    log_dict[i] = {}  # making dic of dict to save all the necessary information
    # individual epoch info that we get, accessing every line one by one to extract important variables and saving it
    # into dict
    list_0 = splited_data[i].split("\n")

    train = list_0[2].split("train loss : ")
    # train_loss = train[1]
    log_dict[i]["train_loss"] = float(train[1])

    valid = list_0[3].split("validation loss: ")
    # validation_loss = valid[1]
    log_dict[i]["validation_loss"] = float(valid[1])

    dice = list_0[4].split("Average global foreground Dice: ")
    average_global_foreground_dice = dice[1]
    dc1 = (average_global_foreground_dice.split("["))[1].split(", ")[0]
    dc2 = (average_global_foreground_dice.split(", "))[1].split(", ")[0]
    dc3 = (average_global_foreground_dice.split(", "))[2].split("]")[0]
    log_dict[i]["dc1"] = float(dc1)
    log_dict[i]["dc2"] = float(dc2)
    log_dict[i]["dc3"] = float(dc3)

    # lr_m = list_0[6].split("lr: ")
    # learning_rate = lr_m[1]
    #
    # time = list_0[7].split("This epoch took ")
    # epoch_time = time[1]

# %%

# print(train_loss)
# print(validation_loss)
# print(average_global_foreground_dice)
# print(dc1)
# print(dc2)
# print(dc3)
# print(learning_rate)
# print(epoch_time)

# %%
# access dict as dataframe(something like Excel file) for plotting
import pandas as pd

df = pd.DataFrame.from_dict(log_dict).T

#%% plotting all variables using matplot
import matplotlib.pyplot as plt
#
plt.plot(df)
plt.show()
#
#%% plotly = library for plotting but more efficient compared to matplotlib
import plotly.express as px
#
# showing only dc variable to plot for 100 epoch
fig = px.line(df.iloc[:100, [2, 3, 4]])
fig.show()
fig.write_image("output/fig.png")
#
#%%
# import plotly.express as px
#
# plotting all var on 100 epochs
# fig = px.line(df.iloc[:100])
# fig.show()
# fig.write_image("output/all_var_100.png")
#
#%% plot train_loss on 1000 epochs
# fig = px.line(df, y="train_loss")
# fig.write_image("output/train_loss.png")