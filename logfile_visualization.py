#%%
f = open("input/run1.txt", "r")
content = f.read()

#%%
start = '2022-05-04 17:05:29.496988:'
end = '2022-05-07 21:12:16.923839:'
data = (content.split(start))[1].split(end)[0]
# print(data)

#%%
splited_data = data.split("\n\n")
print(splited_data[0])

#%%
log_dict = {}
for i in range(1000):
    log_dict[i]= {}
    list_0 = splited_data[i].split("\n")

    train=list_0[2].split("train loss : ")
    # train_loss = train[1]
    log_dict[i]["train_loss"]=float(train[1])

    valid = list_0[3].split("validation loss: ")
    # validation_loss = valid[1]
    log_dict[i]["validation_loss"]= float(valid[1])

    dice = list_0[4].split("Average global foreground Dice: ")
    average_global_foreground_dice = dice[1]
    dc1 = (average_global_foreground_dice.split("["))[1].split(", ")[0]
    dc2 = (average_global_foreground_dice.split(", "))[1].split(", ")[0]
    dc3 = (average_global_foreground_dice.split(", "))[2].split("]")[0]
    log_dict[i]["dc1"] =float(dc1)
    log_dict[i]["dc2"] =float(dc2)
    log_dict[i]["dc3"] =float(dc3)

    # lr_m = list_0[6].split("lr: ")
    # learning_rate = lr_m[1]
    #
    # time = list_0[7].split("This epoch took ")
    # epoch_time = time[1]

#%%

# print(train_loss)
# print(validation_loss)
print(average_global_foreground_dice)
print(dc1)
print(dc2)
print(dc3)
# print(learning_rate)
# print(epoch_time)


import pandas as pd
df = pd.DataFrame.from_dict(log_dict).T


#%%
import matplotlib.pyplot as plt
plt.plot(df)
plt.show()


#%%
import plotly.express as px

fig = px.line(df.iloc[:100,[2,3,4]])
fig.show()
fig.write_image("output/fig.png")


#%%
import plotly.express as px

fig = px.line(df.iloc[:100])
fig.show()
fig.write_image("output/all_var_100.png")

#%%

fig = px.line(df, y="train_loss")
fig.write_image("output/train_loss.png")

#%%
import plotly.express as px

# Creating the Figure instance
fig = px.line(x=[1, 2, 3], y=[1, 2, 3])

# printing the figure instance
fig.show(renderer='png')
fig.write_image("output/fig1.png")