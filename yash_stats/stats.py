#%%
import shutil
from pathlib import Path
import pandas as pd
from utilities.confusionMatrix_dependent_functions import *

#%%
dataset_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data")
data_path = dataset_path / "Task905_BrainCancerClassification"

#%%
rs_path = data_path / "testing"
seg_path = rs_path / "result"
gt_path = rs_path / "trail"
dict_yash_result = {}

calc_stats(gt_path, seg_path, dict_yash_result)

#%% dict to df
df_yash_result = pd.DataFrame.from_dict(dict_yash_result, orient='index')

#%%
df_yash_result.to_csv(str(data_path / "result_csv_file/yash_result") + ".csv")

#%%
df = df_yash_result

#%%
import pandas as pd
import plotly.graph_objects as go

# suppose df is your DataFrame

fig = go.Figure()

fig.add_trace(go.Box(y=df['dice'], name='Dice'))

fig.update_layout(title='Dice Box Plot',
                  yaxis=dict(range=[0, 1]))
# save it in folder data_path / "result_csv_file"
fig.write_image(str(data_path / "result_csv_file/dice") + ".pdf")
# fig.write_image("dice_boxplot.png")  # saves the plot as a png
# fig.write_image("dice_boxplot.pdf")  # or you can uncomment this line to save the plot as a pdf
# fig.show()
