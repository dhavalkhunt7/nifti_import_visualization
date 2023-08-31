#%%
from bardapi import Bard
import os
import pandas as pd

#%%
api_key = 'ZQjOL__J6E8mciC2S8LnSAziHROubhVkePeH6Be6JodyXcBQpQMF0hQPqgrh_IyuNbg7kg.'

#Replace XXXX with the values you get from __Secure-1PSID
os.environ['_BARD_API_KEY']= api_key

#%%
# Set your input text
input_text = "what is google bard?"

# Send an API request and get a respons
bard_output = Bard().get_answer(input_text)['content']
print(bard_output)


#%%
data_path = "data/sql.csv"

#%% import csv from data_path as df
df = pd.read_csv(data_path)

#%% add column "sql_bard" to df
df["sql_bard"] = ""

#%% df to dict
df_dict = df.to_dict()

#%%
text_1 = df_dict["Text"][1]
print(text_1)

#%%
input_text = "create an sql queries from the given text: " + text_1


#%%
# Set your input text
# input_text = "what is google bard?"

# Send an API request and get a respons
bard_output = Bard().get_answer(input_text)['content']
print(bard_output)

#%%
print(bard_output)

#%% from output get the sql query
sql_query = bard_output.split("```sql")[1].split("```")[0]

print(sql_query)

#%% now remove if there is more than one space between words
sql_query_1 = " ".join(sql_query.split())

print(sql_query_1)

#%% now add the sql_query_1 to the dict in new column "sql_bard" for the same text
df_dict["sql_bard"][0] = sql_query_1

#%% now add the sql query to the dict in new column "sql_bard" for the same text
print(df_dict["sql_bard"][0])

#%% now automate the process for all the text in the dict if sql_bard is ""
for i in range(len(df_dict["sql_bard"])):
    if df_dict["sql_bard"][i] == "":
        input_text = "create an sql queries from the given text: " + df_dict["Text"][i]
        bard_output = Bard().get_answer(input_text)['content']
        sql_query = bard_output.split("```sql")[1].split("```")[0]
        sql_query_1 = " ".join(sql_query.split())
        df_dict["sql_bard"][i] = sql_query_1
        print(i)
        print(df_dict["sql_bard"][i])

#%% dict to df
df = pd.DataFrame.from_dict(df_dict)

#%% df to csv with the same name
df.to_csv(data_path, index=False)