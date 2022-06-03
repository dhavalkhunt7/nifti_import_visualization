#%%
import mat4py as mat4py
import scipy.io as sc
from scipy import stats


#%%
mat = sc.loadmat('../../projects/data_adrian/Table_24h_DV_only.mat')

type(mat)
# dict_all_files = mat['None']
# type(dict_all_files)
#%%

for i in mat.items():
    print(i)

#%%
mat.shape()


#%%
def dict_dims(mydict):
    d1 = len(mydict)
    d2 = 0
    for d in mydict:
        d2 = max(d2, len(d))
    return d1, d2

#%%
dict_dims(mat)