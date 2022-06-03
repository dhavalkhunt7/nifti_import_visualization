#%%
import mat4py as mat4py
import scipy.io as sc
from scipy import stats


#%%
mat = sc.loadmat('../../projects/data_adrian/Table_24h_DV_only.mat')

type(mat)
dict_all_files = mat['None']
type(dict_all_files)
#%%

for i in mat.items():
    print(i)

#%%
mat['None'][0][3]
