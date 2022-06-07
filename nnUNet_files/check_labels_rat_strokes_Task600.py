#%%
from pathlib import Path

import numpy as np
import nibabel as nib

#%%
# %% data import paths
img_dir = Path("../../Documents/WSIC/data WSIC/Task600")
labels_dir = img_dir / "labelsTr"
count = 0
all_values =[]
for i in labels_dir.glob("*.nii.gz"):
    # print(i.name)
    img = nib.load(i)
    data = img.get_fdata()
    # print(img)
    unique_values = np.unique(data)
    all_values.append(unique_values)
    count += 1

print(count)

#%%
img = nib.load(labels_dir / "Rat114.nii.gz")
data = img.get_fdata()
print(np.unique(data))

#%%
np.unique(all_values)