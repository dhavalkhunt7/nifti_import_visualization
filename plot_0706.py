#%%
from pathlib import Path

import nibabel as nib
import numpy as np; np.random.seed(1)
import matplotlib.pyplot as plt


#%%
x = np.arange(100)
y = np.abs(np.cumsum(np.random.rand(100)-0.5))/4.
y1 = np.copy(y)
y1[y1 < 0.7] = np.nan

plt.plot(x,y)
plt.plot(x,y1)

plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

# values of x
x = np.array([1, 2, 3, 4, 5,
              6, 7, 8, 9, 10])

# values of y
y = np.array([10, 9, 8, 7, 6, 5,
              4, 3, 2, 1])

# empty list, will hold color value
# corresponding to x
col = []

for i in range(0, len(x)):
    if x[i] < 7:
        col.append('blue')
    else:
        col.append('magenta')

for i in range(len(x)):
    # plotting the corresponding x with y
    # and respective color
    plt.scatter(x[i], y[i], c=col[i], s=10,
                linewidth=0)

plt.show()

#%%
label_path = Path("nifti_store/BRATS_1020.nii.gz")


#%%
img = nib.load(label_path)
data = img.get_fdata()
print(data.shape)

#%%
data
#%%

slices_list = [75, 110, 125]  # for x and y
slices_z = [38, 50, 70]
img_xslices = [data[s, :, :] for s in slices_list]
img_yslices = [data[:, s, :] for s in slices_list]
img_zslices = [data[:, :, s] for s in slices_z]

#%%

# plt.imshow(img_zslices[2], vmin=0, vmax=1, cmap="Reds")
plt.imshow(img_zslices[2], vmin=0, vmax=3, cmap='BrBG')
plt.show()
