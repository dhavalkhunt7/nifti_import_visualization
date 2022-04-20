import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
# %%
import numpy as np

path1 = 'input/imageT/BRATS_001.nii.gz'
img1 = nib.load(path1)

# %%
t1_hdr= img1.header
print(t1_hdr)

#%%
t1_hdr.keys()

#%%
t1_data = img1.get_fdata()
type(t1_data)

masked_data = np.ma.masked_equal(t1_data, 0,copy=False) # excluding zero
print(np.min(masked_data))
print(np.max(t1_data))

print(t1_data.shape)

print(t1_data[125, 125, 79, 2])
mid_slice_x = t1_data[125, :, :, 2]

plt.imshow(mid_slice_x.T, cmap='gist_heat_r', origin='lower')
plt.colorbar(label='signal intensity')
plt.show()

#%% affine_


#%%
sns.heatmap(mid_slice_x.T, cmap='coolwarm', robust=True)
plt.show()


#%%

