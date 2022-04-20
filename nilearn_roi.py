#%%
import nib as nib
import nibabel
from matplotlib import pyplot as plt
from nilearn import plotting,image
from nilearn.plotting import plot_epi, plot_stat_map

path = 'input/imageT/BRATS_001.nii'

#plotting.plot_prob_atlas(path, display_mode='x') sagital
#plt.show()
#plotting.plot_prob_atlas(path, display_mode='y') coronal
#plt.show()
#plotting.plot_prob_atlas(path, display_mode='z') # axial
#plt.show()
# ortho
#tiled
# mosaic
plotting.plot_prob_atlas(path, display_mode='mosaic')
plt.show()

#%% 4d to 3d by taking 1 t
first_volume = image.index_img(path,2)

plotting.plot_glass_brain(first_volume, display_mode='lyrz', threshold=3)
plt.show()

#%% smoothing

smoothened_img =image.smooth_img(first_volume, fwhm=3)
plotting.plot_img(smoothened_img)
plt.show()

#%%
fwhm_img = image.smooth_img(path, fwhm=25)
cut_coords = [-25,-37,-6]
mean_img = image.mean_img(fwhm_img)
plotting.plot_epi(mean_img, cut_coords=cut_coords)
plt.show()
#%%
plotting.plot_epi(first_volume)
plt.show()

#%%
image.get_data(fwhm_img).shape

#%%
#plotting.plot_prob_atlas(path)
#plt.show()

display = plotting.plot_stat_map(image.index_img(path,3), colorbar=True)
#display.add_overlay(image.index_img(path, 3), cmap= plotting.cm.black_green, colorbar = True)
plt.show()



#%% loading all volumes in 4d
for img in image.iter_img(path):
    plotting.plot_stat_map(img, threshold=3, display_mode="z", cut_coords=1, colorbar=True)
plt.show()



#%% visualization trail 1
view=plotting.view_img(image.mean_img(path), threshold=None)
view.open_in_browser()


#%% background enhancement
plotting.plot_stat_map(first_volume, dim=1)
plt.show()

#%%
plotting.plot_epi(first_volume)
plt.show()


#%%
import nibabel as nib
from nilearn.datasets import load_mni152_template
# cope_img = nib.load(path)
# cope_4d = cope_img.get_fdata()
# cope_4d.shape
#
# #mni = load_mni152_template().get_fdata()
#
# plt.figure(figsize=(12,6))
# subjects = ['sub-%s' % str(i).zfill(2) for i in range(2, 14)]

# for i in range(cope_4d.shape[-1]):
#     plt.subplot(2,6,i+1)
#     plt.title(subjects[i], fontsize=20)
#     plt.imshow(mni[:,:,70].T, origin='lower', cmap='gray')
#     this_data = cope_4d[:,:,70]
#     plt.imshow(this_data.T, cmap='seismic',origin='lower', vmin=-500, vmax=500, alpha=0.6)
#     plt.axes('off')
# plt.tight_layout()
# plt.show()


#%%
mask_path = 'input/labelsT/BRATS_001.nii.gz'


plotting.plot_roi(mask_path, bg_img=first_volume, cmap='Paired', colorbar=True)
plt.show()

print(mask_path)

masked_img = image.load_img(mask_path)
masked_data = image.get_data(masked_img)
h_masked_data = masked_img.header
print(h_masked_data)

plotting.plot_img(mask_path)
# plt.show()

#%%

plotting.plot_roi(masked_img, first_volume)
plt.show()