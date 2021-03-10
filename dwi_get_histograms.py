import numpy as np
import glob
import pandas as pd
import SimpleITK as sitk
from skimage.measure import regionprops
from utils import cerebellum_normalization


def get_lesion(regions):
    output = pd.DataFrame()
    ipsilateral = regions[0].intensity_image
    contralateral = regions[1].intensity_image
    ipsi_pixels = ipsilateral.flatten()
    ipsi_pixels = ipsi_pixels[ipsi_pixels != 0.0]
    ipsi_dist = np.histogram(ipsi_pixels, bins=np.arange(0,5,0.01))[0]
    contra_pixels = contralateral.flatten()
    contra_pixels = contra_pixels[contra_pixels != 0.0]
    contra_dist = np.histogram(contra_pixels, bins=np.arange(0,5,0.01))[0]
    trace = (ipsi_dist - contra_dist)
    trace = np.sum(trace[trace > 0])
    return trace

import matplotlib.pyplot as plt
plt.hist(ipsi_pixels, bins=400)
plt.hist(contra_pixels, bins=400)
plt.hist(ipsi_pixels-contra_pixels, bins=400)
plt.show()

plt.plot(ipsi_dist)
plt.plot(contra_dist)
plt.plot(ipsi_dist-contra_dist)
plt.plot((ipsi_dist-contra_dist)-contra_dist)
plt.show()

day_list = ['day00', 'day01']
data_df = pd.DataFrame()
region_labels = pd.read_csv(
    "G:\SIGMA\SIGMA_Rat_Brain_Atlases\SIGMA_Anatomical_Atlas\SIGMA_Anatomical_Brain_Atlas_ListOfStructures.csv",
    sep = ',')
cerebellum_int = [221, 222, 341, 342, 991, 992, 1001, 1002, 1011, 1012]

for day in day_list:

    output_df = pd.DataFrame()
    print(day)
    atlas_path = glob.glob('G:/coregistered-files/' + day + '/*_atlas.nii')
    volume_path = glob.glob('G:/coregistered-files/' + day + '/*_remasked-volume.nii')
    template_path = glob.glob('G:/coregistered-files/' + day + '/*_template.nii')
    diffusion_path = glob.glob('G:/coregistered-files/' + day + '/*_diffusion.nii')
    atlas_path.sort()
    volume_path.sort()
    template_path.sort()
    diffusion_path.sort()
    i = 1
    list_len = len(atlas_path)
    data = {}

    for atlas, volume, template, diffusion in zip(atlas_path, volume_path, template_path, diffusion_path):
        print(f'File {i}/{list_len}')
        i += 1

        atlas_array = sitk.ReadImage(atlas)
        atlas_array = sitk.GetArrayFromImage(atlas_array)
        volume_array = sitk.ReadImage(volume)
        volume_array = sitk.GetArrayFromImage(volume_array)
        diffusion_array = sitk.ReadImage(diffusion)
        diffusion_array = sitk.GetArrayFromImage(diffusion_array)

        region_ints = np.unique(atlas_array)
        region_ints = region_ints[region_ints != 0]
        ipsilateral_h = region_ints[region_ints % 2 == 0]
        contralateral_h = region_ints[region_ints % 2 == 1]

        for region in ipsilateral_h:
            atlas_array[atlas_array == region] = 1
        for region in contralateral_h:
            atlas_array[atlas_array == region] = 2

        normed_diff_array = cerebellum_normalization(atlas_array, diffusion_array, cerebellum_int)

        volume_regions = regionprops(atlas_array, normed_diff_array)

        lesion_size = get_lesion(volume_regions)

        rat = atlas.split('\\')[1][:-10] + '_masked-img.nii'
        rat_series = pd.Series(lesion_size, index=[rat])
        output_df = pd.concat([output_df, rat_series], ignore_index=False)
        print(rat + ' is completed!')

    output_df.to_csv(f'G:/mri-results/t2_data_{day}_diffusion_lesion.csv', sep = ';')

output_df = pd.read_csv("G:/mri-results/t2_data_day00_diffusion_lesion.csv", sep=';', index_col=0)
groups = pd.read_csv("G:/mri-results/exp_groups.csv", sep=';', index_col=0)
rats_in_group = [data.split('_')[:4] for data in output_df.index]
rats_in_group = ['_'.join(rat) for rat in rats_in_group]
index_dict = {key: value for key, value in zip(output_df.index, rats_in_group)}
output_df = output_df.rename(index=index_dict)
output_df = pd.merge(output_df, groups, right_index=True, left_index=True)
output_df.to_csv(f'G:/mri-results/t2_data_day00_diffusion_lesion_grps.csv', sep = ';')

df = pd.read_csv("G:/mri-results/dwi_lesions_v2.csv", sep=';', index_col=0)
mph = df[df['Group'] == 'MPH']
ctrl = df[df['Group'] == 'Ctrl']
vehicle = df[df['Group'] == 'Cyclo']
plt.scatter(mph['day00'], mph['day01'])
plt.scatter(ctrl['day00'], ctrl['day01'])
plt.scatter(vehicle['day00'], vehicle['day01'])
plt.legend(['MPH', 'Ctrl', 'Cyclo'])

plt.show()