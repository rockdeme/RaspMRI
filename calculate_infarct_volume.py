"""Calculate hyperintensive brain volume

This script is used to count the hyperintensive voxels based on the 95% CI of the healthy counterparts.
"""

import glob
import pandas as pd
import SimpleITK as sitk
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from utils import cerebellum_normalization, get_voxels


cerebellum_int = [221, 222, 341, 342, 991, 992, 1001, 1002, 1011, 1012]

day_list = ['day03', 'day07', 'day21']
data_df = pd.DataFrame()

for day in day_list:
    print(day)
    atlas_path = glob.glob('G:/coregistered-files/' + day + '/*_atlas.nii')
    volume_path = glob.glob('G:/coregistered-files/' + day + '/*_remasked-volume.nii')
    template_path = glob.glob('G:/coregistered-files/' + day + '/*_template.nii')
    i = 1
    list_len = len(atlas_path)
    data = {}
    for atlas, volume, template in zip(atlas_path, volume_path, template_path):
        print(f'File {i}/{list_len}')
        i += 1

        atlas_array = sitk.ReadImage(atlas)
        atlas_array = sitk.GetArrayFromImage(atlas_array)
        volume_array = sitk.ReadImage(volume)
        volume_array = sitk.GetArrayFromImage(volume_array)
        template_array = sitk.ReadImage(template)
        template_array = sitk.GetArrayFromImage(template_array)

        template_array_norm = cerebellum_normalization(atlas_array, template_array, cerebellum_int)
        volume_array_norm = cerebellum_normalization(atlas_array, volume_array, cerebellum_int)

        volume_regions = regionprops(atlas_array, volume_array_norm)
        template_regions = regionprops(atlas_array, template_array_norm)

        infarct_size = get_voxels(volume_regions, template_regions)

        rat = '_'.join(atlas.split('\\')[-1].split('_')[0:4])
        data[rat] = infarct_size
        print(f"{rat}: {data[rat]}")
        day_df = pd.DataFrame.from_dict(data, orient='index', columns=[day])

    data_df = pd.concat([data_df, day_df], axis=1)

data_df.to_csv('G:\mri-results\infarct_volume_hyper.csv', sep=';', decimal=',')


# quick plotting

data_df = pd.read_csv('G:\mri-results\infarct_volume_hyper.csv', sep=';', decimal=',')


def select_treatment_group(data, name: str):
    selected = data[data['Group'] == name]
    return selected


ctrl = select_treatment_group(data_df, 'Ctrl')
cyclo = select_treatment_group(data_df, 'Cyclo')
mph = select_treatment_group(data_df, 'MPH')

data = [ctrl['day03'], cyclo['day03'], mph['day03']]
plt.boxplot(data)
locs, labels = plt.xticks()
plt.xticks(locs, ['Ctrl', 'Cyclo', 'MPH'])
plt.ylim([0, 2000000])
plt.show()

data = [ctrl['day07'], cyclo['day07'], mph['day07']]
plt.boxplot(data)
locs, labels = plt.xticks()
plt.xticks(locs, ['Ctrl', 'Cyclo', 'MPH'])
plt.ylim([0, 2000000])
plt.show()

data = [ctrl['day21'], cyclo['day21'], mph['day21']]
plt.boxplot(data)
locs, labels = plt.xticks()
plt.xticks(locs, ['Ctrl', 'Cyclo', 'MPH'])
plt.ylim([0, 2000000])
plt.show()
