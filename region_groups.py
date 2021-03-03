import numpy as np
import glob
import pandas as pd
import SimpleITK as sitk
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from utils import cerebellum_normalization, get_voxels
from sklearn.preprocessing import LabelEncoder


def get_regions(regions):
    indices = []
    intensities = []
    areas = []
    for region in regions:
        indices.append(region.label)
        intensities.append(region.mean_intensity)
        areas.append(region.area)
    return pd.DataFrame(data = {'Intensity': intensities, 'Area': areas}, index=indices)


cerebellum_int = [221, 222, 341, 342, 991, 992, 1001, 1002, 1011, 1012]

day_list = ['day03', 'day07', 'day21']
data_df = pd.DataFrame()
region_labels = pd.read_csv(
    "G:\SIGMA\SIGMA_Rat_Brain_Atlases\SIGMA_Anatomical_Atlas\SIGMA_Anatomical_Brain_Atlas_ListOfStructures.csv",
    sep = ',')
le = LabelEncoder()
region_labels['le r territories'] = (le.fit_transform(region_labels['Territories'])+1) * 2000
region_labels['le l territories'] = region_labels['le r territories'] + 1

for day in day_list:
    output_df = region_labels.copy()
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

        for territory in np.unique(region_labels['le r territories']):
            territory_df = region_labels[region_labels['le r territories'] == territory]
            for reg in territory_df['Right Hemisphere Label']:
                atlas_array[atlas_array == reg] = territory

        for territory in np.unique(region_labels['le l territories']):
            territory_df = region_labels[region_labels['le l territories'] == territory]
            for reg in territory_df['Left Hemisphere Label']:
                atlas_array[atlas_array == reg] = territory

        volume_array = sitk.ReadImage(volume)
        volume_array = sitk.GetArrayFromImage(volume_array)
        template_array = sitk.ReadImage(template)
        template_array = sitk.GetArrayFromImage(template_array)

        volume_regions = regionprops(atlas_array, volume_array)
        # template_regions = regionprops(atlas_array, template_array)

        regions_df = get_regions(volume_regions)

        rat = atlas.split('\\')[1][:-10] + '_masked-img.nii'

        rat_df = region_labels.merge(regions_df, right_index=True, left_on='le l territories')\
            .drop(['Original Atlas'], axis = 1).rename(columns = {'Intensity': 'Left Hemisphere Intensity', 'Area': 'Left '
                                                                  'Hemisphere Area'})
        rat_df = rat_df.merge(regions_df, right_index=True, left_on='le r territories')\
            .rename(columns={'Intensity': 'Right Hemisphere Intensity', 'Area': 'Right Hemisphere Area'})
        rat_df = rat_df.iloc[:,-4:].add_prefix(rat + '_')
        output_df = output_df.merge(rat_df, left_index=True, right_index=True)
        print(rat + ' is completed!')
    output_df.to_csv(f'G:/mri-results/t2_data_{day}_territories.csv', sep = ';')



df = pd.read_csv(f'G:/mri-results/t2_data_day03_territories.csv', sep=';', index_col=0)
df = df.drop(['Original Atlas', 'Left Hemisphere Label', 'Right Hemisphere Label',
       'Matter', 'System', 'Region of interest',], axis=1)
df = df.drop_duplicates()

cerebellum = df.loc[df['Territories'] == 'Cerebellum']

for column in df.columns:
    pass

normalized_df.to_csv("G:/mri-results/t2_data_day03_normalized.csv", sep = ';')