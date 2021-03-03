import numpy as np
import glob
import pandas as pd
import SimpleITK as sitk
from skimage.measure import regionprops


def get_region_histogram(regions):
    for region in regions:
        indices.append(region.label)
        intensities.append(region.mean_intensity)
        areas.append(region.area)
    return pd.DataFrame(data = {'Intensity': intensities, 'Area': areas}, index=indices)


day_list = ['day03', 'day07', 'day21']
data_df = pd.DataFrame()
region_labels = pd.read_csv(
    "G:\SIGMA\SIGMA_Rat_Brain_Atlases\SIGMA_Anatomical_Atlas\SIGMA_Anatomical_Brain_Atlas_ListOfStructures.csv",
    sep = ',')

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
        volume_array = sitk.ReadImage(volume)
        volume_array = sitk.GetArrayFromImage(volume_array)

        region_ints = np.unique(atlas_array)
        region_ints = region_ints[region_ints != 0]
        ipsilateral_h = region_ints[region_ints % 2 == 0]
        contralateral_h = region_ints[region_ints % 2 == 1]

        for region in ipsilateral_h:
            atlas_array[atlas_array == region] = 1
        for region in contralateral_h:
            atlas_array[atlas_array == region] = 2

        volume_regions = regionprops(atlas_array, volume_array)

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


df_list = glob.glob('G:/mri-results/*_territories.csv')
for path in df_list:
    df = pd.read_csv(path, sep=';', index_col=0)
    df = df.drop(['Original Atlas', 'Left Hemisphere Label', 'Right Hemisphere Label',
           'Matter', 'System', 'Region of interest',], axis=1)
    df = df.drop_duplicates()

    animals = [name.split('nii_')[0] for name in df.iloc[:, 3:].columns]
    animals = np.unique(animals)

    cerebellum = df.loc[df['Territories'] == 'Cerebellum']
    normalized_df = df.copy()
    for animal in animals:
        filtered = df.filter(like=animal)
        normalized_df[filtered.columns[0]] = filtered.iloc[:, 0] / filtered.iloc[1, 0]
        normalized_df[filtered.columns[2]] = filtered.iloc[:, 2] / filtered.iloc[1, 2]
    filename = path.split('\\')[1][:-4]
    normalized_df.to_csv(f"G:/mri-results/{filename}_normalized.csv", sep = ';')

pixels = volume_regions[0].intensity_image.flatten()
pixels = pixels[pixels != 0]
plt.hist(pixels, bins=300)
plt.show()