import numpy as np
import glob
import pandas as pd
import SimpleITK as sitk
from skimage.measure import regionprops
from utils import cerebellum_normalization
import matplotlib.pyplot as plt

def get_region_histogram(regions):
    output = pd.DataFrame()
    brain = regions[0].intensity_image

    brain_pixels = brain.flatten()
    brain_pixels = brain_pixels[brain_pixels != 0.0]
    max_val = np.max(brain_pixels)
    brain_hist = np.histogram(brain_pixels, bins=np.linspace(0,max_val,11))[0]
    output['Voxel count'] = brain_hist

    return output



day_list = ['day03', 'day07', 'day21']
data_df = pd.DataFrame()
region_labels = pd.read_csv(
    "G:\SIGMA\SIGMA_Rat_Brain_Atlases\SIGMA_Anatomical_Atlas\SIGMA_Anatomical_Brain_Atlas_ListOfStructures.csv",
    sep = ',')
cerebellum_int = [221, 222, 341, 342, 991, 992, 1001, 1002, 1011, 1012]

for day in day_list:

    print(day)
    atlas_path = glob.glob('G:/coregistered-files/' + day + '/*_atlas.nii')
    volume_path = glob.glob('G:/coregistered-files/' + day + '/*_remasked-volume.nii')
    template_path = glob.glob('G:/coregistered-files/' + day + '/*_template.nii')
    i = 1
    list_len = len(atlas_path)
    data = {}
    output_df = pd.DataFrame(np.arange(0, 10, 1))

    for atlas, volume, template in zip(atlas_path, volume_path, template_path):
        print(f'File {i}/{list_len}')
        i += 1

        atlas_array = sitk.ReadImage(atlas)
        atlas_array = sitk.GetArrayFromImage(atlas_array)
        volume_array = sitk.ReadImage(volume)
        volume_array = sitk.GetArrayFromImage(volume_array)

        normed_volume_array = cerebellum_normalization(atlas_array, volume_array, cerebellum_int)

        atlas_array[atlas_array != 0] = 1

        volume_regions = regionprops(atlas_array, normed_volume_array)

        histo_df = get_region_histogram(volume_regions)

        rat = atlas.split('\\')[1][:-10] + '_masked-img.nii'
        rat_df = histo_df.iloc[:,-4:].add_prefix(rat + '_')
        output_df = output_df.merge(rat_df, left_index=True, right_index=True)
        print(rat + ' is completed!')

    output_df.to_csv(f'G:/mri-results/t2_data_{day}_intensity_profile.csv', sep = ';')
