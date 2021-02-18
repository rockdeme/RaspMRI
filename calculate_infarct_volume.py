import glob
import SimpleITK as sitk

def get_voxels(regions, template_regions):
    counter = 0
    for region, template_region in zip(regions, template_regions):
        template_mean = np.mean(template_region.intensity_image[template_region.intensity_image > 0])
        template_std = np.std(template_region.intensity_image[template_region.intensity_image > 0])
        mean_plus_std = np.nonzero(region.intensity_image[region.intensity_image >
                                                          (template_mean + 2 * template_std)])[0].size
        mean_minus_std = np.nonzero(region.intensity_image[(template_mean - 2 * template_std) >
                                                           region.intensity_image])[0].size
        selected_voxels = mean_plus_std + mean_minus_std
        counter += selected_voxels
    return counter


atlas_path = glob.glob('G:/coregistered-files/day21/*_atlas.nii')
volume_path = glob.glob('G:/coregistered-files/day21/*_remasked-volume.nii')
template_path = glob.glob('G:/coregistered-files/day21/*_template.nii')

cerebellum_int = [221, 222, 341, 342, 991, 992, 1001, 1002, 1011, 1012]


for atlas, volume, template in zip(atlas_path, volume_path, template_path):
    atlas_array = sitk.ReadImage(atlas)
    atlas_array = sitk.GetArrayFromImage(atlas_array)




regions = regionprops(atlas_res, original_volume_array)
template_regions = regionprops(atlas_res, template_res)