import SimpleITK as sitk
import glob
import napari
from rbm.core.utils import resample_img
from utils import rescale_voxel_size
from pipeline.registration_utilities import start_plot, end_plot, update_multires_iterations, plot_values
import numpy as np
from skimage.measure import regionprops
import pandas as pd


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return 2.0 * intersection / (np.sum(y_true_f) + np.sum(y_pred_f))


def create_binary_volume(volume):
    volume_copy = np.copy(volume)
    volume_copy[volume_copy > 0] = 1
    return volume_copy


def affine_registration(fixed_image, moving_image, atlas, plot=False):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.AffineTransform(3),
                                                          sitk.CenteredTransformInitializerFilter.MOMENTS)
    # we can also output the initial transform if needed so I won't delete these two lines for now
    # moving_initial = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0,
    #                                moving_image.GetPixelID())
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    if plot:
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())
    atlas_resampled = sitk.Resample(atlas, fixed_image, final_transform, sitk.sitkNearestNeighbor, 0.0,
                                    moving_image.GetPixelID())

    fixed_binary = create_binary_volume(sitk.GetArrayFromImage(fixed_image))
    moving_binary = create_binary_volume(sitk.GetArrayFromImage(moving_resampled))
    dice_score = dice_coef_np(fixed_binary, moving_binary)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return moving_resampled, atlas_resampled, dice_score


def get_regions(regions):
    indeces = []
    intensities = []
    areas = []
    for region in regions:
        indeces.append(region.label)
        intensities.append(region.mean_intensity)
        areas.append(region.area)
    return pd.DataFrame(data = {'Intensity': intensities, 'Area': areas}, index=indeces)


def mask_original_volume(raw_array, atlas_resampled):
    array = np.copy(raw_array)
    if str(type(atlas_resampled)) == "<class 'SimpleITK.SimpleITK.Image'>":
        atlas_resampled = sitk.GetArrayFromImage(atlas_resampled)
        atlas_resampled = atlas_res.astype(int)
    elif str(type(atlas_resampled)) == "<class 'numpy.ndarray'>":
        pass
    else:
        raise Exception('Input is not defined correctly!')
    atlas_mask = atlas_resampled > 0
    array[~atlas_mask] = 0
    return array

template_path = 'G:/SIGMA/cranial_template_ex_vivo.nii'
atlas_path = 'G:/SIGMA/cranial_atlas.nii'

mask_path = 'G:/masked-brains/day00\\'
files = glob.glob('G:/mri-dataset/diffusion_0h/*/immediately_B0.nii.gz')
region_labels = pd.read_csv(
    "G:\SIGMA\SIGMA_Rat_Brain_Atlases\SIGMA_Anatomical_Atlas\SIGMA_Anatomical_Brain_Atlas_ListOfStructures.csv",
    sep = ',')
output_df = region_labels.copy()
# mri_volume_path = [item for item in files if 'whole_brain' not in item]
# mri_volume_path = [item for item in mri_volume_path if 'bias2' not in item]

atlas = sitk.ReadImage(atlas_path)
atlas_array = sitk.GetArrayFromImage(atlas)
template = sitk.ReadImage(template_path,  sitk.sitkFloat32)

files.sort()
list_len = len(files)
i = 1
for mri_volume in files:
    mask = mask_path + mri_volume.split('\\')[1] + '_masked-img.nii'
    print('---------------------------------------------------------')
    print(f'File {i}/{list_len}')
    i += 1
    print(mask)
    print(mri_volume)
    t2wi = sitk.ReadImage(mask, sitk.sitkFloat32)
    t2wi_rescaled = rescale_voxel_size(t2wi)
    t2wi_rescaled_resampled = resample_img(t2wi_rescaled, new_spacing=template.GetSpacing(), interpolator=sitk.sitkLinear)
    fixed_image = t2wi_rescaled_resampled
    moving_image = template
    m_res, atlas_res, dice = affine_registration(fixed_image, moving_image, atlas)
    original_volume = sitk.ReadImage(mri_volume)
    original_volume_rescaled = rescale_voxel_size(original_volume)
    original_volume_rescaled_resampled = resample_img(original_volume_rescaled, new_spacing=template.GetSpacing(),
                                                      interpolator=sitk.sitkLinear)
    original_volume_array = sitk.GetArrayFromImage(original_volume_rescaled_resampled)
    atlas_res = sitk.GetArrayFromImage(atlas_res)
    atlas_res = atlas_res.astype(int)
    template_res = sitk.GetArrayFromImage(m_res)

    original_volume_array_masked = mask_original_volume(original_volume_array, atlas_res)
    original_volume_masked = sitk.GetImageFromArray(original_volume_array_masked)
    original_volume_masked.SetSpacing(template.GetSpacing())

    atlas_res_imgobj = sitk.GetImageFromArray(atlas_res.astype(np.uint16))
    atlas_res_imgobj.SetSpacing(template.GetSpacing())
    template_res_imgobj = m_res
    template_res_imgobj.SetSpacing(template.GetSpacing())
    output_string = 'G:/coregistered-files/day00/' + mri_volume.split('\\')[1]
    sitk.WriteImage(atlas_res_imgobj, (output_string + '_atlas.nii'))
    sitk.WriteImage(original_volume_masked, output_string + '_remasked-volume.nii')
    sitk.WriteImage(template_res_imgobj, output_string + '_template.nii')

    regions = regionprops(atlas_res, original_volume_array)
    template_regions = regionprops(atlas_res, template_res)

    regions_df = get_regions(regions)

    rat_df = region_labels.merge(regions_df, right_index=True, left_on='Left Hemisphere Label')\
        .drop(['Original Atlas'], axis = 1).rename(columns = {'Intensity': 'Left Hemisphere Intensity', 'Area': 'Left '
                                                              'Hemisphere Area'})
    rat_df = rat_df.merge(regions_df, right_index=True, left_on='Right Hemisphere Label')\
        .rename(columns={'Intensity': 'Right Hemisphere Intensity', 'Area': 'Right Hemisphere Area'})
    rat_df = rat_df.iloc[:,-4:].add_prefix(mask.split('\\')[-1] + '_')
    output_df = output_df.merge(rat_df, left_index=True, right_index=True)
    print(mask.split('\\')[-1] + ' is completed!')

output_df.to_csv("G:/mri-results/dwi_data_day00.csv", sep = ';')

normalized_df = output_df
cerebellum = output_df.loc[output_df['Territories'] == 'Cerebellum']
for mask in mri_volume_path:
    filtered = cerebellum.filter(like = mask.split('\\')[1])
    left = np.sum(filtered.iloc[:,0] * (filtered.iloc[:,1] / np.sum(filtered.iloc[:,1])))
    right = np.sum(filtered.iloc[:,2] * (filtered.iloc[:,3] / np.sum(filtered.iloc[:,3])))
    filtered_data = output_df.filter(like = mask.split('\\')[1])
    normalized_df[filtered_data.columns[0]] = filtered_data.iloc[:, 0] / left
    normalized_df[filtered_data.columns[2]] = filtered_data.iloc[:, 2] / right

normalized_df.to_csv("G:/mri-results/t2_data_day03_normalized.csv", sep = ';')

# %gui qt magic command
viewer = napari.Viewer()
viewer.add_labels(sitk.GetArrayFromImage(atlas))
viewer.add_labels((atlas_res))
viewer.add_image(sitk.GetArrayFromImage(m_res))
viewer.add_image(sitk.GetArrayFromImage(original_volume_rescaled_resampled))
viewer.add_image(original_volume_array)

volume = sitk.ReadImage('G:/coregistered-files/day21/VI_2_a_26_21days_after_stroke_t2_cor_30_remasked-volume.nii',  sitk.sitkFloat32)
template = sitk.ReadImage('G:/coregistered-files/day21/VI_2_a_26_21days_after_stroke_t2_cor_30_template.nii',  sitk.sitkFloat32)
viewer.add_image(sitk.GetArrayFromImage(volume))
viewer.add_image(sitk.GetArrayFromImage(template))