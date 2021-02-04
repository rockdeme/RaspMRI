import SimpleITK as sitk
import glob
import napari
from rbm.core.utils import resample_img
from input_functions import rescale_voxels
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []


# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()


# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.show()


# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# metric_values list.
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return 2.0 * intersection / (np.sum(y_true_f) + np.sum(y_pred_f))


def create_binary_volume(volume):
    volume_copy = np.copy(volume)
    volume_copy[volume_copy > 0] = 1
    return volume_copy


def affine_registration(fixed_image, moving_image, atlas, plot = False):
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
    atlas_resampled = sitk.Resample(atlas, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())

    fixed_binary = create_binary_volume(sitk.GetArrayFromImage(fixed_image))
    moving_binary = create_binary_volume(sitk.GetArrayFromImage(moving_resampled))
    dice_score = dice_coef_np(fixed_binary, moving_binary)

    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return moving_resampled, atlas_resampled, dice_score




files = glob.glob('G:/masked-brains/*.nii')
template_path = 'G:/SIGMA/cranial_template.nii'
atlas_path = 'G:/SIGMA/SIGMA_Rat_Brain_Atlases/SIGMA_Anatomical_Atlas/SIGMA_Anatomical_Brain_Atlas.nii'
atlas = sitk.ReadImage(atlas_path)
atlas_array = sitk.GetArrayFromImage(atlas)
t2wi = sitk.ReadImage(files[0],  sitk.sitkFloat32)
template = sitk.ReadImage(template_path,  sitk.sitkFloat32)
t2wi_rescaled = rescale_voxels(t2wi)
t2wi_rescaled_resampled = resample_img(t2wi_rescaled, new_spacing=template.GetSpacing(), interpolator=sitk.sitkLinear)
fixed_image = t2wi_rescaled_resampled
moving_image = template

m_res, atlas_res, dice = affine_registration(fixed_image, moving_image, atlas)












#todo method 1
initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                      moving_image,
                                                      sitk.AffineTransform(3),
                                                      sitk.CenteredTransformInitializerFilter.MOMENTS)

moving_initial = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())


registration_method = sitk.ImageRegistrationMethod()

registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
# Scale the step size differently for each parameter, this is critical!!!
registration_method.SetOptimizerScalesFromPhysicalShift()

registration_method.SetInitialTransform(initial_transform, inPlace=False)

registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

final_transform_v1 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                 sitk.Cast(moving_image, sitk.sitkFloat32))


moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform_v1, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))



#todo method 2
registration_method = sitk.ImageRegistrationMethod()

registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                  numberOfIterations=100)  # , estimateLearningRate=registration_method.EachIteration)
registration_method.SetOptimizerScalesFromPhysicalShift()

final_transform = sitk.AffineTransform(initial_transform)
registration_method.SetInitialTransform(final_transform)
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                               update_multires_iterations)
registration_method.AddCommand(sitk.sitkIterationEvent,
                               lambda: plot_values(registration_method))

final_transform_v2 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),sitk.Cast(moving_image, sitk.sitkFloat32))

moving_resampled_2 = sitk.Resample(moving_image, fixed_image, final_transform_v2, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

asd = sitk.GetArrayFromImage(atlas_res)

# %gui qt magic command
viewer = napari.Viewer()
viewer.add_labels(sitk.GetArrayFromImage(atlas))
viewer.add_labels(sitk.GetArrayFromImage(atlas_res))

viewer.add_image(sitk.GetArrayFromImage(fixed_image))
viewer.add_image(sitk.GetArrayFromImage(moving_initial))
viewer.add_image(sitk.GetArrayFromImage(moving_resampled))
viewer.add_image(sitk.GetArrayFromImage(moving_image))
viewer.add_image(sitk.GetArrayFromImage(moving_resampled_2))
