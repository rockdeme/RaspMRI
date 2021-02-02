import SimpleITK as sitk
import glob
import napari
from rbm.core.utils import resample_img
from input_functions import rescale_voxels
import matplotlib.pyplot as plt
from IPython.display import clear_output
import SimpleITK


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

initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                      moving_image,
                                                      sitk.Euler3DTransform(),
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








# %gui qt magic command
viewer = napari.Viewer()
viewer.add_labels(atlas_array)
viewer.add_labels(likelihood_array_copy)

viewer.add_image(sitk.GetArrayFromImage(fixed_image))
viewer.add_image(sitk.GetArrayFromImage(moving_initial))
viewer.add_image(sitk.GetArrayFromImage(moving_resampled))
viewer.add_image(sitk.GetArrayFromImage(moving_image))
