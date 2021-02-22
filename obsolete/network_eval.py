from rbm.core.paras import PreParas, KerasParas
from rbm.core.dice import dice_coef, dice_coef_loss
from rbm.core.utils import resample_img
from rbm.core.eval import out_LabelHot_map_2D
from keras.models import load_model
import SimpleITK as sitk
import numpy as np
from utils import preprocess, rescale_voxel_size
from rbm.core.dice import dice_coef_np
import matplotlib.pyplot as plt

# Some test code to save the network's output properly and to load a single likelihood array, label and ground truth to
# to calculate the dice score.

# Default Parameters Preparation
pre_paras = PreParas()
pre_paras.patch_dims = [1, 128, 128]
pre_paras.patch_label_dims = [1, 128, 128]
pre_paras.patch_strides = [1, 32, 32]
pre_paras.n_class = 2

# Parameters for Keras model
keras_paras = KerasParas()
keras_paras.outID = 0
keras_paras.thd = 0.5
keras_paras.loss = 'dice_coef_loss'
keras_paras.img_format = 'channels_last'
keras_paras.model_path = '../rbm/scripts/rat_brain-2d_unet.hdf5'

# load model
seg_net = load_model(keras_paras.model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

imgobj = sitk.ReadImage('C:/mri/VI_2_a_26_21days_after_stroke_t2_cor_30.4.1.nii.gz')
img_rescaled = rescale_voxel_size('C:/mri/VI_2_a_26_21days_after_stroke_t2_cor_30.4.1.nii.gz')
normed_array, resampled_imgobj = preprocess(img_rescaled)

out_label_map, out_likelihood_map = out_LabelHot_map_2D(normed_array, seg_net, pre_paras, keras_paras)

out_label_img = sitk.GetImageFromArray(out_label_map.astype(np.uint8))
out_likelihood_img = sitk.GetImageFromArray(out_likelihood_map.astype(np.float))

resampled_label_map = resample_img(out_label_img, new_spacing=(1.3671875, 1.3671875, 1.0), new_size=imgobj.GetSize(), interpolator=sitk.sitkNearestNeighbor)
resampled_likelihood_img = resample_img(out_likelihood_img, new_spacing=(1.3671875, 1.3671875, 1.0), new_size=imgobj.GetSize(), interpolator=sitk.sitkNearestNeighbor)

sitk.WriteImage(resampled_label_map, 'C:/mri/label.nii')
sitk.WriteImage(resampled_likelihood_img, 'C:/mri/likelihood.nii')

ground_truth = sitk.ReadImage('C:/mri/VI_2_a_26_21days_after_stroke_t2_cor_30.4.1-whole_brain.nii.gz')
likelihood = sitk.ReadImage('C:/mri/likelihood.nii')
label = sitk.ReadImage('C:/mri/label.nii')
ground_truth_array = sitk.GetArrayFromImage(ground_truth)
label_array = sitk.GetArrayFromImage(label)
likelihood_array = sitk.GetArrayFromImage(likelihood)

dice_list = []
for i in np.arange(0, 1, 0.01):
    likelihood_array_copy = np.copy(likelihood_array)
    likelihood_array_copy[i > likelihood_array_copy] = 0
    likelihood_array_copy[likelihood_array_copy > 0] = 1
    dice = dice_coef_np(ground_truth_array, likelihood_array_copy)
    dice_list.append(dice)

plt.hist(ground_truth_array.flatten())
plt.show()

plt.plot(dice_list, color = 'indigo', linewidth=3.0)
plt.ylim([0,1])
plt.title('Changes in dice score by varying the likelihood map threshold')
plt.ylabel('Dice Score')
plt.grid()
plt.xlabel('Likelihood Threshold')
plt.show()


dice_coef_np(ground_truth_array, label_array)

# %gui qt magic command
viewer = napari.Viewer()
viewer.add_labels(out_label_map)
viewer.add_image(out_likelihood_map)
viewer.add_image(normed_array)

