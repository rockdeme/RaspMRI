import os
import numpy as np
import napari
import tensorflow as tf
from keras.models import load_model
import SimpleITK as sitk
from rbm.core.paras import PreParas, KerasParas
from rbm.core.dice import dice_coef, dice_coef_loss
from rbm.core.utils import min_max_normalization, resample_img
from rbm.core.eval import out_LabelHot_map_2D

# Let's make sure we see some GPUs
tf.config.list_physical_devices('GPU')

# This is the main function written by our korean friends. It loads up the pretrained keras model, reads our input image
# using simpleITK and resamples it.
def brain_seg_prediction(input_path, output_path, pre_paras, keras_paras):
    # load model
    seg_net = load_model(keras_paras.model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    print('U-Net model loaded!')
    imgobj = sitk.ReadImage(input_path)
    # re-sample to 0.1x0.1x0.1
    resampled_imgobj = resample_img(imgobj, new_spacing=[0.1, 0.1, 0.1], interpolator=sitk.sitkLinear)
    print('Image resampled!')
    img_array = sitk.GetArrayFromImage(resampled_imgobj)
    normed_array = min_max_normalization(img_array)
    out_label_map, out_likelihood_map = out_LabelHot_map_2D(normed_array, seg_net, pre_paras, keras_paras)
    out_label_img = sitk.GetImageFromArray(out_label_map.astype(np.uint8))
    out_label_img.CopyInformation(resampled_imgobj)
    resampled_label_map = resample_img(out_label_img, new_spacing=imgobj.GetSpacing(), new_size=imgobj.GetSize(), interpolator=sitk.sitkNearestNeighbor)
    # Save the results
    sitk.WriteImage(resampled_label_map, output_path)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
keras_paras.model_path = os.path.join(os.getcwd(), 'rbm', 'scripts', 'rat_brain-2d_unet.hdf5')



input_path = 'C:/mri/sub-001_ses-1_anat_sub-001_ses-1_acq-RARE_T2w.nii.gz'
output_path = 'C:/mri/demo-mask.nii'

%load_ext line_profiler
%lprun -f brain_seg_prediction brain_seg_prediction(input_path, output_path, pre_paras, keras_paras)


brain_seg_prediction(input_path, output_path, pre_paras, keras_paras)

#todo
imgobj = sitk.ReadImage(input_path)
# re-sample to 0.1x0.1x0.1
resampled_imgobj = resample_img(imgobj, new_spacing=[0.1, 0.1, 0.1], interpolator=sitk.sitkLinear)
print('Image resampled!')
img_array = sitk.GetArrayFromImage(resampled_imgobj)
normed_array = min_max_normalization(img_array)
img = normed_array

with napari.gui_qt():
    %gui qt
    viewer = napari.Viewer()
    viewer.add_image(img_array)
%gui qt
viewer = napari.Viewer()
viewer.add_image(img_array)

# basic way to create a dependency list and how to install it
# pip freeze > requirements.txt
# cat requirements.txt
# pip install -r requirements.txt
