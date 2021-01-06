import os
from RodentMRISkullStripping.rbm.core.paras import PreParas, KerasParas
from RodentMRISkullStripping.rbm.core.dice import dice_coef, dice_coef_loss
from RodentMRISkullStripping.rbm.core.utils import min_max_normalization, resample_img
from RodentMRISkullStripping.rbm.core.eval import out_LabelHot_map_2D
from keras.models import load_model
from RodentMRISkullStripping.rbm import __version__
import SimpleITK as sitk
import numpy as np
import argparse
import tensorflow as tf
tf.config.list_physical_devices('GPU')


def brain_seg_prediction(input_path, output_path,
                         pre_paras, keras_paras):
    # load model
    seg_net = load_model(keras_paras.model_path,
                         custom_objects={'dice_coef_loss': dice_coef_loss,
                                         'dice_coef': dice_coef})

    imgobj = sitk.ReadImage(input_path)

    # re-sample to 0.1x0.1x0.1
    resampled_imgobj = resample_img(imgobj,
                                    new_spacing=[0.1, 0.1, 0.1],
                                    interpolator=sitk.sitkLinear)

    img_array = sitk.GetArrayFromImage(resampled_imgobj)
    normed_array = min_max_normalization(img_array)
    out_label_map, out_likelihood_map = out_LabelHot_map_2D(normed_array,
                                                            seg_net,
                                                            pre_paras,
                                                            keras_paras)

    out_label_img = sitk.GetImageFromArray(out_label_map.astype(np.uint8))
    out_label_img.CopyInformation(resampled_imgobj)

    resampled_label_map = resample_img(out_label_img,
                                       new_spacing=imgobj.GetSpacing(),
                                       new_size=imgobj.GetSize(),
                                       interpolator=sitk.sitkNearestNeighbor)
    # Save the results
    sitk.WriteImage(resampled_label_map, output_path)



