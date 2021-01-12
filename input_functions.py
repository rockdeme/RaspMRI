import os
import numpy as np
import napari
import SimpleITK as sitk
from rbm.core.paras import PreParas, KerasParas
from rbm.core.utils import min_max_normalization, resample_img, dim_2_categorical


def preprocess(input_path):
    imgobj = sitk.ReadImage(input_path)
    # re-sample to 0.1x0.1x0.1
    resampled_imgobj = resample_img(imgobj, new_spacing=[0.1, 0.1, 1], interpolator=sitk.sitkLinear)
    print('Image resampled!')
    img_array = sitk.GetArrayFromImage(resampled_imgobj)
    img = min_max_normalization(img_array)
    return img


def create_training_examples(img):
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

    patch_dims = pre_paras.patch_dims
    strides = pre_paras.patch_strides
    length, col, row = img.shape
    it = 0
    z_dimension = len([*range(0, length-patch_dims[0]+1, strides[0])]) * len([*range(0, col-patch_dims[1]+1, strides[1])]) * len([*range(0, row-patch_dims[2]+1, strides[2])])
    z_stack = np.zeros([z_dimension, patch_dims[1], patch_dims[2]])
    for i in range(0, length-patch_dims[0]+1, strides[0]):
        for j in range(0, col-patch_dims[1]+1, strides[1]):
            for k in range(0, row-patch_dims[2]+1, strides[2]):
                cur_patch=img[i:i+patch_dims[0],j:j+patch_dims[1],k:k+patch_dims[2]]
                print(i,i+patch_dims[0],j,j+patch_dims[1],k,k+patch_dims[2])
                z_stack[it,:,:] = cur_patch
                it += 1
                print(it)
    #kek = np.expand_dims(z_stack, axis = 0)
    #if keras_paras.img_format == 'channels_last':
    #    z_stack = np.transpose(z_stack, (0, 2, 3, 1))
    return z_stack
