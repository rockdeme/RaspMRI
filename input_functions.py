import os
import numpy as np
import SimpleITK as sitk
from rbm.core.paras import PreParas
from rbm.core.utils import min_max_normalization, resample_img


def rescale_voxels(input_path):
    """
    The function takes the input nii/nii.gz file and rewrites the metadata to correct for the previous
    10x voxel upscale.
    :param input_path: input string of the file
    :return: SimpleITK image object
    """
    imgobj = sitk.ReadImage(input_path)
    keys = ['pixdim[1]', 'pixdim[2]', 'pixdim[3]']
    for key in keys:
        original_key = imgobj.GetMetaData(key)
        if original_key == '':
            raise Exception('Voxel parameter not set for file: ' + input_path)
        print('Old voxel dimension: ' + original_key)
        imgobj.SetMetaData(key, str(round(float(original_key)/10, 5)))
        print('New voxel dimension: ' + imgobj.GetMetaData(key))
    new_parameters = [param / 10 for param in list(imgobj.GetSpacing())]
    imgobj.SetSpacing(new_parameters)
    return imgobj


def preprocess(input):
    """
    The function takes either the imgobj or the input string and resamples/normalizes it as described in the Hsu et al.
    paper.
    :param input: SimpleITK image object or string
    :return: Rescaled image array
    """
    if str(type(input)) == "<class 'SimpleITK.SimpleITK.Image'>":
        imgobj = input
    elif type(input) == str:
        imgobj = sitk.ReadImage(input)
    else:
        raise Exception('Input is not defined correctly!')
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

    patch_dims = pre_paras.patch_dims
    strides = pre_paras.patch_strides
    length, col, row = img.shape
    it = 0
    z_dimension = len([*range(0, length-patch_dims[0]+1, strides[0])]) * \
                  len([*range(0, col-patch_dims[1]+1, strides[1])]) * \
                  len([*range(0, row-patch_dims[2]+1, strides[2])])

    z_stack = np.zeros([z_dimension, patch_dims[1], patch_dims[2]])
    for i in range(0, length-patch_dims[0]+1, strides[0]):
        for j in range(0, col-patch_dims[1]+1, strides[1]):
            for k in range(0, row-patch_dims[2]+1, strides[2]):
                cur_patch=img[i:i+patch_dims[0],j:j+patch_dims[1],k:k+patch_dims[2]]
                print(i,i+patch_dims[0],j,j+patch_dims[1],k,k+patch_dims[2])
                z_stack[it,:,:] = cur_patch
                it += 1
                print(it)
    return z_stack
