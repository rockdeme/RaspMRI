import numpy as np
import SimpleITK as sitk
from rbm.core.paras import PreParas
from rbm.core.utils import min_max_normalization, resample_img
from skimage import measure


def rescale_voxel_size(input_path):
    """
    Takes the input nii/nii.gz file and rewrites the metadata to correct for the previous 10x voxel upscale.
    :param input_path: input string of the file
    :return: SimpleITK image object
    """
    if str(type(input_path)) == "<class 'SimpleITK.SimpleITK.Image'>":
        imgobj = input_path
    elif type(input_path) == str:
        imgobj = sitk.ReadImage(input_path)
    else:
        raise Exception('Input is not defined correctly!')
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
    Takes either the imgobj or the input string and resamples/normalizes it as described in the Hsu et al.
    paper.
    :param input: SimpleITK image object or string
    :return: Rescaled image array and the image object
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
    return img, resampled_imgobj


def normalize_img(input):
    """
    Takes either the imgobj or the input string and normalizes it as described in the Hsu et al.
    paper.
    :param input: SimpleITK image object or string
    :return: Rescaled image array and the image object
    """
    if str(type(input)) == "<class 'SimpleITK.SimpleITK.Image'>":
        imgobj = input
    elif type(input) == str:
        imgobj = sitk.ReadImage(input)
    else:
        raise Exception('Input is not defined correctly!')
    img_array = sitk.GetArrayFromImage(imgobj)
    img = min_max_normalization(img_array)
    img = sitk.GetImageFromArray(img)
    img.SetSpacing(imgobj.GetSpacing())
    return img


def create_training_examples(img):
    """
    Creates 128x128 fragments with 32 px strides from each image of a Z-stack.
    :param img: input Z-stack as numpy array
    :return: 3D numpy array
    """
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


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return 2.0 * intersection / (np.sum(y_true_f) + np.sum(y_pred_f))


def sort_array(array):
    dat = measure.label(array)
    props = measure.regionprops(dat)
    area_dict = {}
    i = 1
    for area in props:
        area_dict.update({area.area: i})
        i += 1
    dat[dat != area_dict[max(area_dict, key=int)]] = 0
    dat[dat != 0] = 1
    return dat


def get_voxels(regions, template_regions):
    counter = 0
    for region, template_region in zip(regions, template_regions):
        template_mean = np.mean(template_region.intensity_image[template_region.intensity_image > 0])
        template_std = np.std(template_region.intensity_image[template_region.intensity_image > 0])
        mean_plus_std = np.nonzero(region.intensity_image[region.intensity_image >
                                                          (template_mean + (2 * template_std))])[0].size
        # mean_minus_std = np.nonzero(region.intensity_image[(template_mean - (2 * template_std)) >
        #                                                    region.intensity_image])[0].size
        # selected_voxels = mean_plus_std + mean_minus_std
        selected_voxels = mean_plus_std
        counter += selected_voxels
    return counter


def cerebellum_normalization(atlas_array, array, cerebellum_int):
    atlas_copy = atlas_array.copy()
    for region in cerebellum_int:
        atlas_copy[atlas_copy == region] = 1
    atlas_copy[atlas_copy != 1] = 0
    cer_intensity = np.mean(array[atlas_copy > 0])
    array_norm = array / cer_intensity
    return array_norm
