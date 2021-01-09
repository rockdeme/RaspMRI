import os
import numpy as np
import napari
import SimpleITK as sitk
from rbm.core.paras import PreParas, KerasParas
from rbm.core.utils import min_max_normalization, resample_img, dim_2_categorical


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

imgobj = sitk.ReadImage(input_path)
# re-sample to 0.1x0.1x0.1
resampled_imgobj = resample_img(imgobj, new_spacing=[0.1, 0.1, 0.1], interpolator=sitk.sitkLinear)
print('Image resampled!')
img_array = sitk.GetArrayFromImage(resampled_imgobj)
img = min_max_normalization(img_array)




patch_dims = pre_paras.patch_dims
label_dims = pre_paras.patch_label_dims
strides = pre_paras.patch_strides
n_class = pre_paras.n_class

# build new variables for output
length, col, row = img.shape
categorical_map = np.zeros((n_class, length, col, row), dtype=np.uint8)
likelihood_map = np.zeros((length, col, row), dtype=np.float32)
counter_map = np.zeros((length,col,row), dtype=np.float32)
length_step = int(patch_dims[0]/2)

"""-----predict the whole image from two directions, small to large and large to small----"""
img_array = np.zeros([(length-patch_dims[0]+1)*9,128,128])
it = 0

z_stack = len([*range(0, length-patch_dims[0]+1, strides[0])]) * len([*range(0, col-patch_dims[1]+1, strides[1])]) * len([*range(0, row-patch_dims[2]+1, strides[2])])

for i in range(0, length-patch_dims[0]+1, strides[0]):
    for j in range(0, col-patch_dims[1]+1, strides[1]):
        for k in range(0, row-patch_dims[2]+1, strides[2]):
            cur_patch=img[i:i+patch_dims[0],j:j+patch_dims[1],k:k+patch_dims[2]][:]
            print(i,i+patch_dims[0],j,j+patch_dims[1],k,k+patch_dims[2])
            img_array[it,:,:] = (cur_patch)
            it += 1
            print(it)



                .reshape([1,patch_dims[0],patch_dims[1],patch_dims[2]])


            if keras_paras.img_format == 'channels_last':
                cur_patch = np.transpose(cur_patch, (0, 2, 3, 1))

            cur_patch_output = cur_patch  # instead of predicting with the network I just assign the input as output

            # if there are multiple outputs
            if isinstance(cur_patch_output,list):
                cur_patch_output = cur_patch_output[keras_paras.outID]
            cur_patch_output = np.squeeze(cur_patch_output)
            cur_patch_out_label = cur_patch_output.copy()
            cur_patch_out_label[cur_patch_out_label >= keras_paras.thd] = 1
            cur_patch_out_label[cur_patch_out_label < keras_paras.thd] = 0

            middle = i + length_step
            cur_patch_out_label = dim_2_categorical(cur_patch_out_label,n_class)

            categorical_map[:, middle, j:j+label_dims[1], k:k+label_dims[2]] \
                = categorical_map[:, middle, j:j+label_dims[1], k:k+label_dims[2]] + cur_patch_out_label
            likelihood_map[middle, j:j+label_dims[1], k:k+label_dims[2]] \
                = likelihood_map[middle, j:j+label_dims[1], k:k+label_dims[2]] + cur_patch_output
            counter_map[middle, j:j+label_dims[1], k:k+label_dims[2]] += 1

for i in range(length, patch_dims[0]-1, -strides[0]):
    for j in range(col, patch_dims[1]-1, -strides[1]):
        for k in range(row, patch_dims[2]-1, -strides[2]):

            cur_patch=img[i-patch_dims[0]:i,
                          j-patch_dims[1]:j,
                          k-patch_dims[2]:k][:].reshape([1, patch_dims[0], patch_dims[1], patch_dims[2]])
            if keras_paras.img_format == 'channels_last':
                cur_patch = np.transpose(cur_patch, (0, 2, 3, 1))

            cur_patch_output = cur_patch # instead of predicting with the network I just assign the input as output

            if isinstance(cur_patch_output,list):
                cur_patch_output = cur_patch_output[keras_paras.outID]
            cur_patch_output = np.squeeze(cur_patch_output)

            cur_patch_out_label = cur_patch_output.copy()
            cur_patch_out_label[cur_patch_out_label >= keras_paras.thd] = 1
            cur_patch_out_label[cur_patch_out_label < keras_paras.thd] = 0

            middle = i - patch_dims[0] + length_step
            cur_patch_out_label = dim_2_categorical(cur_patch_out_label,n_class)
            categorical_map[:, middle, j-label_dims[1]:j, k-label_dims[2]:k] = \
                categorical_map[:, middle, j-label_dims[1]:j, k-label_dims[2]:k] + cur_patch_out_label
            likelihood_map[middle, j-label_dims[1]:j, k-label_dims[2]:k] = \
                likelihood_map[middle, j-label_dims[1]:j, k-label_dims[2]:k] + cur_patch_output
            counter_map[middle, j-label_dims[1]:j, k-label_dims[2]:k] += 1

label_map = np.zeros([length,col,row],dtype=np.uint8)
for idx in range(0,length):
    cur_slice_label = np.squeeze(categorical_map[:, idx,].argmax(axis=0))
    label_map[idx,] = cur_slice_label

counter_map = np.maximum(counter_map, 10e-10)
likelihood_map = np.divide(likelihood_map,counter_map)

%gui qt
viewer = napari.Viewer()
viewer.add_image(counter_map)
viewer.add_image(categorical_map)
viewer.add_image(likelihood_map)
viewer.add_image(cur_patch)
viewer.add_image(img_array)