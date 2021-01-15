import SimpleITK as sitk

from input_functions import preprocess, rescale_voxels, create_training_examples

input_path = 'C:/mri/X.nii.gz'
input_rescaled = 'C:/mri/X.nii'

input_test = 'C:/mri/rat.nii.gz'

img = rescale_voxels(input_path)
imgobj_rescaled = sitk.ReadImage(input_rescaled)
imgobj = sitk.ReadImage(input_path)

img_processed = preprocess('C:/mri/X.nii')
img_processed = preprocess(img)

img_array = sitk.GetArrayFromImage(imgobj)
img_array_rescaled = sitk.GetArrayFromImage(imgobj_rescaled)

training_img = create_training_examples(img_processed)

%gui qt
viewer = napari.Viewer()
viewer.add_image(img_array)
viewer.add_image(img_array_rescaled)
imgobj = sitk.ReadImage(input_rescaled)
for k in img.GetMetaDataKeys():
    v = img.GetMetaData(k)
    l = imgobj.GetMetaData(k)
    print("({0}) = {1} || {2}".format(k, v, l))

img.SetMetaData('pixdim[3]', '1')
img.SetMetaData('pixdim[4]', '1')
img.SetMetaData('cal_max', '14.3982')
img.SetMetaData('cal_min', '0.00174785')
img.SetMetaData('dim[0]', '4')
img.SetMetaData('dim[6]', '0')
img.SetMetaData('dim[7]', '0')
img.SetMetaData('slice_end', '29')