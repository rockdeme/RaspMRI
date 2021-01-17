import SimpleITK as sitk
import napari
import tensorflow as tf
from input_functions import preprocess, rescale_voxels, create_training_examples

input_path = 'C:/mri/VI_2_a_26_21days_after_stroke_t2_cor_30.4.1.nii.gz'
mask_input_path = 'C:/mri/VI_2_a_26_21days_after_stroke_t2_cor_30.4.1-whole_brain.nii.gz'

imgobj = sitk.ReadImage(input_path)
img_rescaled = rescale_voxels(input_path)
img_preprocessed = preprocess(img_rescaled)
training_img = create_training_examples(img_preprocessed)

mask_img_rescaled = rescale_voxels(mask_input_path)
mask_img_preprocessed = preprocess(mask_img_rescaled)
mask_training_img = create_training_examples(mask_img_preprocessed)

img_slice = training_img[100,:,:]
label_img_slice = mask_training_img[100,:,:]


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Create a dictionary with features that may be relevant.
def image_example(image, label):
    image.tobytes()
    feature = {
      'height': _float_feature(image.shape[0]),
      'width': _float_feature(image.shape[1]),
      'image': _bytes_feature(image.tobytes()),
      'label': _bytes_feature(label.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

record_file = 'F:/images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for img, label in zip(img_slice,label_img_slice):
    tf_example = image_example(img, label)
    writer.write(tf_example.SerializeToString())






%gui qt
viewer = napari.Viewer()
viewer.add_image(img_preprocessed)
viewer.add_image(training_img)
viewer.add_image(mask_training_img)

imgobj = sitk.ReadImage(img_preprocessed)
for k in imgobj.GetMetaDataKeys():
    v = imgobj.GetMetaData(k)
    print("({0}) = {1} || {2}".format(k, v))
