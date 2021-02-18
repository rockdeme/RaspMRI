import SimpleITK as sitk
import napari
from rbm.core.dice import dice_coef, dice_coef_loss
from rbm.core.paras import KerasParas
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from functools import partial
from input_functions import preprocess, rescale_voxel_size, create_training_examples


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a dictionary with features that may be relevant.
def image_example(image, label):
    image.tobytes()
    feature = {
      'height': _int64_feature(image.shape[0]),
      'width': _int64_feature(image.shape[1]),
      'image': _bytes_feature(image.tobytes()),
      'label': _bytes_feature(label.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# WRITE TFRECORDS DATA

files = tf.io.gfile.glob('F:/Downloads/T2_21days_all/*/*.nii.gz')

masks = [path for path in files if 'whole_brain' in path]
images = [item for item in files if 'whole_brain' not in item]
images = [item for item in images if 'bias2' not in item]

for input_path, mask_input_path in zip(images,masks):
    split = input_path.split('\\')
    imgobj = sitk.ReadImage(input_path)
    img_rescaled = rescale_voxel_size(input_path)
    img_preprocessed = preprocess(img_rescaled)
    training_img = create_training_examples(img_preprocessed)

    mask_img_rescaled = rescale_voxel_size(mask_input_path)
    mask_img_preprocessed = preprocess(mask_img_rescaled)
    mask_training_img = create_training_examples(mask_img_preprocessed)

    img_slice = training_img[100,:,:]
    label_img_slice = mask_training_img[100,:,:]

    record_file = 'F:/mri-dataset/' + split[3] + '.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
      for i in range(training_img.shape[0]):
        tf_example = image_example(training_img[i,:,:], mask_training_img[i,:,:])
        writer.write(tf_example.SerializeToString())


# DATASET FROM TFRECORDS DATA

tfrecord_files = tf.io.gfile.glob("F:/*.tfrecords")

def read_tfrecord(example):
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, image_feature_description)
    image_raw = example['image']
    label_raw = example['label']
    width = example['width']
    height = example['height']
    image = tf.io.decode_raw(image_raw, tf.float64)
    image = tf.reshape(image, [width, height])
    label = tf.io.decode_raw(label_raw, tf.float64)
    label = tf.reshape(label, [width, height])
    return image, label


def load_dataset(filenames):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def get_dataset(filenames, BATCH_SIZE, labeled=True):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

train_dataset = get_dataset(tfrecord_files[0:20], 16)
validation_dataset = get_dataset(tfrecord_files[21:25], 16)

# Parameters for Keras model
keras_paras = KerasParas()
keras_paras.outID = 0
keras_paras.thd = 0.5
keras_paras.loss = 'dice_coef_loss'
keras_paras.img_format = 'channels_last'
keras_paras.model_path = '/content/drive/MyDrive/mri-dataset/rat_brain-2d_unet.hdf5'

seg_net = load_model(keras_paras.model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
seg_net.summary()

history = seg_net.fit(train_dataset, epochs = 5, validation_data = validation_dataset)

def loss_plot(history):
    plt.clf()
    plt.plot(history.history['loss'], color = 'grey')
    plt.plot(history.history['val_loss'], color = 'indigo')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


loss_plot(history)

image_batch, label_batch = next(iter(train_dataset))
pred = seg_net.predict(image_batch)
fig = plt.figure(figsize=(9, 18))
for i in range(1, 21, 3):
    fig.add_subplot(7, 3, i)
    plt.imshow(image_batch[i].numpy())
    fig.add_subplot(7, 3, i+1)
    plt.imshow(label_batch[i].numpy())
    fig.add_subplot(7, 3, i+2)
    plt.imshow(pred[i][:, :, 0])
plt.show()

# %gui qt magic command
viewer = napari.Viewer()
viewer.add_image(img_preprocessed)
viewer.add_image(training_img)
viewer.add_image(mask_training_img)

