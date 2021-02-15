import tensorflow as tf

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
def image_example(image, label = None):
    if label:
        feature = {
          'height': _int64_feature(image.shape[0]),
          'width': _int64_feature(image.shape[1]),
          'image': _bytes_feature(image.tobytes()),
          'label': _bytes_feature(label.tobytes()),
        }
    else:
        feature = {
          'height': _int64_feature(image.shape[0]),
          'width': _int64_feature(image.shape[1]),
          'image': _bytes_feature(image.tobytes())
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
    img_rescaled = rescale_voxels(input_path)
    img_preprocessed = preprocess(img_rescaled)
    training_img = create_training_examples(img_preprocessed)

    mask_img_rescaled = rescale_voxels(mask_input_path)
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





#todo method 1
initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                      moving_image,
                                                      sitk.AffineTransform(3),
                                                      sitk.CenteredTransformInitializerFilter.MOMENTS)

moving_initial = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())


registration_method = sitk.ImageRegistrationMethod()

registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
# Scale the step size differently for each parameter, this is critical!!!
registration_method.SetOptimizerScalesFromPhysicalShift()

registration_method.SetInitialTransform(initial_transform, inPlace=False)

registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

final_transform_v1 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                 sitk.Cast(moving_image, sitk.sitkFloat32))

moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform_v1, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

#todo method 2
registration_method = sitk.ImageRegistrationMethod()

registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                  numberOfIterations=100)
registration_method.SetOptimizerScalesFromPhysicalShift()

final_transform = sitk.AffineTransform(initial_transform)
registration_method.SetInitialTransform(final_transform)
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                               update_multires_iterations)
registration_method.AddCommand(sitk.sitkIterationEvent,
                               lambda: plot_values(registration_method))

final_transform_v2 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),sitk.Cast(moving_image, sitk.sitkFloat32))

moving_resampled_2 = sitk.Resample(moving_image, fixed_image, final_transform_v2, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

# %gui qt magic command
viewer = napari.Viewer()
viewer.add_labels(sitk.GetArrayFromImage(atlas))
viewer.add_labels(sitk.GetArrayFromImage(atlas_res))

viewer.add_image(sitk.GetArrayFromImage(fixed_image))
viewer.add_image(sitk.GetArrayFromImage(m_res))


viewer.add_image(sitk.GetArrayFromImage(moving_initial))
viewer.add_image(sitk.GetArrayFromImage(moving_resampled))
viewer.add_image(sitk.GetArrayFromImage(moving_image))
viewer.add_image(sitk.GetArrayFromImage(moving_resampled_2))


means_df = output_df.drop(columns = list(output_df.filter(regex='Area')))
means = means_df.iloc[:,7:].mean(axis=1)
stdevs = means_df.iloc[:,7:].std(axis=1)

means_df = normalized_df.drop(columns = list(normalized_df.filter(regex='Area')))
means = means_df.iloc[:,7:].mean(axis=1)
stdevs = means_df.iloc[:,7:].std(axis=1)
left_df = means_df.drop(columns = list(means_df.filter(regex='Right')))
l_means = left_df.iloc[:,7:].mean(axis=1)
l_stdevs = left_df.iloc[:,7:].std(axis=1)
right_df = means_df.drop(columns = list(means_df.filter(regex='Left')))
r_means = right_df.iloc[:,7:].mean(axis=1)
r_stdevs = right_df.iloc[:,7:].std(axis=1)

plt.errorbar(means_df.index, means, yerr=stdevs, fmt='.k', ecolor= 'indigo')
plt.show()

plt.errorbar(means_df.index, l_means, yerr=l_stdevs, fmt='.k', ecolor= 'indigo')
plt.ylim([0.25, 2.5])
plt.show()
plt.errorbar(means_df.index, r_means, yerr=r_stdevs, fmt='.k', ecolor= 'indigo')
plt.ylim([0.25, 2.5])
plt.show()


complete_df.to_csv("G:/data.csv")

plt.plot(complete_df['Left Hemisphere Intensity'])
plt.plot(complete_df['Right Hemisphere Intensity'])
plt.show()
