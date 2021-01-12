import os
import numpy as np
from keras.models import load_model
from rbm.core.paras import KerasParas
from rbm.core.dice import dice_coef, dice_coef_loss
from input_functions import preprocess, create_training_examples
from sklearn.model_selection import train_test_split

# Parameters for Keras model
keras_paras = KerasParas()
keras_paras.outID = 0
keras_paras.thd = 0.5
keras_paras.loss = 'dice_coef_loss'
keras_paras.img_format = 'channels_last'
keras_paras.model_path = os.path.join(os.getcwd(), 'rbm', 'scripts', 'rat_brain-2d_unet.hdf5')

X_input_path = 'C:/mri/X.nii'
y_input_path = 'C:/mri/y.nii'

X_img = preprocess(X_input_path)
y_img = preprocess(y_input_path)

X = create_training_examples(X_img)
y = create_training_examples(y_img)

X = X[0:round(X.shape[0]/4), :, :]
y = y[0:round(y.shape[0]/4), :, :]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

seg_net = load_model(keras_paras.model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
seg_net.summary()
history = seg_net.fit(X_train, y_train, batch_size=8, epochs = 2, validation_data = (X_test, y_test))

X_input_path = 'C:/mri/X.nii'
y_input_path = 'C:/mri/y.nii'

X_img = preprocess(X_input_path)
y_img = preprocess(y_input_path)

X = create_training_examples(X_img)

#%gui qt
viewer = napari.Viewer()
viewer.add_image(counter_map)
viewer.add_image(categorical_map)
viewer.add_image(z_stack)
viewer.add_image(X_img)
viewer.add_image(y_img)