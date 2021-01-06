from keras.models import load_model

loaded_model = load_model("rbm/scripts/rat_brain-2d_unet.hdf5", compile=False)
loaded_model.summary()