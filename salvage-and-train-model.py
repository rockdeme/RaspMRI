from keras.models import load_model
from rbm.core.dice import dice_coef
from rbm.core.dice import dice_coef_loss


loaded_model = load_model("rbm/scripts/rat_brain-2d_unet.hdf5",
                          custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
loaded_model.summary()