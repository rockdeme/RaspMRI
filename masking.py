import glob
import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt


# Evaluate the network's performance on the manual labels

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


files = glob.glob('E:/nii-files/*.nii')
labels = [label for label in files if '_label.nii' in label]
likelihoods = [label for label in files if '_likelihood.nii' in label]

files = glob.glob('F:/Downloads/T2_21days_all/*/*.nii.gz')
masks = [path for path in files if 'whole_brain' in path]
images = [item for item in files if 'whole_brain' not in item]
images = [item for item in images if 'bias2' not in item]

best_lh = 0.9438

for likelihood_path, image in zip(likelihoods, images):
    t2wi = sitk.ReadImage(image)
    t2wi_array = sitk.GetArrayFromImage(t2wi)
    likelihood = sitk.ReadImage(likelihood_path)
    likelihood_array = sitk.GetArrayFromImage(likelihood)
    likelihood_array[best_lh > likelihood_array] = 0
    likelihood_array[likelihood_array > 0] = 1
    mask = sort_array(likelihood_array)

