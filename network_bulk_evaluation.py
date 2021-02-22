import glob
import SimpleITK as sitk
import numpy as np
import pandas as pd
from utils import sort_array, dice_coef_np
import matplotlib.pyplot as plt

# Evaluate the network's performance on the manual labels

files = glob.glob('G:/nii-files/*.nii')
labels = [label for label in files if '_label.nii' in label]
likelihoods = [label for label in files if '_likelihood.nii' in label]

files = glob.glob('F:/Downloads/T2_21days_all/*/*.nii.gz')
masks = [path for path in files if 'whole_brain' in path]

dices = []

for label_path, likelihood_path, mask in zip(labels, likelihoods, masks):
    ground_truth = sitk.ReadImage(mask)
    likelihood = sitk.ReadImage(likelihood_path)
    label = sitk.ReadImage(label_path)
    ground_truth_array = sitk.GetArrayFromImage(ground_truth)
    label_array = sitk.GetArrayFromImage(label)
    likelihood_array = sitk.GetArrayFromImage(likelihood)

    dice_list = []
    print('Calculating dice scores!')
    for i in np.arange(0, 1, 0.01):
        likelihood_array_copy = np.copy(likelihood_array)
        likelihood_array_copy[i > likelihood_array_copy] = 0
        likelihood_array_copy[likelihood_array_copy > 0] = 1
        likelihood_array_copy = sort_array(likelihood_array_copy)
        dice = dice_coef_np(ground_truth_array, likelihood_array_copy)
        dice_list.append(dice)
    print('Dices scores calculated for ' + label_path)
    dices.append(dice_list)


df = pd.DataFrame.from_records(dices)
df = df.T
mean = df.mean(axis=1)
stdev = df.std(axis=1)
t = np.arange(0,1,0.01)
fig, ax = plt.subplots(1)
ax.plot(t,mean, lw=2, color='indigo')
plt.ylim([0.7, 1.05])
plt.xlim([-0.005, 1.006])
text_str = 'Max Dice Score: ' + str(np.round(mean.max(), 4)) + u"\u00B1" + str(np.round(stdev.iloc[mean.idxmax()], 3))
plt.text(0.3, mean.max()+0.02, text_str, fontsize = 12)
plt.hlines(mean.max(),-10,110, colors='red')
ax.fill_between(t,mean+stdev, mean-stdev, facecolor='indigo', alpha=0.5, zorder = 2)
plt.grid(zorder = 00)
plt.title('Day 21')
plt.ylabel('Dice Score')
plt.xlabel('Likelihood Threshold')
plt.show()

plt.hist(likelihood_array[likelihood_array > 0.001].flatten(), bins=20, color='indigo', zorder = 2)
plt.yscale('log')
plt.title('Likelihood Map Intensity Histogram')
plt.ylabel('Pixel count')
plt.xlabel('Likelihood')
plt.grid(zorder = 0)
plt.show()

