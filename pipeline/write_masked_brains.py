"""Write masked brain volumes

This module will take the likelihood maps and the raw T2 volumes to output the masked brain based on the likelihood
threshold value defined as a variable. 
"""

import glob
import SimpleITK as sitk
from utils import sort_array


files = glob.glob('G:/likelihood-maps-and-brain-masks/day00/*.nii')
labels = [label for label in files if '_label.nii' in label]
likelihoods = [label for label in files if '_likelihood.nii' in label]

images = glob.glob('G:/mri-dataset/diffusion_0h/*/immediately_B0.nii.gz')
images = [path for path in files if 'scan' in path]

best_lh = 0.750

likelihoods.sort()
images.sort()

i = 1
list_len = len(likelihoods)
for likelihood_path, image in zip(likelihoods, images):
    print(likelihood_path)
    print(image)
    print(f'File {i}/{list_len}')
    t2wi = sitk.ReadImage(image)
    t2wi_array = sitk.GetArrayFromImage(t2wi)
    likelihood = sitk.ReadImage(likelihood_path)
    likelihood_array = sitk.GetArrayFromImage(likelihood)
    likelihood_array[best_lh > likelihood_array] = 0
    likelihood_array[likelihood_array > 0] = 1

    mask = sort_array(likelihood_array)
    mask = mask == 1

    t2wi_array[~mask] = 0
    out_t2wi_img = sitk.GetImageFromArray(t2wi_array)
    out_t2wi_img.SetSpacing(t2wi.GetSpacing())

    output_string = 'G:/masked-brains/day00/' + image.split('\\')[1] + '_masked-img.nii'
    sitk.WriteImage(out_t2wi_img, output_string)
    print(image.split('\\')[1] + ' is completed!')
    i += 1
