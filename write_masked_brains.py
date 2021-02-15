import glob
import SimpleITK as sitk
from input_functions import sort_array

"""
This module will take the likelihood maps and the raw T2 volumes and output the masked brain based on the likelihood
threshold value defined as a variable. 
"""

files = glob.glob('G:/likelihood-maps-and-brain-masks/day07/*.nii')
labels = [label for label in files if '_label.nii' in label]
likelihoods = [label for label in files if '_likelihood.nii' in label]

files = glob.glob('G:/mri-dataset/T2_7days/*/*.nii.gz')
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

    output_string = 'G:/masked-brains/day07/' + image.split('\\')[1] + '_masked-img.nii'
    sitk.WriteImage(out_t2wi_img, output_string)
    print(image.split('\\')[1] + ' is completed!')
    i+=1


