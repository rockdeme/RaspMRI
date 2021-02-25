import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.measure import regionprops


atlas_path = 'G:/SIGMA/cranial_atlas.nii'

atlas = sitk.ReadImage(atlas_path)
atlas_array = sitk.GetArrayFromImage(atlas)
atlas_array = atlas_array.astype(int)

regions = regionprops(atlas_array)

size = []
for region in regions:
    size.append(region.area)

plt.hist(size, bins=30)
plt.yscale("log")
plt.show()

plt.boxplot(size)
plt.yscale("log")
plt.show()

day03 = pd.read_csv("G:/mri-results/t2_data_day03_normalized.csv", sep = ';')
areas = np.array(day03['VI_7_d_87_72h_masked-img.nii_Right Hemisphere Area'].tolist())
areas.sort()
areas = areas / sum(areas)
plt.plot(areas)
plt.show()
plt.hist(areas, bins=30)
plt.show()
