import scipy.io as scp
from sklearn.decomposition import PCA
import numpy as np
import sys
import os

# arg1 - image file
# arg2 - ground truth file
# arg3 - number of output bands
# arg4 - output folder

Image_file = sys.argv[1]
Image_gt_file = sys.argv[2]
num_of_bands = int(sys.argv[3])
Output_folder = sys.argv[4]

Image_data_loaded_mat = scp.loadmat(Image_file)
print(Image_data_loaded_mat)
Image_data = Image_data_loaded_mat['data']
Image_gt = scp.loadmat(Image_gt_file)['data']
print("HS Image shape: ", Image_data.shape)
print("HS ground truth shape: ", Image_gt.shape)

num_of_pixels = Image_data.shape[0] * Image_data.shape[1]
pixel_bandwidth = Image_data.shape[2]
print('Number of pixels: ', num_of_pixels)

classified_pixels = np.reshape(Image_data, (num_of_pixels, pixel_bandwidth))
band_max = classified_pixels.max(axis = 0)
print(band_max.shape)
classified_pixels = classified_pixels / band_max
print(classified_pixels)

'''
cp_ptr = 0
for x in range(Image_data.shape[0]):
	for y in range(Image_data.shape[1]):
		if Image_gt[x, y] != 0:
			classified_pixels[cp_ptr] = Image_data[x, y]
			cp_ptr += 1
'''

print('Classified pixels shape: ', classified_pixels.shape)

PCA_solver = PCA(n_components = num_of_bands)
Image_reduced_data = PCA_solver.fit_transform(classified_pixels)
Image_reduced_data = np.ascontiguousarray(Image_reduced_data)
print("Pixels with reduced dimensionalty: ", Image_reduced_data.shape)


new_image_shape = (Image_data.shape[0], Image_data.shape[1], num_of_bands)
Image_reduced_data = np.reshape(Image_reduced_data, new_image_shape)
print("Image with reduced dimensionalty: ", Image_reduced_data.shape)

new_file_name = Output_folder + os.path.sep + Image_file.split('/')[-1].split('\\')[-1].replace('.mat', '_' + str(num_of_bands) + 'bands.mat')
Image_data_loaded_mat['data'] = Image_reduced_data
scp.savemat(new_file_name, Image_data_loaded_mat)