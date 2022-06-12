
import os
import json

from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import scipy.io as scp
import numpy as np

def createFile(input_folder, info, bands=20, iPCA_method=True):
    #PCA methods can only decrease size
    assert(bands > 0 and bands < info['shape'][2])
    
    PCA_method = IncrementalPCA if iPCA_method else PCA
    PCA_name = 'iPCA' if iPCA_method else 'PCA'
    
    image_file = input_folder + '/image.mat'
    
    Image_data_mat = scp.loadmat(image_file)
    Image_data = Image_data_mat['data']
    print("HS Image shape: ", Image_data.shape)

    
    num_of_pixels = Image_data.shape[0] * Image_data.shape[1]
    pixel_bandwidth = Image_data.shape[2]
    print('Number of pixels: ', num_of_pixels)
    
   
    all_pixels = np.reshape(Image_data, (num_of_pixels, pixel_bandwidth))
    
    
    ## pre-PCA normalization
    band_max = all_pixels.max(axis = 0)
    print(band_max.shape)
    all_pixels = all_pixels / band_max
    
    
    print(all_pixels)

    print('All pixels shape: ', all_pixels.shape)

    solver = PCA_method(n_components = bands)
    Image_reduced_data = solver.fit_transform(all_pixels)
    Image_reduced_data = np.ascontiguousarray(Image_reduced_data, dtype=np.float32)
    print("Pixels with reduced dimensionalty: ", Image_reduced_data.shape)


    new_image_shape = (Image_data.shape[0], Image_data.shape[1], bands)
    Image_reduced_data = np.reshape(Image_reduced_data, new_image_shape)
    
    ## post-PCA Normalization
    #max = np.max(Image_reduced_data, axis = (0,1))
    #Image_reduced_data = Image_reduced_data / max
    
    print("Image with reduced dimensionalty: ", Image_reduced_data.shape)

    new_file_name = input_folder + '/' + PCA_name + '/' + str(bands) + '.mat'
    Image_data_mat['data'] = Image_reduced_data
    scp.savemat(new_file_name, Image_data_mat)