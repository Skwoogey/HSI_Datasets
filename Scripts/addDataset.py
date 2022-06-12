

import sys
import json
import os
from copy import copy

import scipy.io as scp
import numpy as np
from sklearn.preprocessing import scale

def unify_mat(data_dict, output_file, new_field_name='data', dtype=np.float32):
    new_data_dict = copy(data_dict)
    
    field_name = list(new_data_dict.keys())[-1]

    data_array = np.ascontiguousarray(new_data_dict[field_name], dtype=dtype)
    
    if dtype == np.float32:
        max = np.max(data_array, axis=(0, 1))
        data_array = data_array / max
        
        '''
        og_shape = data_array.shape
        data_array = data_array.reshape((-1, data_array.shape[-1]))
        print(data_array.shape, np.mean(data_array, axis=0), np.std(data_array, axis=0))
        data_array = scale(data_array)
        print(data_array.shape, np.mean(data_array, axis=0), np.std(data_array, axis=0))
        data_array = data_array.reshape(og_shape)
        '''
        

    del new_data_dict[field_name]
    new_data_dict[new_field_name] = data_array
    scp.savemat(output_file, new_data_dict)


info = {
    'shape': (-1,-1,-1),
    'labels': []
}

def addImageInfo(image):
    global info
    info["shape"] = list(list(image.values())[-1].shape)
    
def addGTInfo(gt):
    unique, counts = np.unique(list(gt.values())[-1], return_counts=True)
    global info
    for label in zip(unique[1:],counts[1:]):
        info["labels"].append({
            'label': -1,
            'count': -1
        })
        info["labels"][-1]["label"] = int(label[0])
        info["labels"][-1]["count"] = int(label[1])


def main():
    base_folder = sys.argv[1].replace('\\', '/')
    og_files_folder = base_folder + '/original_dataset/'
    og_files = os.listdir(og_files_folder)
    if len(og_files) != 2:
        raise ValueError('There should be only 2 files in "original_dataset" folder. Image and ground truth')
        
    og_files = [scp.loadmat(og_files_folder + file) for file in og_files]
    
    for file in og_files:
        shape = list(file.values())[-1].shape
        if len(shape) == 3:
            name = '/image.mat'
            dtype = np.float32
            addImageInfo(file)
        elif len(shape) == 2:
            name = '/gt.mat'
            dtype = int
            addGTInfo(file)
            
        unify_mat(file, base_folder + name, dtype=dtype)
    
    print(info)
    with open(base_folder + '/info.json', 'w') as info_file:
        json.dump(info, info_file)

if __name__ == '__main__':
    main()