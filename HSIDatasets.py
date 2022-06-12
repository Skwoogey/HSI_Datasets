
import os
import json

from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
import scipy.io as scp
import numpy as np

from Scripts.createFile import createFile

#print(os.path.dirname(os.path.abspath(__file__)))

datasets_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + '/Datasets/'
available_datasets = os.listdir(datasets_path)
    

def getDataset(dsCode, bands=-1, iPCA_method=True):
    if dsCode not in available_datasets:
        raise ValueError('No such available dataset, manually add it first')
        
    info = getDatasetInfo(dsCode)
    
    dsCode = datasets_path + dsCode
    with open(dsCode + '/info.json', 'r') as file: 
        info = json.load(file)
        
    if bands == -1 or bands == info['shape'][2]:
        image_name = dsCode + '/image.mat'
    else:
        PCA_name = 'iPCA' if iPCA_method else 'PCA'
        image_name = dsCode + '/' + PCA_name + '/' + str(bands) + '.mat'
        if not os.path.exists(image_name):
            createFile(dsCode, info, bands, iPCA_method)
    
    gt_name = dsCode + '/gt.mat'
    
    return image_name, gt_name
    
def getDatasetInfo(dsCode):
    if dsCode not in available_datasets:
        raise ValueError('No such available dataset, manually add it first')
        
    dsCode = datasets_path + dsCode
    with open(dsCode + '/info.json', 'r') as file: 
        info = json.load(file)
        
    return info
    
if __name__ == "__main__":
    getDataset('SL', 20)