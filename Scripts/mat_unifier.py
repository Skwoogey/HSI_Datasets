import scipy.io as scp
from sklearn.decomposition import PCA
import numpy as np
import sys
import os

# arg1 - file
# arg2 - output folder
# arg3 - field name
# arg4 - new parameter name. default "data"

if len(sys.argv) < 3:
	print('too little arguments')
	exit()

Image_file = sys.argv[1]
Output_folder = sys.argv[2]
field_name = sys.argv[3] if len(sys.argv) >= 4 else None
new_field_name = sys.argv[4] if len(sys.argv) >= 5 else 'data'

Image_data_loaded_mat = scp.loadmat(Image_file)
print(list(Image_data_loaded_mat.keys()))
print(list(Image_data_loaded_mat.keys())[-1])

if field_name == None:
	field_name = list(Image_data_loaded_mat.keys())[-1]

Image_data_loaded_mat[new_field_name] = Image_data_loaded_mat[field_name]

new_file_name = Output_folder + os.path.sep + Image_file.split('/')[-1].split('\\')[-1]
Image_data_loaded_mat[new_field_name] = Image_data_loaded_mat[field_name]
del Image_data_loaded_mat[field_name]
print(Image_data_loaded_mat)
scp.savemat(new_file_name, Image_data_loaded_mat)