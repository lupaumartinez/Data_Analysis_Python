# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 23:19:18 2022

@author: lupau
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def closer(x,value):
    # returns the index of the closest element to value of the x array
    out = np.argmin(np.abs(x-value))
    return out

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

#%%

base_folder =  r'C:\Users\lupau\OneDrive\Documentos'
daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-174344_Luminescence_Load_grid_in_best_center_size_13'
exceptions_NP = []
parent_folder = os.path.join(base_folder, daily_folder)

save_folder = manage_save_directory(parent_folder, 'processed_sum_all_spectrums')

image_size_px = 10
camera_px_length = 1002

size_ROI = 13

list_of_folders = os.listdir(parent_folder)
list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
list_of_folders_slow = [f for f in list_of_folders if re.search('Slow_Confocal_Spectrum',f)]
list_of_folders_slow.sort()
list_of_folders_slow = list_of_folders_slow[:10]

lenNP = len(list_of_folders_slow) - len(exceptions_NP)
print(' cantidad NP', lenNP )

sum_total_NP = np.zeros((camera_px_length))

plt.figure()

for f in list_of_folders_slow:
    
    NP_folder = f.split('Slow_Confocal_Spectrum_')[1]
    
    if NP_folder in exceptions_NP:
        print('salteo', NP_folder)
        continue
    
    print(f)
    
    folder = os.path.join(parent_folder,f)
    
    list_of_files = os.listdir(folder)
    wavelength_filename = [f for f in list_of_files if re.search('wavelength',f)]
    list_of_files.sort()
    list_of_files = [f for f in list_of_files if not os.path.isdir(folder+f)]
    list_of_files = [f for f in list_of_files if ((not os.path.isdir(folder+f)) \
                                                  and (re.search('_i\d\d\d\d_j\d\d\d\d.txt',f)))]
    L = len(list_of_files)            
    
    data_spectrum = []
    name_spectrum = []
    specs = []
        
    print(L, 'spectra were acquired.')
    
    for k in range(L):
        name = os.path.join(folder,list_of_files[k])
        data_spectrum = np.loadtxt(name)
        name_spectrum.append(list_of_files[k])
        specs.append(data_spectrum)
    
    wavelength_filepath = os.path.join(folder,wavelength_filename[0])
    londa = np.loadtxt(wavelength_filepath)
    
    # ALLOCATING
    matrix_spec_raw = np.zeros((image_size_px,image_size_px,camera_px_length))
    
    sum_spec = np.zeros((camera_px_length))
       
   
    for i in range(image_size_px):
        for j in range(image_size_px):
            matrix_spec_raw[i,j,:] = np.array(specs[i*image_size_px+j])/size_ROI
            
            sum_spec[:] = sum_spec[:] + matrix_spec_raw[i,j,:]
            
    del specs
    
    sum_spec = sum_spec/image_size_px**2
    
    plt.plot(londa, sum_spec)
    
    sum_total_NP = sum_total_NP + sum_spec
    
sum_total_NP = sum_total_NP/lenNP
plt.plot(londa, sum_total_NP, 'k--')

data = np.array([londa, sum_total_NP]).T
np.savetxt(os.path.join(save_folder, 'sum_all_spectrums.txt'), data, header = 'wavelegnth (nm), mean all spectrums')
  
plt.savefig(os.path.join(save_folder, 'sum_all_spectrums..png'))
plt.show()
   
    