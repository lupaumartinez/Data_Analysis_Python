# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:48:11 2022

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
NP_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-161433_Luminescence_Load_grid_in_best_center_size_13'
NP_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-174344_Luminescence_Load_grid_in_best_center_size_13'

parent_folder = os.path.join(base_folder, NP_folder)

save_folder = manage_save_directory(parent_folder,'pl_maximum')

image_size_px = 10
camera_px_length = 1002
window, deg, repetitions = 51, 1, 2
mode = 'interp'

start_notch_0 = 583 #522 # in nm where notch starts ~525 nm (safe zone)
end_notch_0  = 606 #543 # in nmwhere notch starts ~540 nm (safe zone)
start_power_0  = 606 #543 # in nm from here we are going to calculate irradiance
end_power_0  = 660 #560#600#560 # in nm from start_power up to this value we calculate irradiance
lower_londa_0  =   515 #nm 510
upper_londa_0  = start_notch_0  - 2

list_of_folders = os.listdir(parent_folder)
list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
list_of_folders_slow = [f for f in list_of_folders if re.search('Slow_Confocal_Spectrum',f)]
list_of_folders_slow.sort()
list_of_folders_slow = list_of_folders_slow[:10]

print('cantidad NP', len(list_of_folders_slow))


fig=plt.figure(num=1,clear=True)
plt.style.use('default')
ax = fig.add_subplot()
ax = plt.gca()
ax.set_xlabel(r'Wavelength (nm)')
ax.set_ylabel('Intensity (a.u.)')

for f in list_of_folders_slow:
    
    print(f)
    
    NP = f.split('Slow_Confocal_Spectrum_')[1]
    
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
    
    start_notch = closer(londa, start_notch_0 )
    end_notch = closer(londa, end_notch_0 )
    start_power = closer(londa, start_power_0 )
    end_power = closer(londa, end_power_0 )
    lower_londa_index = closer(londa, lower_londa_0 )
    upper_londa_index = closer(londa, upper_londa_0 )
    
    # ALLOCATING
    matrix_spec_raw = np.zeros((image_size_px,image_size_px,camera_px_length))
    matrix_spec_smooth = np.zeros((image_size_px,image_size_px,camera_px_length))
   # matrix_spec = np.zeros((image_size_px,image_size_px,camera_px_length))
   # matrix_spec_normed = np.zeros((image_size_px,image_size_px,camera_px_length))
       
    for i in range(image_size_px):
        for j in range(image_size_px):
            matrix_spec_raw[i,j,:] = np.array(specs[i*image_size_px+j])
    del specs
    
    ######################## SMOOTH ############################
    ######################## SMOOTH ############################
    ######################## SMOOTH ############################
    
    print(start_notch)
    
    # SPLIT SIGNALS INTO STOKES AND ANTI-STOKES
    matrix_stokes_raw = matrix_spec_raw[:,:,end_notch:]
    londa_stokes = londa[end_notch:]
    
    matrix_antistokes_raw = matrix_spec_raw[:,:,:start_notch]
    londa_antistokes = londa[:start_notch]
    
    # SMOOTHING
    print('Smoothing signals...')
    aux_matrix_stokes_smooth = sig.savgol_filter(matrix_stokes_raw, 
                                               window, deg, axis = 2, 
                                               mode=mode)
    aux_matrix_antistokes_smooth = sig.savgol_filter(matrix_antistokes_raw, 
                                               window, deg, axis = 2, 
                                               mode=mode)
    
    for i in range(repetitions - 1):
        aux_matrix_stokes_smooth = sig.savgol_filter(aux_matrix_stokes_smooth,
                                                   window, deg, axis = 2, 
                                                   mode=mode)
        aux_matrix_antistokes_smooth = sig.savgol_filter(aux_matrix_antistokes_smooth,
                                                   window, deg, axis = 2, 
                                                   mode=mode)
    # Merge
    matrix_stokes_smooth = aux_matrix_stokes_smooth
    matrix_antistokes_smooth = aux_matrix_antistokes_smooth
    matrix_spec_smooth[:,:,end_notch:] = matrix_stokes_smooth
    matrix_spec_smooth[:,:,:start_notch] = matrix_antistokes_smooth
    
    matrix_spec_smooth[:,:,start_notch:end_notch] = np.nan
    
    matrix_stokes_smooth = matrix_spec_smooth[:,:,start_power:end_power]
    aux_sum = np.sum(matrix_stokes_smooth, axis=2)
    
    print('Finding max and bkg (min) spectra...')            
    
    imin, jmin = np.unravel_index(np.argmin(aux_sum, axis=None), aux_sum.shape)
    bkg_smooth = matrix_spec_smooth[imin, jmin, :]
    
    imax, jmax = np.unravel_index(np.argmax(aux_sum, axis=None), aux_sum.shape)
    max_smooth = matrix_spec_smooth[imax, jmax, :]
    
    max_signal = max_smooth - bkg_smooth
    
    ax.plot(londa, max_signal, label = '%s'%NP)
    
  #  matrix_spec = matrix_spec_smooth - bkg_smooth
  #  matrix_spec_normed = (matrix_spec_smooth - bkg_smooth) / (max_smooth - bkg_smooth)
    
  #  matrix_spec = matrix_spec[:, :, end_notch:]
  #  matrix_spec_normed = matrix_spec_normed[:, :, end_notch:]
    
    #for i in range(image_size_px):
     #   for j in range(image_size_px):
      #      pixel_name = 'i%02d_j%02d' % (i,j)
       #     ax.plot(londa_stokes, matrix_spec)
       
    data = np.array([londa, max_signal]).T
    name = os.path.join(save_folder, 'max_signal_PL_%s.txt'%NP)
    header = 'wavelength (nm), Counts'
    np.savetxt(name, data, header = header)
    
plt.legend()  
plt.savefig(os.path.join(save_folder, 'max_signal_PL.png'))
plt.show()