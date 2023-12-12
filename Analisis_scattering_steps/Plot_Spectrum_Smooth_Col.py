# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:16:39 2022

@author: lupau
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

base_folder = r'C:\Users\lupau\OneDrive\Documentos'
daily_folder = r'2022-06-29 Nanostars R20\20220629-143210_Scattering_Steps_Load_grid_INICIAL_4seg'

folder_NPs = r'more_NPs_per_photo'

parent_folder = os.path.join(base_folder, daily_folder, folder_NPs)

savefolder = os.path.join(parent_folder, 'smooth_spectrums')
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

cols = np.array([0,1,2,3,4])

n = 0
window_smooth = 31 #51

large_data = 723
totalNP = 10

plt.figure()

for col in cols:
    
    name_col = 'Col_%03d'%col
    matrix = np.zeros((large_data, totalNP +1))  
   # print(name_col)
    
    folder_col = os.path.join(parent_folder, name_col, 'fig_spectrums')
    
    list_files = os.listdir(folder_col)
    list_files =  [f for f in list_files if re.search('NP',f)]
    list_files.sort()
    
    for NP_file in list_files:
        
        NP = NP_file.split('NP_')[1]
        NP = int(NP.split('.txt')[0])
        
       # print(NP_file)
        
        file = os.path.join(folder_col, NP_file)
        
        data = np.loadtxt(file)
        
        londa = data[:, 0]
        spec = data[:, 1]
        #plt.plot(londa, spec)
        
        spec =  signal.savgol_filter(spec, window_smooth, 1, mode = 'mirror')
        #plt.plot(londa, spec)
        
        matrix[:, 0] = londa
        matrix[:, NP+1] = spec
        
        s = spec - min(spec)
        s = s/max(s)
        plt.plot(londa, s)
        
        n = n + 1
   
    name = os.path.join(savefolder, 'matrix_%s.txt'%name_col)
    np.savetxt(name, matrix)
    
plt.show()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Scattering')
name = os.path.join(parent_folder, 'fig_all_spectrums_smooth.png')
plt.savefig(name)
plt.close()
print(n)
