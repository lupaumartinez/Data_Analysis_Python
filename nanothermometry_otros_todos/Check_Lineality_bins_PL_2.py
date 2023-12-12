# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:45:30 2022

@author: lupau
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

#%%

plt.close('all')
list_p = []

base_folder = r'C:\Users\lupau\OneDrive\Documentos'

daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-161433_Luminescence_Load_grid_in_best_center_size_13'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data')
list_p.append(parent_folder)
daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-174344_Luminescence_Load_grid_in_best_center_size_13'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_1')
list_p.append(parent_folder)

colors = ['g', 'orange']

fig, ax = plt.subplots()

for i in range(len(colors)):
    
    parent_folder = list_p[i]
    
    color = colors[i]
        
    #folder = os.path.join(parent_folder, 'Mean Spectrums per NP')
    
    #list_of_files = os.listdir(folder)
    #list_of_files = [f for f in list_of_files if re.search('txt',f)]
    #list_of_files.sort()
    
    folder2 = os.path.join(parent_folder, 'PL stokes normalized from bins')
    
    list_of_files2 = os.listdir(folder2)
    list_of_files2 = [f for f in list_of_files2 if re.search('txt',f)]
    list_of_files2.sort()
    
   # print(len(list_of_files), len(list_of_files2))
    
    for i in range(len(list_of_files2)):
        
       # file = os.path.join(folder, list_of_files[i])
       # data = np.loadtxt(file, skiprows = 1)
       # londa = data[:, 0]
       # spec = data[:, 1]
        
        file = os.path.join(folder2, list_of_files2[i])
        data = np.loadtxt(file, skiprows = 1)
        londa_stokes = data[:, 0]
        norm_spec1 = data[:, 1]
        norm_spec2 = data[:, 2]
        
        cross = londa_stokes[np.argmin(np.abs(norm_spec2 - norm_spec1))]
        print(color, cross)
        
        
        b = norm_spec1/max(norm_spec1)
        c = norm_spec2/max(norm_spec2)
        
        if color == 'orange':
            londa_stokes = londa_stokes - (606-543)
            cross = londa_stokes[np.argmin(np.abs(norm_spec2 - norm_spec1))]
            print('corregido', color, cross)
        
            
        ax.plot(londa_stokes, c, '-', color = color)
        ax.plot(londa_stokes, b, '--', color = color)
        
    plt.show()