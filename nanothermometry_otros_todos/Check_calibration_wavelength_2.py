# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:45:30 2022

@author: lupau
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')

list_p = []

base_folder = r'C:\Users\lupau\OneDrive\Documentos'

daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-161433_Luminescence_Load_grid_in_best_center_size_13'
parent_folder = os.path.join(base_folder, daily_folder)
list_p.append(parent_folder)
daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-174344_Luminescence_Load_grid_in_best_center_size_13'
parent_folder = os.path.join(base_folder, daily_folder)
list_p.append(parent_folder)

colors = ['g', 'orange']

plt.figure()

for i in range(len(colors)):
    
    folder = os.path.join(list_p[i], 'processed_sum_all_spectrums')
    
    list_of_files = os.listdir(folder)
    list_of_files = [f for f in list_of_files if re.search('txt',f)]
    
    color = colors[i]
    
    for f in list_of_files:
        
        file = os.path.join(folder, f)
        data = np.loadtxt(file, skiprows = 1)
        londa = data[:, 0]
        spec = data[:, 1]
        
        spec_norm = spec #(spec - min(spec))/(max(spec) - min(spec))
        
        if color == 'orange':
            londa = londa - (610-550)
        
        plt.plot(londa, spec_norm, color = color)
      #  plt.xlim(523, 544)
      #  plt.ylim(0, 0.2)
        plt.show()