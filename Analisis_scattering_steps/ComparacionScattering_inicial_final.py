# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:35:56 2022

@author: Luciana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sig

base_folder = r'C:\Ubuntu_archivos\Printing'
daily_folder = r'2022-07-11 Nanostars P2R20 post impresion'

save_folder = os.path.join(base_folder, daily_folder, 'comparacion_scattering')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

folder_initial = r'Scattering_initial'
folder_final = r'Scattering_final'

list_f = [folder_initial, folder_final]

colors = ['b', 'm']

cols = np.array([1,2,3,5,6,7])

for col in cols:

    name_col = 'Col_%03d'%col
    
    plt.figure()
    
    for j in range(2):
        
        f = list_f[j]
        
        f2 = os.path.join(base_folder, daily_folder, f)
        
        f2 = os.path.join(f2, r'photos\smooth_spectrums')
        
       # list_files = os.listdir(f2)
       # list_files =  [f for f in list_files if re.search('txt',f)]
       # list_files.sort()
        
        c = colors[j]
        
        col_file = 'matrix_%s.txt'%name_col
        
        file = os.path.join(f2, col_file)
        
        data = np.loadtxt(file)
        
        londa = data[:, 0]
        matrix_spec = data[:, 1:]
            
        for k in range(matrix_spec.shape[1]):
            
            spec = matrix_spec[:, k]
            
            if max(spec)<0.1:
                print('no tiene espectro', col_file, k)
                continue
            
            spec = spec - min(spec)
            spec = spec/max(spec)
        
            plt.plot(londa, spec, color = c)
            
    plt.show()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Scattering')
    namefig = os.path.join(save_folder, 'Comparacion_Scattering_inicial_final_%s.png'%name_col)
    plt.savefig(namefig)

#%%

colors = ['C0', 'C0','C0', 'C1','C1', 'C2']
times = [30, 30, 30, 60, 60, 120]
cols = [5,6,7,1,2,3]

f2 = os.path.join(base_folder, daily_folder, r'Scattering_initial\photos\smooth_spectrums')

plt.figure()

for j in range(6):
    
    c = colors[j]
    
    col = cols[j]
    name_col = 'Col_%03d'%col
    col_file = 'matrix_%s.txt'%name_col
    
    file = os.path.join(f2, col_file)
    
    data = np.loadtxt(file)
    
    londa = data[:, 0]
    matrix_spec = data[:, 1:]
        
    for k in range(matrix_spec.shape[1]):
        
        spec = matrix_spec[:, k]
        
        if max(spec)<0.1:
            print('no tiene espectro', col_file, k)
            continue
        
        spec = spec - min(spec)
        spec = spec/max(spec)
    
        plt.plot(londa, spec, color = c)
        
plt.show()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Scattering')
namefig = os.path.join(save_folder, 'Scattering_initial.png')
plt.savefig(namefig)