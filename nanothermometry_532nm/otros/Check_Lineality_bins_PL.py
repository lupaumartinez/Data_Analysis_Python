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

base_folder =  r'C:\Ubuntu_archivos\Printing'
daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-161433_Luminescence_Load_grid_in_best_center_size_13'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data')
exceptions_NP = []
np_col = 5

save_folder = os.path.join(parent_folder, 'PL stokes normalized from bins')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

list_of_folders = os.listdir(parent_folder)
list_of_folders_NP = [f for f in list_of_folders if re.search('Col',f)]

print(len(list_of_folders_NP)-len(exceptions_NP))

size_roi = 13

file = os.path.join(parent_folder, list_of_folders_NP[0], 'pl_in_bins', 'londa_%s.dat'%list_of_folders_NP[0])
londa = np.loadtxt(file)

desired_stokes = np.where((londa >= 543) & (londa < londa[-1]))

londa_stokes = londa[desired_stokes]
N = len(londa_stokes)

matrix_sum1 = np.zeros((np_col, N))
matrix_sum2 = np.zeros((np_col, N))
count = np.zeros(np_col)


NP = []
Col = []

color = []
for e in sns.color_palette():
    color.append(e)

plt.figure()

for NP_folder in list_of_folders_NP:

    if NP_folder in exceptions_NP:
        print('salteo', NP_folder)
        continue
    
    name_col = NP_folder.split('_NP')[0]
    name_col = int(name_col.split('Col_')[1])
     
    name_NP = int(NP_folder.split('NP_')[1])
    
    if name_NP >= np_col:
        name_col = int(name_NP/np_col) + 1
        print(name_NP, name_col)
    
    NP_unique = int(name_NP-np_col*(name_col-1))
    
    Col.append(name_col)
    NP.append(name_NP)
   
    folder = os.path.join(parent_folder, NP_folder)
    
    file = os.path.join(folder, 'pl_in_bins', 'all_bins_%s.dat'%NP_folder)
    matrix_specs_bins = np.loadtxt(file)
    totalbins = matrix_specs_bins.shape[1]
    
   # spec_max = matrix_specs_bins[:, totalbins-1]/size_roi
   # stokes_max = spec_max[desired_stokes]
    
    
    #plt.ylim(0, 1)
   # plt.title(NP_folder)
    
    sum_stokes_norm_1 = np.zeros(len(londa_stokes))
    sum_stokes_norm_2 = np.zeros(len(londa_stokes))
    
    for i in range(1,totalbins):
        
        spec = matrix_specs_bins[:, i]/size_roi
        stokes_i = spec[desired_stokes]
        
        stokes_i =  stokes_i
        
        if np.mean(stokes_i) < 0:
            print('salteo', NP_folder, 'bin reference ', i)
            continue
        
        for j in range(1,totalbins):
            
            spec = matrix_specs_bins[:, j]/size_roi
            stokes_j = spec[desired_stokes]
            stokes_j =  stokes_j
            
            if np.mean(stokes_j) < 0:
                print('salteo', NP_folder, 'bin', j)
                continue
            
            stokes_norm = stokes_j/stokes_i
            
              # stokes_j =  stokes_j/max( stokes_j)
            
            if j < i: #bin menor a bin referencia
                #plt.plot(londa_stokes, stokes_norm)# color = color[NP_unique])
                sum_stokes_norm_1 = sum_stokes_norm_1 + stokes_norm
                
            if j > i:#bin mayor a bin referencia
                #plt.plot(londa_stokes, stokes_norm)# color = color[NP_unique])
                sum_stokes_norm_2 = sum_stokes_norm_2 + stokes_norm
   
    sum1 = (sum_stokes_norm_1-min(sum_stokes_norm_1))/(max(sum_stokes_norm_1)-min(sum_stokes_norm_1))
    sum2 = (sum_stokes_norm_2-min(sum_stokes_norm_2))/(max(sum_stokes_norm_2)-min(sum_stokes_norm_2))
    
    if np.where(NP_unique == range(np_col)):
        count[NP_unique] = count[NP_unique] + 1
        matrix_sum2[NP_unique, :] = matrix_sum2[NP_unique, :] + sum2
        matrix_sum1[NP_unique, :] = matrix_sum1[NP_unique, :] + sum1
        
    plt.plot(londa_stokes, sum1, '--',  color = 'k')
    plt.plot(londa_stokes, sum2, '-',  color = color[NP_unique])
        
   # plt.savefig(os.path.join(save_folder, 'bins_pl_stokes_normalized_%s.png'%NP_folder))
   # plt.close()
   
plt.savefig(os.path.join(save_folder, 'bin_pl_stokes_normalized_per_NP.png'))
   
plt.figure()

for i in range(np_col):
    
    spec = matrix_sum1[i, :]/count[i]
    spec2 = matrix_sum2[i, :]/count[i]
    data = np.array([londa_stokes, spec, spec2]).T
    np.savetxt(os.path.join(save_folder, 'bin_pl_stokes_normalized_mean_%d.txt'%i), data, header = 'wavelegnth (nm), normalized minor bin ref, normalized bigger bin ref')
    plt.plot(londa_stokes, spec, 'k--')
    plt.plot(londa_stokes, spec2, '-')
    
    plt.show()
    
plt.savefig(os.path.join(save_folder, 'bin_pl_stokes_normalized_mean_per_NP.png'))