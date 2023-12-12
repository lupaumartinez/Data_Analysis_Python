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

base_folder = r'C:\Ubuntu_archivos\Printing'

#%%
daily_folder = r'2022-06-22 Au60 satelites Pd\20220622-115036_Luminescence_Load_grid_in_best_center'
exceptions_NP = []
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_298nm')
np_col = 10

daily_folder = r'2022-06-22 Au60 satelites Pd\20220621-185503_Luminescence_Load_grid_in_best_center'
exceptions_NP = []
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_315nm')
np_col = 10

#%%

#base_folder = r'C:\Users\lupau\OneDrive\Documentos'

#daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220615-124348_Luminescence_10x5_3.0umx0.0um_size_ROI_19_in_best_center'
#exceptions_NP = []
#parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_319nm')
#np_col = 10

#daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220614-165226_Luminescence_10x10_3.0umx0.0um'
#exceptions_NP = []
#parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_326nm')
#np_col = 10

#%%

#daily_folder = r'2022-06-15 AuNP 60 control satelites Pd\20220615-182446_Luminescence_Load_grid_size_ROI_19_in_best_center\processed_data_328nm'
#parent_folder = os.path.join(base_folder, daily_folder)
#exceptions_NP = []
#np_col = 10

#daily_folder = r'2022-05-12 (Au NP 60 nm control satelites paladio)\20220513-105850_Luminescence_Load_grid\processed_data'
#parent_folder = os.path.join(base_folder, daily_folder)
#exceptions_NP = []
#np_col = 7 #6

#daily_folder = r'2022-05-11 (Au NP 60 nm satelites paladio)\20220510-171340_Luminescence_Load_grid\processed_data'
#parent_folder = os.path.join(base_folder, daily_folder)
#exceptions_NP = []#'Col_001_NP_041']#'Col_001_NP_016', 'Col_001_NP_055', 'Col_001_NP_065']
#p_col = 7#8

#%%

save_folder = os.path.join(parent_folder, 'Mean Spectrums per NP')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

list_of_folders = os.listdir(parent_folder)
list_of_folders_NP = [f for f in list_of_folders if re.search('Col',f)]

print(len(list_of_folders_NP)-len(exceptions_NP))

plt.figure()


NP = []
Col = []

color = []
for e in sns.color_palette():
    color.append(e)

size_roi = 19

file = os.path.join(parent_folder, list_of_folders_NP[0], 'pl_in_bins', 'londa_%s.dat'%list_of_folders_NP[0])
londa = np.loadtxt(file)
N = len(londa)

matrix_mean_spec_per_NP = np.zeros((np_col, N))
count = np.zeros(np_col)

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
    
    file = os.path.join(folder, 'pl_in_bins', 'londa_%s.dat'%NP_folder)
    londa = np.loadtxt(file)
    N = len(londa)
    
    desired_stokes = np.where((londa >= 543) & (londa <= 560))
    desired_antistokes = np.where((londa >= 515) & (londa <= 520))
    
    londa_stokes = londa[desired_stokes]
    londa_antistokes = londa[desired_antistokes]
    
    file = os.path.join(folder, 'pl_in_bins', 'all_bins_%s.dat'%NP_folder)
    matrix_specs_bins = np.loadtxt(file)
    totalbins = matrix_specs_bins.shape[1]
    
   # matrix_stokes = np.zeros((len(londa_stokes), totalbins))
   # matrix_antistokes = np.zeros((len(londa_antistokes), totalbins))
    
    stokes_int = np.zeros(totalbins)
    antistokes_int = np.zeros(totalbins)
    
    sum_bin_stokes = np.zeros(len(desired_stokes))
    sum_bin_antistokes = np.zeros(len(desired_antistokes))
    sum_spec = np.zeros(len(londa))
    
    for i in range(totalbins):
        
        spec = matrix_specs_bins[:, i]/size_roi
        sum_spec = sum_spec + spec
        
    sum_spec = sum_spec/totalbins
    
    if np.where(NP_unique == range(np_col)):
        count[NP_unique] = count[NP_unique] + 1
        matrix_mean_spec_per_NP[NP_unique, :] = matrix_mean_spec_per_NP[NP_unique, :] + sum_spec
        
    plt.plot(londa, sum_spec, color = color[NP_unique])
    plt.show()
    
plt.savefig(os.path.join(save_folder, 'spectrums_per_NP.png'))
    
#%%
plt.figure()

for i in range(np_col):
    
    spec = matrix_mean_spec_per_NP[i, :]/count[i]
    data = np.array([londa, spec]).T
    np.savetxt(os.path.join(save_folder, 'mean_spectrums_per_NP_%d.txt'%i), data, header = 'wavelegnth (nm), mean counts')
    plt.plot(londa, spec)
    
    plt.show()
plt.savefig(os.path.join(save_folder, 'mean_spectrums_per_NP.png'))