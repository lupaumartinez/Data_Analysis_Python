#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:59:12 2019

@author: luciana
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def compare_scattering(common_path_1, common_path_2, save_folder):
     
    name_file_1 = os.path.join(common_path_1, 'max_wavelength.txt')
    data_1 = np.loadtxt(name_file_1, skiprows=1)
    NP_1 = data_1[:, 0]
    max_wavelength_1 = data_1[:, 1]
    
    name_file_2 = os.path.join(common_path_2, 'max_wavelength.txt')
    data_2 = np.loadtxt(name_file_2, skiprows=1)
    NP_2 = data_2[:, 0]
    max_wavelength_2 = data_2[:, 1]
    
    m = min(len(NP_1), len(NP_2))
    NP = []
    dif_wave = []
    for i in range(m):
        if NP_2[i] ==  NP_1[i]:
            NP.append(NP_2[i])
            dif_wave.append(max_wavelength_2[i] - max_wavelength_1[i])
            
    mean = np.mean(dif_wave)
    std = np.std(dif_wave)
            
    print('Difencia', NP, dif_wave, mean, std)
    
    print('Ploteo diferencias')
    
    plt.figure()
    
    plt.hist(dif_wave, bins=5, normed = True, rwidth=0.9, color='C2', label = 'N = %d'%m)
    plt.xlabel('Difference Max Wavelength (nm)')
    plt.ylabel('Frequency')
    
    plt.axvspan(mean - std, mean + std, color = 'green', alpha = 0.3)
    plt.axvline(mean, color = 'red', linestyle = '--')
    plt.text(mean + 0.001*mean, 0.09, ' %2d + %2d nm'%(mean, std), color = 'red')
              
    plt.legend(loc='upper right') 
    
    plt.ylim(0, 0.10)
    plt.xlim(mean - 2*std, mean + 2*std)
    
    figure_name = os.path.join(save_folder, 'hist_diff_compare_Scattering.png') 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
        
    plt.figure()
    
    plt.plot(NP_1,  max_wavelength_1, 'o', label = 'First Scattering')
    plt.plot(NP_2,  max_wavelength_2, 'o', label = 'Last Scattering')
              
    plt.xlabel('NP')
    plt.ylabel('Max Wavelength Scattering (nm)')
    plt.legend(loc='upper right') 
    
   # plt.ylim(530, 580)
    
    figure_name = os.path.join(save_folder, 'compare_Scattering.png') 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
            
    return
    
if __name__ == '__main__':
    
    base_folder_sca = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2019/Mediciones_PySpectrum/'
    
    daily_folder_1 = '2019-10-14 (scattering SS 50 nm)/scattering_SS50nm/'
    daily_folder_2 = '2019-10-21  (growth HAuCl4 1 mW 532nm on SS 50nm PDDA)/scattering post growth/'
    
    number_col = 8
    col_folder = 'col_ 0%d/fig_normalized_col_ 0%d'%(number_col,number_col) 
    
    common_path_1 = os.path.join(base_folder_sca, daily_folder_1, col_folder)
    common_path_2 = os.path.join(base_folder_sca, daily_folder_2, col_folder)
    
    save_folder = os.path.join(common_path_2, 'fig_compare_Scatterings')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    compare_scattering(common_path_1, common_path_2, save_folder)