# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:03:37 2019

@author: Luciana
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
        

def post6_process_spectrum(common_path):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(common_path,f))]
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Plot Error SPR Wavelength Stokes from fitting raman of water on live')
    plt.figure()
    
    for i in range(L):
        
        NP = list_of_folders[i].split('_')[-1]

        path_folder = os.path.join(common_path, list_of_folders[i])
        list_of_files = os.listdir(path_folder)
                      
        for file in list_of_files:
            if file.startswith('data_live_fitting'):
                name_file = os.path.join(path_folder, file)
                a = np.loadtxt(name_file, skiprows=1)
                time_londa = a[:, 0]
                londa_max_pl = a[:, 1]
                
                inf_londa = np.array(londa_max_pl[:-1])
                sup_londa = np.array(londa_max_pl[1:])
                
                diff_londa = np.round(sup_londa - inf_londa,1)
                
                error = round(np.mean(np.abs(diff_londa)))
                
                std = round(np.std(np.abs(diff_londa)))
                
                plt.plot(inf_londa, diff_londa, '*', label = 'NP_%s'%(NP))
                
                print(NP, 'error:', error, 'error std', std)
                
    
   # plt.xlim(-2, 270)
    #plt.ylim(543, 620)            
    plt.xlabel('$\u03BB_{max}$ (nm)')
    plt.ylabel('$\u0394\u03BB_{max}$ (nm)')
    plt.legend(loc='upper right', fontsize =  'xx-small') 
    figure_name = os.path.join(common_path, 'all_error_SPR.png') 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
                
    return
    
    
if __name__ == '__main__':
    
   # base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PySpectrum/'

   # daily_folder = '2020-02-06 (AuNPz 60 nm growth)/Growth_HAuCl4_1mW/20200206-151616_Growth_6x1_Col3'
   
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'

    daily_folder = '2021-03-10 (growth HauCl4 on AuNPz)/20210310-191030_Growth_12x1_585'
    
    common_path = os.path.join(base_folder, daily_folder, 'processed_data_sustrate_bkg_True')
    
    post6_process_spectrum(common_path)
    