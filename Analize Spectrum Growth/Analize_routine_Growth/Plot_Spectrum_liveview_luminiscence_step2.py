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
        
def post_process_spectrum(common_path):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(common_path,f))]
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Plot Max Wavelength Stokes')
    plt.figure()
    
    for i in range(L):
        
        NP = list_of_folders[i].split('_')[-1]

        path_folder = os.path.join(common_path, list_of_folders[i])
        list_of_files = os.listdir(path_folder)
        
        for file in list_of_files:
            if file.startswith('data_plots'):
                name_file = os.path.join(path_folder, file)
                a = np.loadtxt(name_file, skiprows=1)
                time = a[:, 0]
                max_wavelength = a[:, 1]
                plt.plot(time , max_wavelength, 'o', label = 'NP_%s'%(NP))
                
    plt.xlabel('Time (s)')
    plt.ylabel('$\u03BB_{max}$ (nm)')
 #   plt.xlim(-2,  270)
  #  plt.ylim(543, 620) 
    plt.legend(loc='upper right', fontsize = 'xx-small') 
    figure_name = os.path.join(common_path, 'all_data_max_wavelength.png') 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
                
    return

def post2_process_spectrum(common_path, trace_BS_bool):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(common_path,f))]
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Plot integrates')
 #   f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    for i in range(L):
        
        NP = list_of_folders[i].split('_')[-1]

        path_folder = os.path.join(common_path, list_of_folders[i])
        list_of_files = os.listdir(path_folder)
        
        for file in list_of_files:
            if file.startswith('data_plots'):
                name_file = os.path.join(path_folder, file)
                a = np.loadtxt(name_file, skiprows=1)
                time = a[:, 0]
                max_intensity_stokes = a[:, 2]
                integrate_stokes = a[:, 3]
                integrate_antistokes = a[:, 4]
                
        if trace_BS_bool:
            if file.startswith('data_power'):
                name_file = os.path.join(path_folder, file)
                b = np.loadtxt(name_file, skiprows=1)
                time_BFP = b[:, 0]
                power_BFP = b[:, 1]
            
          #  ax0.plot(time , max_intensity_stokes, '--o', label = 'NP_%s'%(NP))
        ax1.plot(time , integrate_stokes, '-', label = 'NP_%s'%(NP))
        ax2.plot(time , integrate_antistokes, '-', label = 'NP_%s'%(NP))
        
        if trace_BS_bool:
            ax3.plot(time_BFP, power_BFP, '-', label = 'NP_%s'%(NP))
            
    ax1.set_xlabel('Time (s)')
    ax2.set_xlabel('Time (s)')
    ax3.set_xlabel('Time (s)')
    ax1.set_ylabel('Integrate Stokes')
    ax2.set_ylabel('Integrate Anti-Stokes')
    ax3.set_ylabel('Power BFP (mW)')
    plt.legend(loc='upper right', fontsize =  'xx-small') 
    f.set_tight_layout(True)
    figure_name = os.path.join(common_path, 'all_data.png') 
    plt.savefig(figure_name , dpi = 400)
    plt.show()
                
    return

def post3_process_spectrum(common_path):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(common_path,f))]
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Plot SPR Wavelength Stokes from fitting raman of water on live')
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
                plt.plot(time_londa, londa_max_pl, '*', label = 'NP_%s'%(NP))
                
            if file.startswith('data_live_poly_fitting'):
                name_file = os.path.join(path_folder, file)
                a = np.loadtxt(name_file, skiprows=1)
                time_poly = a[:, 0]
                max_poly = a[:, 1]
                plt.plot(time_poly, max_poly, 'o', label = 'NP_%s'%(NP))
    
   # plt.xlim(-2, 270)
    #plt.ylim(543, 620)            
    plt.xlabel('Time (s)')
    plt.ylabel('$\u03BB_{max}$ (nm)')
    plt.legend(loc='upper right', fontsize =  'xx-small') 
    figure_name = os.path.join(common_path, 'all_live_fitting_SPR.png') 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
                
    return

def post4_process_spectrum(common_path, moment):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(common_path,f))]
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Plot Stokes and fitting Lorentz', moment)
   
    save_folder = os.path.join(common_path, 'all_data_%s_Stokes'%moment)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    plt.figure()
    
    for i in range(L):
        
        NP = list_of_folders[i].split('_')[-1]

        path_folder = os.path.join(common_path, list_of_folders[i])
        list_of_files = os.listdir(path_folder)
        
                      
        for file in list_of_files:
            
            if file.startswith('data_%s_Stokes'%moment):
                name_file = os.path.join(path_folder, file)
                a = np.loadtxt(name_file, skiprows=1)
                wavelength = a[:, 0]
                stokes = a[:, 1]
                lorentz = a[:, 2]
                
                plt.plot(wavelength, stokes/max(stokes), label = 'NP_%s'%(NP))
                plt.plot(wavelength, lorentz/max(lorentz), '--k')   
                
                header_text = 'Wavelength (nm), Intensity, Fit Lorentz'
                name = os.path.join(save_folder, '%s_Stokes_NP_%s.txt'%(moment,NP))
                np.savetxt(name, a, header = header_text)
                          
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend(loc='upper right') 
   # plt.xlim(500, 640)
   # plt.ylim(0.5, 1.05)
    figure_name = os.path.join(common_path, 'all_%s_Stokes.png'%moment) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
                        
    return


def post5_process_spectrum(common_path):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(common_path,f))]
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Plot Total time and LSPR final of growth')
    
    all_data = np.zeros((L, 5))
    all_data_poly = np.zeros((L, 5))
    
    save_folder = os.path.join(common_path, 'all_data_growth')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    plt.figure()
    
    for i in range(L):
        
        NP = list_of_folders[i].split('_')[-1]

        path_folder = os.path.join(common_path, list_of_folders[i])
        list_of_files = os.listdir(path_folder)
                      
        for file in list_of_files:
            
            if file.startswith('data_growth_final'):
                
                name_file = os.path.join(path_folder, file)
                a = np.loadtxt(name_file, skiprows=1)
                
                total_time_BS = a[0]
                mean_power_BS = a[1]
                
                total_time = a[2]
                final_spr = a[3]
                
                all_data[i, 0] = NP
                all_data[i, 1:] = a
                
                plt.plot(total_time, final_spr, 'o', label = 'NP_%s'%(NP))
                
            if file.startswith('data_growth_poly_final'):
                
                name_file = os.path.join(path_folder, file)
                a = np.loadtxt(name_file, skiprows=1)
                
                total_time_BS = a[0]
                mean_power_BS = a[1]
                
                total_time = a[2]
                final_spr = a[3]
                
                all_data_poly[i, 0] = NP
                all_data_poly[i, 1:] = a
                
                plt.plot(total_time, final_spr, '*', label = 'NP_%s'%(NP))
               
    plt.xlabel('Total time (s)')
    plt.ylabel('Final Wavelength LSPR (nm)')
    plt.legend(loc='upper right') 
   # plt.xlim(500, 640)
   # plt.ylim(0.5, 1.05)
    figure_name = os.path.join(common_path, 'all_data_growth.png')
    plt.savefig(figure_name , dpi = 400)
    plt.close()            
                
    
    header_text = 'NP, Total Time BS (s), Mean Power BS (mW), Total Time Spectrum (s), Max Poly SPR live (nm)'
    name = os.path.join(save_folder,  'all_data_poly_growth.txt')
    np.savetxt(name, all_data_poly, header = header_text)
    
    header_text = 'NP, Total Time BS (s), Mean Power BS (mW), Total Time Spectrum (s), Londa SPR live (nm)'
    name = os.path.join(save_folder,  'all_data_growth.txt')
    np.savetxt(name, all_data, header = header_text)
    
    return

    
    
if __name__ == '__main__':
    
   # base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PySpectrum/'

   # daily_folder = '2020-02-06 (AuNPz 60 nm growth)/Growth_HAuCl4_1mW/20200206-151616_Growth_6x1_Col3'
   
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'

    daily_folder = '2021-03-26 (Growth)/20210326-132548_Growth_12x2_560'
    
    common_path = os.path.join(base_folder, daily_folder, 'processed_data_sustrate_bkg_True')
    
    post_process_spectrum(common_path)
    post2_process_spectrum(common_path)
    post3_process_spectrum(common_path)
    post4_process_spectrum(common_path, 'initial')
    post4_process_spectrum(common_path, 'final')
    post5_process_spectrum(common_path)
    
    