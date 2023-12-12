# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:53:47 2019

@author: Luciana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def process_spectrum(folder, common_path, sustrate_bkg_bool):

    NP = folder.split('Spectrum_')[-1]
    
    if NP == folder: #esto es para poder analizar un video individual que no sea de la rutina Growth
        NP = 'one'

    save_folder = os.path.join(common_path,'%s'%NP)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    name_spectrum = []
    specs = []
    
    name_spectrum_bkg = []
    specs_bkg = []
    
    list_of_files = os.listdir(folder)
    list_of_files.sort()
    
    time_poly = []
    max_poly = []
        
    for file in list_of_files:
        
        if file.startswith('Calibration_Shamrock'):
            name_file = os.path.join(folder, file)
            wavelength = np.loadtxt(name_file)
            
        if file.startswith('Line_Spectrum'):
            name_file = os.path.join(folder, file)
            a = file.split('step_')[-1]
            b = int(a.split('.txt')[0])
            name_spectrum.append(b)
            specs.append(np.loadtxt(name_file))
            
        if sustrate_bkg_bool:
            
            if file.startswith('Background_Line_Spectrum'):
                name_file = os.path.join(folder, file)
                a = file.split('step_')[-1]
                b = int(a.split('.txt')[0])
                name_spectrum_bkg.append(b)
                specs_bkg.append(np.loadtxt(name_file))
                 
        if file.startswith('Max_PolySpectrum'):
                name_file = os.path.join(folder, file)
                data = np.loadtxt(name_file, comments='#')
                time_poly.append(data[0])
                max_poly.append(data[1])
            
    specs = [specs for _,specs in sorted(zip(name_spectrum,specs))]            
    name_spectrum = sorted(name_spectrum)
    
    specs_bkg = [specs_bkg for _,specs_bkg in sorted(zip(name_spectrum_bkg,specs_bkg))]      
        
    L = len(specs)
            
    print('Number:', NP, 'Spectra acquired:', L)
    
    header_text = 'Wavelength (nm), Intensity'
    
    initial = 0
    final = L
    
    for i in range(initial, final):
        
        spectrum_original = specs[i]
        
        if sustrate_bkg_bool:
            
            spectrum = spectrum_original - specs_bkg[i]
            
        else:
            
            spectrum = spectrum_original

        if i == initial:
            name = os.path.join(save_folder, 'data_initial_%s.txt'%(NP))
            data = np.array([wavelength, spectrum]).T
            np.savetxt(name, data, header = header_text)
            
        if i == initial+1:
            name = os.path.join(save_folder, 'data_post_initial_%s.txt'%(NP))
            data = np.array([wavelength, spectrum]).T
            np.savetxt(name, data, header = header_text)
                
        if i == final - 1:
            name = os.path.join(save_folder, 'data_final_%s.txt'%(NP))
            data = np.array([wavelength, spectrum]).T
            np.savetxt(name, data, header = header_text)
            
        if i == final - 2:
            name = os.path.join(save_folder, 'data_pre_final_%s.txt'%(NP))
            data = np.array([wavelength, spectrum]).T
            np.savetxt(name, data, header = header_text)

    header_fitting = "Total Time Spectrum (s), Max Poly SPR (nm)"
    data_fitting = np.array([time_poly[-1], max_poly[-1]]).T
    name_data = os.path.join(save_folder, 'data_growth_poly_final_%s.txt'%(NP))
    np.savetxt(name_data, data_fitting, header = header_fitting)

    return


def post_process_spectrum(common_path):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(common_path,f))]
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Plot Total time and LSPR final of growth')
    
    all_data_poly = np.zeros((L, 3))
    
    save_folder = os.path.join(common_path, 'all_data_growth')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    for i in range(L):
        
        NP = list_of_folders[i].split('_')[-1]

        path_folder = os.path.join(common_path, list_of_folders[i])
        list_of_files = os.listdir(path_folder)
                      
        for file in list_of_files:
                
            if file.startswith('data_growth_poly_final'):
                
                name_file = os.path.join(path_folder, file)
                a = np.loadtxt(name_file, skiprows=1)

                all_data_poly[i, 0] = NP
                all_data_poly[i, 1:] = a
    
    header_text = 'NP, Total Time Spectrum (s), Max Poly SPR live (nm)'
    name = os.path.join(save_folder, 'all_data_poly_growth.txt')
    np.savetxt(name, all_data_poly, header = header_text)
    
    return

def post4_process_spectrum(common_path, moment):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(common_path,f))]
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
   
    save_folder = os.path.join(common_path, 'all_data_%s'%moment)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for i in range(L):
        
        NP = list_of_folders[i].split('_')[-1]

        path_folder = os.path.join(common_path, list_of_folders[i])
        list_of_files = os.listdir(path_folder)
                      
        for file in list_of_files:
            
            if file.startswith('data_%s'%moment):
                name_file = os.path.join(path_folder, file)
                a = np.loadtxt(name_file, skiprows=1)

                header_text = 'Wavelength (nm), Intensity'
                name = os.path.join(save_folder, '%s_NP_%s.txt'%(moment,NP))
                np.savetxt(name, a, header = header_text)

    return


if __name__ == '__main__':

  #  base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PySpectrum/'
  
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'

   # daily_folder = '2021-06 (Growth)/2021-06-19 (growth)/20210619-130726_Growth_12x1_560_2s_em100'
   # daily_folder = '2021-06 (Growth)/2021-06-19 (growth)/20210618-184443_Growth_12x1_570_2s'
   # daily_folder = '2021-06 (Growth)/2021-06-19 (growth)/20210618-175939_Growth_12x1_580_2s'
    #daily_folder = '2021-06 (Growth)/2021-06-19 (growth)/20210619-142647_Growth_12x1_560_3s_em60' #20210619-140537_Growth_12x1_570_3s_em60' #'20210619-133702_Growth_12x1_580_3s_em60'
    #daily_folder = '2021-06 (Growth)/2021-06-19 (growth)/20210619-163455_Growth_12x1_560_4s_em50' #20210619-155338_Growth_12x1_570_4s_em50' #20210619-152004_Growth_12x1_580_4s_em50'
    
#    daily_folder = '2021-08 (Growth PL circular)/2021-08-04 (pre growth, PL circular)/laser_polarizacion_circular_MAL/20210804-171638_Growth_10x1'
    
 #   daily_folder = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/growth/'
    
  #  list_daily_folder = ['Pruebas_concentracion_HAuCl4', '20210809-165810_Growth_10x2_col7y8_560nm', '20210809-172719_Growth_10x2_col9y10_570nm', '20210809-180739_Growth_10x2_col11y12_580nm',
  #                       '20210809-190759_Growth_10x1_col3_80s', '20210809-192328_Growth_10x1_col4_140s', '20210809-194745_Growth_10x1_col5_200s']
  
    daily_folder = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/post_growth/'
    
    list_daily_folder = ['20210809-203300_Growth_10x1_col7_80seg_agua','20210809-205507_Growth_10x1_col9_80seg_agua',
                         '20210809-211130_Growth_10x1_col11_80seg_agua','20210809-221629_Growth_6x1_col6_80seg_agua', 
                         '20210809-212908_Growth_10x3_col3_80seg_agua', '20210809-212908_Growth_10x3_col4_80seg_agua',
                         '20210809-212908_Growth_10x3_col5_80seg_agua']
    
    n = 4 #a partir de que  archivo analizo
    
    for i in range(n, len(list_daily_folder[n:])+n):
        
        file = list_daily_folder[i]
        
        folder = os.path.join(daily_folder, file)
        
        print(folder)
     
        parent_folder = os.path.join(base_folder, folder)
        list_of_folders = os.listdir(parent_folder)
        list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
        list_of_folders = [f for f in list_of_folders if re.search('Liveview_Spectrum',f)]
        
        list_of_folders.sort()
        
        L_folder = len(list_of_folders)
        
        #INPUTS
        
        sustrate_bkg_bool = False #True Resta la señal de Background (region de mismo size pero fuera de la fibra optica, en pixel 200)
        
        path_to = os.path.join(parent_folder, 'range_growth_sustrate_bkg_%s'%str(sustrate_bkg_bool))
        
        if not os.path.exists(path_to):
            os.makedirs(path_to)
    
        plot_all = False
        for f in list_of_folders:
            folder = os.path.join(parent_folder,f)
            process_spectrum(folder, path_to, sustrate_bkg_bool)
            plot_all = True
         
        #plot_3D(path_to, frame_list, wavelength, matrix_total, NP)
        
        plt.style.use('default')
        if plot_all:
            common_path = path_to
            post_process_spectrum(common_path)
            post4_process_spectrum(common_path, 'final')
            post4_process_spectrum(common_path, 'initial')
            post4_process_spectrum(common_path, 'pre_final')
            post4_process_spectrum(common_path, 'post_initial')
    