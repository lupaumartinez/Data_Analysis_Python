#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:02:07 2019

@author: luciana
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:03:37 2019

@author: Luciana
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

from Fit_raman_water import lorentz, fit_lorentz, fit_signal_raman_test, three_lorentz, calc_r2, fit_signal_raman_test_two_modes, four_lorentz

def compare_spectrum(common_path_stokes, common_path_scattering, save_folder, name_col, number_NP, fit_bool):
        
    number_col = int(name_col.split('_')[-1])
     
    list_of_folders = os.listdir(common_path_stokes)
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Files from PL', L, list_of_folders)
    
    list_of_folders_scattering = os.listdir(common_path_scattering)
    list_of_folders_scattering = [f for f in list_of_folders_scattering if re.search('NP',f)]
    list_of_folders_scattering.sort()
    M = len(list_of_folders_scattering)  
    
    print('Files from Scattering', M, list_of_folders_scattering)
    
    if L == M:
    
        print('Plot PL, compare with Scattering')
        
    else:
        
        print('The files do not have the same NP. Check files')
        
  #  NP_list = np.zeros(L)
  #  max_wavelength_PL = np.zeros(L)
  #  max_wavelength_sca = np.zeros(L)
    
    mode_one_sca_fiting = np.zeros(L)
    mode_two_sca_fiting = np.zeros(L)
    
    mode_one_PL_fiting = np.zeros(L)
    mode_two_PL_fiting = np.zeros(L)
            
    for i in range(M):
        
        NP = list_of_folders_scattering[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        NP_PL = (number_col- 1)*number_NP + int(NP)
        NP_PL = '%03d'%NP_PL
        
        print('Ploteo %s_NP_%s'%(name_col, NP_PL))
        
        file_PL = 'Luminescence_Steps_Spectrum_%s_NP_%s.txt'%(name_col, NP_PL)
        name_file = os.path.join(common_path_stokes,  file_PL)
        a = np.loadtxt(name_file, skiprows=1)
        wavelength = a[:, 0]
        luminiscence = a[:, 1]
   
    #   luminiscence_NP = a[:, 1]
    #   luminiscence_NP_substrate = a[:, 2]
    #   luminiscence_susbtrate = a[:, 3]
   
        name_file_scattering = os.path.join(common_path_scattering,  list_of_folders_scattering[i])
        b = np.loadtxt(name_file_scattering, skiprows=1)
        wavelength_sca = b[:, 0]
        scattering = b[:, 1]
        
     #   PL_all = (luminiscence_NP_substrate-min(luminiscence_NP_substrate))/(max(luminiscence_NP_substrate)-min(luminiscence_NP_substrate))
        
        min_PL = min(luminiscence[np.where(wavelength> 546)])
        
        PL =  (luminiscence-min_PL)/(max(luminiscence)-min_PL)
        sca = (scattering-min(scattering))/(max(scattering)-min(scattering))
        
        if fit_bool:
            
            fit_sca, wavelength_fitted, mode_lspr_one, mode_lspr_two = fitting_signal2(wavelength_sca, sca, 530)
            mode_one_sca_fiting[i] = mode_lspr_one
            mode_two_sca_fiting[i] = mode_lspr_two
            
            fit_PL, wavelength_fitted_PL, mode_lspr_PL_one, mode_lspr_PL_two = fitting_signal2(wavelength, PL, 540)
            mode_one_PL_fiting[i] = mode_lspr_PL_one
            mode_two_PL_fiting[i] = mode_lspr_PL_two
        
       # NP_list[i] = int(NP)
       # max_wavelength_PL[i] = wavelength[np.argmax(PL)]
       # max_wavelength_sca[i] = wavelength_sca[np.argmax(sca)]
        
        plt.figure()
        
        plt.title('%s_NP_%s'%(name_col, NP))
        plt.plot(wavelength, PL , label = 'PL')
        plt.plot(wavelength_sca, sca, label = 'Scattering')
        
       # if fit_bool:
        #    plt.plot(wavelength_fitted, fit_sca,  'r--', label = 'Fit Scattering')
         #   plt.plot(wavelength_fitted_PL, fit_PL,  'k--', label = 'Fit PL')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right') 
        
        #plt.xlim(500, 1000)
       # plt.ylim(0.5, 1.05)
        
        figure_name = os.path.join(save_folder, 'compare_PL_vs_Scattering_%s_NP_%s.png'%(name_col, NP_PL)) 
        
        #plt.show()
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        quotient_PL_Scattering(name_col, NP, wavelength, PL, wavelength_sca, sca, plot_bool = False)
       # quotient_PL_Scattering(name_col, NP, wavelength, PL_all, wavelength_sca, sca)
       
  #  plt.figure()
  #  plt.title('%s'%(name_col))
  #  plt.plot(NP_list, max_wavelength_PL, 'o', label = 'PL')
  #  plt.plot(NP_list, max_wavelength_sca, 'o', label = 'Scattering')
    
   # if fit_bool:
    #    plt.plot(NP_list, mode_one_sca_fiting, 'o', label = 'mode_one_sca_fiting')
     #   plt.plot(NP_list, mode_two_sca_fiting, 'o', label = 'mode_two_sca_fiting')
        
      #  plt.plot(NP_list, mode_one_PL_fiting, 'o', label = 'mode_one_PL_fiting')
       # plt.plot(NP_list, mode_two_PL_fiting, 'o', label = 'mode_two_PL_fiting')
    
   # plt.xlabel('NP')
   # plt.ylabel('Max wavelength (nm)')
   # plt.ylim(530, 620)
   # plt.legend()
   # figure_name = os.path.join(save_folder, 'max_wavelength_PL_vs_Scattering_%s.png'%(name_col)) 
   # plt.savefig(figure_name, dpi = 400)
   # plt.close()
    
   # name = os.path.join(save_folder,  'max_wavelength_PL_vs_Scattering_%s.txt'%(name_col))
    
    #if fit_bool:
        
     #   data = np.array([NP_list, max_wavelength_PL, max_wavelength_sca, mode_one_PL_fiting, mode_two_PL_fiting, mode_one_sca_fiting, mode_two_sca_fiting]).T
      #  header_txt = 'NP, max wavelength PL (nm), max wavelength Scattering (nm), max wavelength PL fit one (nm), max wavelength PL fit two (nm), max wavelength sca fit one (nm), max wavelength sca fit wo (nm)'
        
    #else:
    
     #   data = np.array([NP_list, max_wavelength_PL, max_wavelength_sca]).T
      #  header_txt = 'NP, max wavelength PL (nm), max wavelength Scattering (nm)'

    #np.savetxt(name, data, header = header_txt)
                    
    return

def fitting_signal(wavelength, spectrum):
    
    print('Fit Signal')
    
    I = 1
    init_londa = 550
    init_width = 50
    C = 0

    init_parameters = np.array([I, init_width, init_londa, C], dtype=np.double)
    
    bounds = ([0.9, 0, 520, 0], [1.1, 100, 650, 0.01])  

    best_lorentz, err = fit_lorentz(wavelength, spectrum, init_parameters, bounds)
    
    fit_scattering = lorentz(wavelength, *best_lorentz)

    return fit_scattering, wavelength

def fitting_signal2(wavelength, spectrum, first_wave):
    
    print('Fit Signal')
    
    I = 1
    init_londa = 550
    init_width = 100
    
    Imode2 = 1
    init_londa2 = 600        
    init_width2 = 100

    I2 = 0.01#100#2500#1000
    I3 = 0.01#100#100#0#0
    C = 0.001#400
    init_parameters = np.array([I, init_width, init_londa, Imode2, init_width2, init_londa2, I2, I3, C], dtype=np.double)
    
    bounds = ([0.9, 0, 530, 0.9, 0, 550, 0, 0, 0], [2, 100, 590, 2, 100, 730, 0.3, 0.3, 0.1])  
    
    last_wave = 800
    
    wavelength_fitted, lorentz_fitted, best_parameters, r2 = fit_signal_raman_test_two_modes(wavelength, spectrum, first_wave, last_wave, init_parameters, bounds)

    mode_lspr_one = best_parameters[2]
    mode_lspr_two = best_parameters[5]
        
    print('Fitting:', best_parameters[0], mode_lspr_one, best_parameters[3], mode_lspr_two)
        
    return lorentz_fitted, wavelength_fitted, mode_lspr_one, mode_lspr_two

def quotient_PL_Scattering(name_col, NP, wavelength_PL, intensity_PL, wavelength_sca, intensity_sca, plot_bool):

    lower_lambda = 547
    upper_lambda = 625
    step = 1000
    
    wavelength_new = np.linspace(lower_lambda, upper_lambda, step)

 #   desired_range = np.where((wavelength_PL>=lower_lambda) & (wavelength_PL<=upper_lambda))  
  #  wavelength_PL = wavelength_PL[desired_range]
  #  intensity_PL = intensity_PL[desired_range]
    
    new_PL = np.interp(wavelength_new, wavelength_PL, intensity_PL)
    new_sca = np.interp(wavelength_new, wavelength_sca, intensity_sca)
    
    quotient = new_PL/new_sca
    quotient = (quotient-min(quotient))/(max(quotient)-min(quotient))
    
 #   window_smooth = 51
 #   q = signal.savgol_filter(quotient, window_smooth, 1, mode = 'mirror')
    
    npol = 3
    p = np.polyfit(wavelength_new , quotient, npol)
    q = np.polyval(p, wavelength_new )
    
    if plot_bool:
                  
        plt.figure()
        plt.title('NP_%s'%(NP))
        plt.plot(wavelength_new, quotient)
        plt.plot(wavelength_new, q, 'k--')  
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Quotient PL/Scattering')
        plt.ylim(-0.02, 1.02)
            
        figure_name = os.path.join(save_folder, 'fig_quotient_PL_Scattering', 'quotient_PL_vs_Scattering_%s_NP_%s.png'%(name_col, NP))
        plt.savefig(figure_name , dpi = 400)
        plt.close()
    
    name = os.path.join(save_folder, 'fig_quotient_PL_Scattering', 'data_%s_NP_%s.txt'%(name_col, NP))
    data = np.array([wavelength_new, new_PL, new_sca, quotient, q]).T
    header_txt = 'wavelength (nm), PL norm, Scattering norm, Quotient PL/Sca norm, Quotient PL/Sca Fit'
    np.savetxt(name, data, header = header_txt)

    return

def compare_quotient(folder, name_col, wave_bulk, spectrum_bulk):
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('data_%s'%name_col,f) ]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    plt.figure()
    plt.title('%s'%(name_col))

    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        file = os.path.join(folder, list_of_folders[i])
        data = np.loadtxt(file, comments = '#')
                          
        wave = data[:, 0]
        quotient = data[:, 3]
        q = data[:,4]

        plt.plot(wave, quotient, label = 'NP_%s'%(NP))
        plt.plot(wave, q, 'k--')
        
    plt.plot(wave_bulk, spectrum_bulk, 'r--', label = 'gold bulk')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Quotient PL/Scattering')
    plt.ylim(-0.02, 1.02)
    plt.xlim(540, 650)
    plt.legend(loc='upper right', fontsize = 'xx-small')
        
    figure_name = os.path.join(folder, '%s_quotient_PL_vs_Scattering.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close() 

    return

def compare_product(folder, name_col, wave_bulk, spectrum_bulk):
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('data_%s'%name_col,f) ]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    plt.figure()
    plt.title('%s'%(name_col))

    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        file = os.path.join(folder, list_of_folders[i])
        data = np.loadtxt(file, comments = '#')
                          
        wave = data[:, 0]
        norm_PL = data[:, 1]
        norm_sca = data[:, 2]
        
        PL_bulk = np.interp(wave, wave_bulk, spectrum_bulk)
        product = norm_sca*PL_bulk

        plt.plot(wave, product, '-', label = 'NP_%s'%(NP))
        plt.plot(wave, norm_PL , '--', label = 'NP_%s'%(NP))
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering*PL bulk')
    plt.ylim(-0.02, 1.02)
    plt.xlim(540, 650)
    plt.legend(loc='upper right', fontsize = 'xx-small')
        
    figure_name = os.path.join(folder, '%s_product_Scattering_PLbulk.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close() 

    return

def compare_difference(folder, name_col, wave_bulk, spectrum_bulk):
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('data_%s'%name_col,f) ]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    plt.figure()
    plt.title('%s'%(name_col))

    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        file = os.path.join(folder, list_of_folders[i])
        data = np.loadtxt(file, comments = '#')
                          
        wave = data[:, 0]
        norm_PL = data[:, 1]
        norm_sca = data[:, 2]

        plt.plot(wave,  norm_sca - norm_PL , '-', label = 'NP_%s'%(NP))
        
#    plt.plot( wave_bulk, spectrum_bulk , 'r--', label = 'gold bulk')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering - PL')
   # plt.ylim(-0.02, 1.02)
    plt.xlim(540, 650)
    plt.legend(loc='upper right', fontsize = 'xx-small')
        
    figure_name = os.path.join(folder, '%s_difference_Scattering_PLbulk.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close() 

    return              
                                       
if __name__ == '__main__':
    
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
        
  #  daily_folder_PL = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/pre_growth/20210809-150932_Luminescence_Steps_10x12'
  #  daily_folder_sca = '2021-08 (Growth PL circular)/2021-08-04 (pre growth, PL circular)/Scattering_unpol/photos'
    
    daily_folder_PL = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)/PL/20210818-174530_Luminescence_Steps_10x12'
    daily_folder_sca = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)/Scattering_unpol/photos'
     
    save_folder = os.path.join(base_folder, daily_folder_PL, 'fig_compare_PL_Scattering')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    save_folder_quotient = os.path.join(save_folder, 'fig_quotient_PL_Scattering')    
    if not os.path.exists(save_folder_quotient):
        os.makedirs(save_folder_quotient)
        
    list_number_col = 1,2, 3, 4,5,6,7, 8, 9,10,11,12
    number_NP = 10 #por columna
    
    fit_bool = False
        
    for number_col in list_number_col:
        
        name_col = 'Col_%03d'%(number_col)
        
        print('Analize ', name_col)
        
        folder_PL = 'processed_data_luminiscence_sustrate_bkg_False'
        common_path = os.path.join(base_folder, daily_folder_PL, folder_PL, name_col)
        common_path_stokes = os.path.join(common_path, 'luminescence_steps')
        
        folder_sca = 'Col_%03d/normalized_Col_%03d'%(number_col,number_col)
        common_path_scattering = os.path.join(base_folder, daily_folder_sca, folder_sca)
        
        compare_spectrum(common_path_stokes, common_path_scattering, save_folder, name_col, number_NP, fit_bool = fit_bool)
        
    #%%
        
    daily_folder_bulk = '2020-01-08 (lampara IR PySpectrum y espejo de oro)/espejo_oro/barrido_z_1um/processed_data_luminiscence_steps_and_glue/Col_001/luminescence_steps/Luminescence_Steps_Spectrum_Col_001_NP_014.txt'
    file_bulk = os.path.join(base_folder, daily_folder_bulk)
    data_bulk = np.loadtxt(file_bulk, skiprows = 1)
    wave_bulk =  data_bulk[:, 0]
    spectrum_bulk = data_bulk[:, 1]
    
    lower_lambda = 547
    upper_lambda = 625
    step = 1000

    desired_range = np.where((wave_bulk>=lower_lambda) & (wave_bulk<=upper_lambda))  
    wavelength_bulk = wave_bulk[desired_range]
    intensity_bulk = spectrum_bulk[desired_range]
    
    spectrum_bulk = (intensity_bulk - min(intensity_bulk ))/(max(intensity_bulk) - min(intensity_bulk ))
        
    for number_col in list_number_col:
        
        name_col = 'Col_%03d'%(number_col)
        
        compare_quotient(save_folder_quotient, name_col, wavelength_bulk, spectrum_bulk)
   #     compare_product(save_folder_quotient, name_col, wavelength_bulk, spectrum_bulk)
   #     compare_difference(save_folder_quotient, name_col, wavelength_bulk, spectrum_bulk)