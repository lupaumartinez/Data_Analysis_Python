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

def compare_spectrum(common_path_stokes, common_path_scattering, save_folder, name_col, fit_bool):
    
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
        
    L = np.min([L,M])
        
    NP_list = np.zeros(L)
    max_wavelength_PL = np.zeros(L)
    max_wavelength_sca = np.zeros(L)
    
    mode_one_sca_fiting = np.zeros(L)
    mode_two_sca_fiting = np.zeros(L)
    
    mode_one_PL_fiting = np.zeros(L)
    mode_two_PL_fiting = np.zeros(L)
            
    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        number_NP = int(NP.split('_')[-1][-1])
        file_scattering = 'NP_%03d.txt'%(number_NP)
    
      #  NP_sca = list_of_folders_scattering[i].split('NP _')[-1]
      #  NP_sca = NP_sca.split('.')[0]
      
        print('Ploteo NP_%s'%(NP))
         
        name_file = os.path.join(common_path_stokes,  list_of_folders[i])
        
        a = np.loadtxt(name_file, skiprows=1)
        wavelength = a[:, 0]
        luminiscence = a[:, 1]
        luminiscence_NP_substrate = a[:, 2]
        luminiscence_susbtrate = a[:, 3]
   
      #  name_file_scattering = os.path.join(common_path_scattering, list_of_folders_scattering[i])
        name_file_scattering = os.path.join(common_path_scattering,  file_scattering)
        
        b = np.loadtxt(name_file_scattering, skiprows=1)
        wavelength_sca = b[:, 0]
        scattering = b[:, 1]
        
        PL_all = (luminiscence_NP_substrate-min(luminiscence_NP_substrate))/(max(luminiscence_NP_substrate)-min(luminiscence_NP_substrate))
        
        PL = (luminiscence-min(luminiscence))/(max(luminiscence)-min(luminiscence))
        sca = (scattering-min(scattering))/(max(scattering)-min(scattering))
        
        if fit_bool:
            
            fit_sca, wavelength_fitted, mode_lspr_one, mode_lspr_two = fitting_signal2(wavelength_sca, sca, 530)
            mode_one_sca_fiting[i] = mode_lspr_one
            mode_two_sca_fiting[i] = mode_lspr_two
            
            fit_PL, wavelength_fitted_PL, mode_lspr_PL_one, mode_lspr_PL_two = fitting_signal2(wavelength, PL, 540)
            mode_one_PL_fiting[i] = mode_lspr_PL_one
            mode_two_PL_fiting[i] = mode_lspr_PL_two
        
        NP_list[i] = number_NP
        max_wavelength_PL[i] = wavelength[np.argmax(PL)]
        max_wavelength_sca[i] = wavelength_sca[np.argmax(sca)]
        
        plt.figure()
        
        plt.title('NP_%s'%(NP))
        plt.plot(wavelength, PL , label = 'PL')
       # plt.plot(wavelength, PL_all , label = 'PL NP + substrate + water')
        plt.plot(wavelength_sca, sca, label = 'Scattering')
        
        if fit_bool:
            plt.plot(wavelength_fitted, fit_sca,  'r--', label = 'Fit Scattering')
            plt.plot(wavelength_fitted_PL, fit_PL,  'k--', label = 'Fit PL')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right') 
        
        #plt.xlim(500, 1000)
       # plt.ylim(0.5, 1.05)
        
        figure_name = os.path.join(save_folder, 'compare_PL_vs_Scattering_%s_NP_%s.png'%(name_col, NP)) 
        
        #plt.show()
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        quotient_PL_Scattering(name_col, NP, wavelength, PL, wavelength_sca, sca)
       # quotient_PL_Scattering(name_col, NP, wavelength, PL_all, wavelength_sca, sca)
       
    plt.figure()
    plt.title('%s'%(name_col))
    plt.plot(NP_list, max_wavelength_PL, 'o', label = 'PL')
    plt.plot(NP_list, max_wavelength_sca, 'o', label = 'Scattering')
    
    if fit_bool:
        plt.plot(NP_list, mode_one_sca_fiting, 'o', label = 'mode_one_sca_fiting')
        plt.plot(NP_list, mode_two_sca_fiting, 'o', label = 'mode_two_sca_fiting')
        
        plt.plot(NP_list, mode_one_PL_fiting, 'o', label = 'mode_one_PL_fiting')
        plt.plot(NP_list, mode_two_PL_fiting, 'o', label = 'mode_two_PL_fiting')
    
    plt.xlabel('NP')
    plt.ylabel('Max wavelength (nm)')
    plt.ylim(530, 750)
    plt.legend()
    figure_name = os.path.join(save_folder, 'max_wavelength_PL_vs_Scattering_%s.png'%(name_col)) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    name = os.path.join(save_folder,  'max_wavelength_PL_vs_Scattering_%s.txt'%(name_col))
    
    if fit_bool:
        
        data = np.array([NP_list, max_wavelength_PL, max_wavelength_sca, mode_one_PL_fiting, mode_two_PL_fiting, mode_one_sca_fiting, mode_two_sca_fiting]).T
        header_txt = 'NP, max wavelength PL (nm), max wavelength Scattering (nm), max wavelength PL fit one (nm), max wavelength PL fit two (nm), max wavelength sca fit one (nm), max wavelength sca fit wo (nm)'
        
    else:
    
        data = np.array([NP_list, max_wavelength_PL, max_wavelength_sca]).T
        header_txt = 'NP, max wavelength PL (nm), max wavelength Scattering (nm)'

    
    np.savetxt(name, data, header = header_txt)
                    
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

def quotient_PL_Scattering(name_col, NP, wavelength_PL, intensity_PL, wavelength_sca, intensity_sca):

    lower_lambda = 547
    upper_lambda = 750
    step = 1000
    
    wavelength_new = np.linspace(lower_lambda, upper_lambda, step)

    desired_range = np.where((wavelength_PL>=lower_lambda) & (wavelength_PL<=upper_lambda))
    
    wavelength_PL = wavelength_PL[desired_range]
    intensity_PL = intensity_PL[desired_range]

    new_PL = np.interp(wavelength_new, wavelength_PL, intensity_PL)
    new_sca = np.interp(wavelength_new, wavelength_sca, intensity_sca)
    
    quotient = new_PL/new_sca
    quotient = (quotient-min(quotient))/(max(quotient)-min(quotient))
    
    window_smooth = 51
    q = signal.savgol_filter(quotient, window_smooth, 1, mode = 'mirror')
    
    plt.figure()
    plt.title('NP_%s'%(NP))
    
  #  plt.plot(wavelength_new, new_PL)
  #  plt.plot(wavelength_new, new_sca)
    
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
    header_txt = 'wavelength (nm), PL norm, Scattering norm, Quotient PL/Sca norm, Quotient PL/Sca smooth'
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
    plt.xlim(540, 840)
    plt.legend(loc='upper right', fontsize = 'xx-small')
        
    figure_name = os.path.join(folder, '%s_quotient_PL_vs_Scattering.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close()            
                                       
if __name__ == '__main__':
    
   # base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PySpectrum/'
    
    base_folder = '/media/luciana/F650A60950A5D0A3/lmartinez/'
    
    daily_folder_bulk = '2020-01-08 (lampara IR PySpectrum y espejo de oro)/espejo_oro/barrido_z_1um/processed_data_luminiscence_steps_and_glue/Col_001/luminescence_steps/Luminescence_Steps_Spectrum_Col_001_NP_014.txt'
    file_bulk = os.path.join(base_folder, daily_folder_bulk)
    data_bulk = np.loadtxt(file_bulk, skiprows = 1)
    wave_bulk =  data_bulk[:, 0]
    spectrum_bulk = data_bulk[:, 1]
    spectrum_bulk = spectrum_bulk/max(spectrum_bulk)
        
    daily_folder_1 = '2020-02-06 (AuNPz 60 nm growth)/20200206-165513_Luminescence_Steps_10x8'
    #daily_folder_1 = '2020-02-06 (AuNPz 60 nm growth)/fila_destruccion_scan_min_Col_001_scan_max_Col_002'
    
    save_folder = os.path.join(base_folder, daily_folder_1, 'fig_compare_PL_Scattering_fitting_lorentz')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    save_folder_quotient = os.path.join(save_folder, 'fig_quotient_PL_Scattering')    
    if not os.path.exists(save_folder_quotient):
        os.makedirs(save_folder_quotient)
        
    list_number_col = 1, 3, 7, 8, 4, 5
    
    fit_bool = True
        
    for number_col in list_number_col:
        
        name_col = 'Col_%03d'%(number_col)
        
        print('Analize ', name_col)
        
        daily_folder = 'processed_data_luminiscence_sustrate_bkg_False'
        
        common_path = os.path.join(base_folder, daily_folder_1, daily_folder, name_col)
        common_path_stokes = os.path.join(common_path, 'luminescence_steps')
        
        daily_folder_sca = '2020-02-07 (Scattering AuNPz 60 nm growth)/lampara_IR_unpol_1s/photos/Col_%03d/normalized_Col_%03d'%(number_col,number_col)
     #   daily_folder_sca = '2020-02-07 (Scattering AuNPz 60 nm growth)/lampara_IR_PY_4s/photos/Col_%03d/normalized_Col_%03d'%(number_col,number_col)
        common_path_scattering = os.path.join(base_folder, daily_folder_sca)
        
        compare_spectrum(common_path_stokes, common_path_scattering, save_folder, name_col, fit_bool = fit_bool)  
        compare_quotient(save_folder_quotient, name_col, wave_bulk, spectrum_bulk)
        