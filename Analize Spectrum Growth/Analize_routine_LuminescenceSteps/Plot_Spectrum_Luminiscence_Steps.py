#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:09:15 2019

@author: luciana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sig
from Correct_Step_and_Glue import glue_steps
from Fit_raman_water import fit_signal_raman_test, three_lorentz, calc_r2, fit_signal_raman_test_two_modes, four_lorentz

def smooth_Signal(signal, window, deg, repetitions):
    
    k = 0
    while k < repetitions:
        signal = sig.savgol_filter(signal, window, deg, mode = 'mirror')
        k = k + 1
        
    return signal
    

def average(n, arr):
    
    end = n*int(len(arr)/n)
    arr_mean = np.mean(arr[:end].reshape(-1,n),1)
    
    return arr_mean

def select_signal(spectrum, wavelength, first_wave, end_wave):
    
    desired_range = np.where((wavelength>=first_wave) & (wavelength<=end_wave))
    wavelength = wavelength[desired_range]
    spectrum = spectrum[desired_range]
    
    return spectrum, wavelength

def process_spectrum(folder, common_path, sustrate_bkg_bool, window, deg, repetitions, wavelength, signal_substrate, grade, fig_lim, fit_raman_water):
    
    NP = folder.split('luminescence_')[-1]
    
    col = NP.split('_NP')[0]
    
    save_folder = os.path.join(common_path,'%s'%col) 
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    save_folder_2 = os.path.join(save_folder, 'luminescence_steps') 
    if not os.path.exists(save_folder_2):
        os.makedirs(save_folder_2)
        
    save_folder_3 = os.path.join(save_folder, 'parameters_fit_raman_water') 
    if not os.path.exists(save_folder_3):
        os.makedirs(save_folder_3)
    
    list_of_files = os.listdir(folder)
    
#    file1 = [f for f in list_of_files if re.search('Calibration',f)][0]
    
#    calibration_wavelength = os.path.join(folder, file1)
#    wavelength = np.loadtxt(calibration_wavelength)
    
    file2 = [f for f in list_of_files if re.search('Line_Spectrum',f) and not re.search('Background',f)][0]
    line_spectrum = os.path.join(folder, file2)
    specs = np.loadtxt(line_spectrum)
    
    if sustrate_bkg_bool:
    
        file3 = [f for f in list_of_files if re.search('Background',f)][0]
        line_spectrum_bkg = os.path.join(folder, file3)
        specs_bkg = np.loadtxt(line_spectrum_bkg)
        
    else:
    
        specs_bkg = np.zeros(len(specs))
    
 #   specs = [spectrum_all for _,spectrum_all in sorted(zip(wavelength,specs))]
 #   specs_substrate = [spectrum_all for _,spectrum_all in sorted(zip(wavelength, signal_substrate))]
    
   # wavelength_all = np.sort(wavelength)
   # specs = np.array(specs)
   # specs_substrate = np.array(specs_substrate)
   
    specs = specs - specs_bkg
   
    wavelength_all, specs = glue_steps(wavelength, specs, number_pixel = 1002, grade = grade, plot_all_step = False)
    
    specs_substrate_stokes, wavelength_stokes = select_signal(signal_substrate, wavelength_all, 545, wavelength_all[-1])
    specs_substrate_antistokes, wavelength_antistokes = select_signal(signal_substrate, wavelength_all, wavelength_all[0], 521)
    
    specs_stokes, wavelength_stokes = select_signal(specs, wavelength_all, 545, wavelength_all[-1])
    specs_antistokes, wavelength_antistokes = select_signal(specs, wavelength_all, wavelength_all[0], 521)

    specs_stokes = smooth_Signal(specs_stokes, window, deg, repetitions)
    specs_antistokes = smooth_Signal(specs_antistokes, window, deg, repetitions)
        
    specs_substrate_stokes = smooth_Signal(specs_substrate_stokes, window, deg, repetitions)
    specs_substrate_antistokes = smooth_Signal(specs_substrate_antistokes, window, deg, repetitions)
    
    specs_notch, wave_notch = select_signal(specs, wavelength_all, 525, 530)
    wavelength_notch = np.linspace(522, 543, 10)
    specs_notch = np.ones(len(wavelength_notch))*np.mean(specs_notch)
    
    wavelength = np.hstack((wavelength_antistokes, wavelength_notch, wavelength_stokes))
    specs = np.hstack((specs_antistokes, specs_notch, specs_stokes))
    specs_substrate = np.hstack((specs_substrate_antistokes, specs_notch, specs_substrate_stokes))

    spectrum = specs
    spectrum_substrate = specs_substrate
    
 #   if fit_raman_water:
  #      print('Fit raman water to NP+substrate')
        
    #    I = 2500 #20000#6000
     #   init_londa = 550
      #  init_width = 50
       # I2 = 100#2500#1000
        #I3 = 100#100#0#0
        #C = 400
        #init_parameters = np.array([I, init_width, init_londa, I2, I3, C], dtype=np.double)
        
        #bounds = ([0, 0, 500, 0, 0, 0], [16500, 300, 1000, 16500, 16500, 700])  
    
   #     ends_notch = 550
    #    last_wave = 650
     #   wavelength_fitted, lorentz_fitted, best_parameters, r2 = fit_signal_raman_test(wavelength, spectrum, ends_notch, last_wave, init_parameters, bounds)
        
      #  lorentz_fitted = three_lorentz(wavelength_stokes, *best_parameters)
       # all_r2 = calc_r2(specs_stokes, lorentz_fitted)
        
        #print('Fit Lorentz NP:', NP, 'r2:', r2, 'all_r2:', all_r2)
        
            
    if fit_raman_water:
        print('Fit raman water to NP+substrate')
        
        I = 2500 #20000#6000
        init_londa = 550
        init_width = 50
        
        Imode2 = 2500
        init_londa2 = 600        
        init_width2 = 50

        I2 = 100#2500#1000
        I3 = 100#100#0#0
        C = 400
        init_parameters = np.array([I, init_width, init_londa, Imode2, init_width2, init_londa2, I2, I3, C], dtype=np.double)
        
        bounds = ([0, 0, 530, 0, 0, 530, 0, 0, 0], [16500, 100, 600, 16500, 100, 730, 16500, 16500, 700])  
        
        ends_notch = 550
        last_wave = 650

        wavelength_fitted, lorentz_fitted, best_parameters, r2 = fit_signal_raman_test_two_modes(wavelength, spectrum, ends_notch, last_wave, init_parameters, bounds)
        
        lorentz_fitted = four_lorentz(wavelength_stokes, *best_parameters)
        all_r2 = calc_r2(specs_stokes, lorentz_fitted)
        
        print('Fit Lorentz NP:', NP, 'r2:', r2, 'all_r2:', all_r2, *best_parameters)
        
        
    print('Ploteo:', NP)
    
    spectrum_NP = spectrum-spectrum_substrate
   # spectrum_NP = spectrum
    
    header_txt = 'Wavelength (nm), Spectrum difference, Spectrum NP + substrate, Spectrum substrate'
    name_data = os.path.join(save_folder_2,'Luminescence_Steps_Spectrum_%s.txt'%(NP))
    data = np.array([wavelength, spectrum_NP, spectrum, spectrum_substrate]).T
    np.savetxt(name_data, data, header = header_txt)
    
    if fit_raman_water:
        header_txt = 'I, londa (nm), width (nm), I_mode2, londa_mode2 (nm), width_mode2 (nm), I_649, I_702, C, r2'
        name_data = os.path.join(save_folder_3,'Parameters_Fit_raman_water_%s.txt'%(NP))
        data = np.array([best_parameters[0],best_parameters[1],best_parameters[2],best_parameters[3],best_parameters[4],best_parameters[5], best_parameters[6], best_parameters[7], best_parameters[8], r2]).T
        np.savetxt(name_data, data, header = header_txt)
    
    plt.figure()      
    plt.title('%1s'%(NP))
    plt.plot(wavelength, spectrum, label = 'signal on NP+substrate')
    plt.plot(wavelength, spectrum_substrate, '--k', label = 'signal on substrate')
    plt.plot(wavelength,  spectrum_NP, '--r', label = 'difference')
    
    if fit_raman_water:
        plt.plot(wavelength_stokes, lorentz_fitted, '--g', label = 'fit with raman water, r2: %s, allr2: %s'%(r2,all_r2))
        plt.axvspan(ends_notch, last_wave, facecolor='green', alpha=0.2)
    
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(521, 543, facecolor='grey', alpha=0.2)
    plt.ylim(fig_lim)
    plt.legend(loc='upper right') 
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Counts')
    
    figure_name = os.path.join(save_folder, 'Luminescence_Steps_Spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
        
    return

def process_spectrum_plot_all(name_col, path_to, save_folder, fig_lim):
    
    folder = os.path.join(path_to, name_col)
    
    print('Plot all NP from:', name_col)
    
    folder = os.path.join(folder, 'luminescence_steps')
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('Luminescence_Steps_Spectrum',f)]
    list_of_folders.sort()
    
    L = len(list_of_folders)
        
    color_map = plt.cm.coolwarm(np.linspace(0,1,L))
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    
    plt.figure()
    
    plt.title('%s'%name_col)
    
    for f in list_of_folders:
        
        name_NP = f.split('_NP_')[-1]
        name_NP =   name_NP.split('.')[0]
        name_NP = 'NP_%s'%name_NP
        
        file = os.path.join(folder, f)
        data = np.loadtxt(file, skiprows=1)

        wavelength = data[:, 0]
        spectrum_NP = data[:, 1]
        
        plt.plot(wavelength, spectrum_NP, label = '%s'%name_NP)
        
    plt.ylim(fig_lim)
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(521, 543, facecolor='grey', alpha=0.2)

    plt.legend(loc='upper right', fontsize = 'x-small') 
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Counts')
    
    figure_name = os.path.join(save_folder, 'all_Luminescence_Steps_Spectrum_%s.png'%name_col) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    return

def process_select_spectrum(name_col, path_to, save_folder, first_wave_select, end_wave_select):
    
    folder = os.path.join(path_to, name_col)
    
    folder = os.path.join(folder, 'luminescence_steps')
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('Luminescence_Steps_Spectrum',f)]
    list_of_folders.sort()
    
    integrate_intensity = []
    max_wave = []
    frame = []
    
    print('Plot max intenstiy and max wavelength from:', name_col)
    
    for f in list_of_folders:
        
        name_NP = f.split('_NP_')[-1]
        name_NP =   name_NP.split('.')[0]
        
        file = os.path.join(folder, f)
        data = np.loadtxt(file, skiprows=1)

        wavelength = data[:, 0]
        spectrum_NP = data[:, 1]
        
        select_spectrum, select_wavelength = select_signal(spectrum_NP, wavelength, first_wave_select, end_wave_select)
        
        max_wave.append(select_wavelength[np.argmax(select_spectrum)])
        integrate_intensity.append(np.sum(select_spectrum))
        frame.append(int(name_NP))
        
   # frame = np.array([-20,-16,-15,-12,-10,-8,-4,-3,-2,-1,0,1,2,3,4,8,10,12,15,16,20])
   # frame = 0.1*(np.array(frame) - 11)
   # frame = 0.05*(np.array(frame) - 5)
   # frame = (np.array(frame) - 13)
  #  frame = 0.02*(np.array(frame) - 9)
   # frame = np.arange(18, -22, -2)
    
    #Max intensity
        
    plt.figure()
    
    plt.title('Select ROI between: %s nm and %s nm'%(first_wave_select, end_wave_select))
    
    plt.plot(frame, integrate_intensity, 'ko-')
        
    plt.xlabel('NP')
   # plt.xlabel('Z (um)')
    plt.ylabel('Integrate Intensity (Counts)')
    
    figure_name = os.path.join(save_folder, 'integrate_intensity_%s-%s_nm_%s.png'%(first_wave_select, end_wave_select, name_col)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    # Max wavelength
    
    plt.figure()
    
    plt.title('Select ROI between: %s nm and %s nm'%(first_wave_select, end_wave_select))
    
    plt.plot(frame, max_wave, 'ro-')
        
    plt.xlabel('NP')
   # plt.xlabel('Z (um)')
    plt.ylabel('Max Wavelength (nm)')
    
    figure_name = os.path.join(save_folder, 'max_wavelength_%s-%s_nm_%s.png'%(first_wave_select, end_wave_select, name_col)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    header_txt = 'NP, Integrate intensity, Max Wavelength (nm)'
    name_data = os.path.join(save_folder,'data_select_ROI_%s-%s_nm_%s.txt'%(first_wave_select, end_wave_select, name_col)) 
    data = np.array([frame, integrate_intensity, max_wave]).T
    np.savetxt(name_data, data, header = header_txt)
    
    
def process_spectrum_plot_all_fit(name_col, path_to, save_folder, fig_lim, first_wave_select, end_wave_select, fig_lim_2):
    
    wavelength = np.linspace(first_wave_select, end_wave_select, 1000)
    
    list_lspr_max = []
    list_lspr_mode1 = []
    list_lspr_mode2 = []
    list_r2  = []
    frame = []
    
    folder = os.path.join(path_to, name_col)
    
    print('Plot fit all NP from:', name_col)
    
    folder = os.path.join(folder, 'parameters_fit_raman_water')
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('Parameters_Fit_raman_water',f)]
    list_of_folders.sort()
    
    L = len(list_of_folders)
        
    color_map = plt.cm.coolwarm(np.linspace(0,1,L))
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    
    plt.figure()
    
    plt.title('%s'%name_col)
    
    for f in list_of_folders:
        
        name_NP = f.split('_NP_')[-1]
        name_NP =   name_NP.split('.')[0]
        frame.append(int(name_NP))
        
        file = os.path.join(folder, f)
        data = np.loadtxt(file, skiprows=1)
        
        Imode1,  width_mode1, londa_mode1, Imode2, width_mode2, londa_mode2, I649, I702, C, r2 = data
        parameters = Imode1,  width_mode1, londa_mode1, Imode2, width_mode2, londa_mode2, I649, I702, C
        
        londa_mode1 = min([londa_mode1, londa_mode2])
        londa_mode2 = max([londa_mode1, londa_mode2])

        spectrum_NP = four_lorentz(wavelength, *parameters)
        wave_max = wavelength[np.argmax(spectrum_NP)]
        
        list_lspr_mode1.append(londa_mode1)
        list_lspr_mode2.append(londa_mode2)
        list_lspr_max.append(wave_max)
        list_r2.append(r2)
        
        name_NP = 'NP_%s'%name_NP
        plt.plot(wavelength, spectrum_NP, label = '%s'%name_NP)
        
    plt.ylim(fig_lim)
    plt.legend(loc='upper right', fontsize = 'x-small') 
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Counts')
    
    figure_name = os.path.join(save_folder, 'all_Fit_Luminescence_%s.png'%name_col) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    # wavelength LSPR
    
    #plt.figure()
    
    fig, ax = plt.subplots()
    plt.title('From fit raman water: Modes LSPR')
    
    ax.plot(frame, list_lspr_mode1, 'bo-', label = 'fit mode 1')
    ax.plot(frame, list_lspr_mode2, 'ro-', label = 'fit mode 2')
    ax.plot(frame, list_lspr_max, 'go-', label = 'lspr max')
       
    ax.set_xlabel('NP')
    ax.set_ylim(fig_lim_2)
    ax.set_ylabel('Wavelength LSPR (nm)')
    ax.tick_params(axis='y', which='major')
    
    ax2 = ax.twinx()    
    ax2.plot(frame, list_r2, 'ko--', label = 'r2')
    ax2.set_ylim(0.8, 1.1)
    ax2.set_ylabel('r2')
    
    ax2.tick_params(axis='y', which='major', labelcolor = 'k')
    
    plt.legend()
    
    figure_name = os.path.join(save_folder, 'fit_wavelength_lspr_%s.png'% name_col)
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    header_txt = 'NP, Wavelength LSPR mode 1 (nm), Wavelength LSPR mode 2 (nm), Wavelength LSPR max (nm), r2'
    name_data = os.path.join(save_folder,'fit_wavelength_lspr_%s.txt'%name_col)
    data = np.array([frame, list_lspr_mode1, list_lspr_mode2, list_lspr_max, list_r2]).T
    np.savetxt(name_data, data, header = header_txt)
    
    return
    
        
if __name__ == '__main__':

    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
    
 #   daily_folder = '2021-08 (Growth PL circular)/2021-08-04 (pre growth, PL circular)/laser_polarizacion_circular_MAL/20210804-175940_Luminescence_Steps_10x12'   
 #   daily_substrate = '2021-08-04 (pre growth, PL circular)/laser_polarizacion_circular_MAL/Spectrum_luminescencesustrate'

 #   daily_folder = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/pre_growth/20210809-150932_Luminescence_Steps_10x12'
 #   daily_substrate = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/pre_growth/Spectrum_luminescencesustrate'
 
    daily_folder = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)/PL/20210818-174530_Luminescence_Steps_10x12'
    daily_substrate = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)/PL/Spectrum_luminescencesustrate'
    
    parent_folder = os.path.join(base_folder, daily_folder)
    
    folder_substrate = os.path.join(base_folder, daily_substrate)
    list_of_files = os.listdir(folder_substrate)
    
    file1 = [f for f in list_of_files if re.search('Calibration',f)][0]
    
    calibration_wavelength = os.path.join(folder_substrate, file1)
    wavelength = np.loadtxt(calibration_wavelength) + 2 #esto es pq tenia mal puesto el gratting offset
    
    sustrate_bkg_bool = False
    
    file2 = [f for f in list_of_files if re.search('Line_Spectrum',f) and not re.search('Background',f)][0]
    line_spectrum = os.path.join(folder_substrate, file2)
    specs = np.loadtxt(line_spectrum)
    
    if sustrate_bkg_bool:
    
        file3 = [f for f in list_of_files if re.search('Background',f)][0]
        line_spectrum_bkg = os.path.join(folder_substrate, file3)
        specs_bkg = np.loadtxt(line_spectrum_bkg)
        
    else:
    
        specs_bkg = np.zeros(len(specs))
    
    substrate = specs - specs_bkg
    
    window, deg, repetitions = 21, 1, 1
    plot_all_col = True  #plotea por cada columna todas sus NPs.
    grade = 2  #correct_step_and_glue grado de weights del glue
    fig_lim = -10, 4000
    
    fit_raman_water = True #fit 3 lorentz to signal NP + substrate
    
    wavelength_substrate, signal_substrate = glue_steps(wavelength, substrate, number_pixel = 1002, grade = grade, plot_all_step = False)

#Aca comienza la rutina

    list_of_folders = os.listdir(parent_folder)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
    list_of_folders = [f for f in list_of_folders if re.search('Spectrum_luminescence',f)]
    list_of_folders.sort()
    
    L_folder = len(list_of_folders)
    
    path_to = os.path.join(parent_folder,'processed_data_luminiscence_sustrate_bkg_%s'%sustrate_bkg_bool)
    
    if not os.path.exists(path_to):
        os.makedirs(path_to)
  
    for f in list_of_folders:
        
        folder = os.path.join(parent_folder,f)
        process_spectrum(folder, path_to, sustrate_bkg_bool, window, deg, repetitions, wavelength, signal_substrate, grade, fig_lim, fit_raman_water)
        
    if plot_all_col:
        
        list_of_folders = os.listdir(path_to)
        list_of_folders = [f for f in list_of_folders if re.search('Col',f)]
        list_of_folders.sort()
        
        save_folder = os.path.join(path_to,'fig_all_data') 
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        fig_lim =  -10, 4000
    
        for f in list_of_folders:
            process_spectrum_plot_all(f, path_to, save_folder, fig_lim = fig_lim)
        plt.style.use('default')
        
        # Para max_wave and max_intensity on process_select_spectrum
        first_wave_select = 546 #550 gold  #sustrate 700    #water  625 #antistokes 500
        end_wave_select = 600 #570 gold  #sustrate 725   #water 670  #antistokes 521
        for f in list_of_folders:
            process_select_spectrum(f, path_to, save_folder, first_wave_select, end_wave_select)
            
        if fit_raman_water:
            
            save_folder_fit = os.path.join(path_to,'fig_fit_all_data')
            if not os.path.exists(save_folder_fit):
                os.makedirs(save_folder_fit)
                
            first_wave_select = 540
            end_wave_select = 650
            
            fig_lim_2 = 520, 640
            
            for f in list_of_folders:
                process_spectrum_plot_all_fit(f, path_to, save_folder_fit, fig_lim, first_wave_select, end_wave_select, fig_lim_2)
            plt.style.use('default')
            
            
            
            