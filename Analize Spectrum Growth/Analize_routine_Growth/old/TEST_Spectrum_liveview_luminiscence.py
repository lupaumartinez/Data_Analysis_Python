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

from scipy.optimize import curve_fit
import scipy.signal as sig

from Fit_raman_water import fit_signal_raman, fit_signal_raman_test

hc = 1239.84193 # Plank's constant times speed of light in eV*nm
k = 0.000086173 # Boltzmann's constant in eV/K

def smooth_Signal(signal, window, deg, repetitions):
    
    k = 0
    while k < repetitions:
        signal = sig.savgol_filter(signal, window, deg, mode = 'mirror')
        k = k + 1
        
    return signal


def process_spectrum(folder, common_path, first_wave, starts_notch, ends_notch, last_wave, window, deg, repetitions, spectrum_time, sustrate_bkg_bool):
    
    NP = folder.split('Spectrum_')[-1]

   # save_folder = os.path.join(common_path,'%s'%NP)
#
   # if not os.path.exists(save_folder):
   #     os.makedirs(save_folder)
    
    name_spectrum = []
    specs = []
    
    name_spectrum_bkg = []
    specs_bkg = []
    
    list_of_files = os.listdir(folder)
    list_of_files.sort()
        
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
        
    specs = [specs for _,specs in sorted(zip(name_spectrum,specs))]            
    name_spectrum = sorted(name_spectrum)
    
    specs_bkg = [specs_bkg for _,specs_bkg in sorted(zip(name_spectrum_bkg,specs_bkg))]      
        
    L = len(specs)
            
    print('Number:', NP, 'Spectra acquired:', L)
    
    initial = 0
    final = L
    
    total_frame = len(range(initial, final))
    
    matrix_spectrum = np.zeros((total_frame, len(wavelength)))
    matrix_spectrum_smooth = np.zeros((total_frame, len(wavelength)))
        
    time = []
    max_spectrum = []
    
    for i in range(initial, final):
        
        spectrum = specs[i]
        spectrum_smooth = smooth_Signal(spectrum, window, deg, repetitions)
        
     #   if sustrate_bkg_bool:
            
     #       spectrum = spectrum - ROI_size*specs_bkg[i]
            
     #   else:
            
     #       spectrum = spectrum
     
      #  spectrum = (spectrum-min(spectrum))/(max(spectrum)-min(spectrum))
      
        index_max = np.argmax(spectrum_smooth)
        
        time.append(spectrum_time*i)
        max_spectrum.append(np.round(wavelength[index_max],3))
        
        matrix_spectrum[i, :] = spectrum
        matrix_spectrum_smooth[i, :] = spectrum_smooth
    
    return common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum

def process_spectrum_prom_delta(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum):
    
    n_prom = 3
    
    spectrum_prom_initial = (matrix_spectrum_smooth[0, :] + matrix_spectrum_smooth[1, :] + matrix_spectrum_smooth[2, :]) / n_prom
    
    spectrum_prom_initial = (spectrum_prom_initial - min(spectrum_prom_initial))/(max(spectrum_prom_initial) - min(spectrum_prom_initial))
    
    time_prom = []
    max_spectrum_prom = []
    
    total_frame = len(range(initial, final))
    color_map = plt.cm.coolwarm(np.linspace(0,1,total_frame))
    plt.figure()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    plt.title('%1s'%(NP))
    
    for i in range(initial+2, final, 3):
        
        spectrum_prom = (matrix_spectrum_smooth[i-2, :] + matrix_spectrum_smooth[i-1, :] + matrix_spectrum_smooth[i, :]) / n_prom
        
        spectrum_prom = (spectrum_prom - min(spectrum_prom))/(max(spectrum_prom) - min(spectrum_prom))
        
        spectrum_prom = spectrum_prom -  spectrum_prom_initial
        
        index_max = np.argmax(spectrum_prom)
        
        time_prom.append(spectrum_time*i)
        max_spectrum_prom.append(np.round(wavelength[index_max],3))
            
        plt.plot(wavelength, spectrum_prom)
        plt.plot(wavelength[index_max], spectrum_prom[index_max], 'ko')
        
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity Normalized')
        
    figure_name = os.path.join(common_path, 'Liveview_delta_spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
        
    plt.style.use('default')
    
    plt.figure()
    plt.plot(time, max_spectrum, 'o')
    plt.plot(time_prom, max_spectrum_prom, 'k*', label = 'promedio de 3')
    plt.xlabel('Time (s)')
    plt.ylabel('max [PL] (nm)')
    plt.legend()
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'Max_delta_PL_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()

    return

def process_spectrum_prom(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum):
    
    n_prom = 3

    time_prom = []
    max_spectrum_prom = []
    max_spectrum_prom_original = []
    
    total_frame = len(range(initial, final))
    color_map = plt.cm.coolwarm(np.linspace(0,1,total_frame))
    plt.figure()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    plt.title('%1s'%(NP))
    
    for i in range(initial+2, final, 3):
        
        spectrum_prom_original = (matrix_spectrum[i-2, :] + matrix_spectrum[i-1, :] + matrix_spectrum[i, :]) / n_prom
        spectrum_prom_original = smooth_Signal(spectrum_prom_original, 51, deg, repetitions)
        
        spectrum_prom = (matrix_spectrum_smooth[i-2, :] + matrix_spectrum_smooth[i-1, :] + matrix_spectrum_smooth[i, :]) / n_prom
        
        time_prom.append(spectrum_time*i)
        
        index_max = np.argmax(spectrum_prom)
        max_spectrum_prom.append(np.round(wavelength[index_max],3))
        
        index_max_original = np.argmax(spectrum_prom_original)
        max_spectrum_prom_original.append(np.round(wavelength[index_max_original],3))        
        
        plt.plot(wavelength, spectrum_prom)
        plt.plot(wavelength, spectrum_prom_original)
        
        plt.plot(wavelength[index_max], spectrum_prom[index_max], 'ko')
        plt.plot(wavelength[index_max_original], spectrum_prom_original[index_max_original], 'mo')
        
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity Normalized')
        
    figure_name = os.path.join(common_path, 'Liveview_spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
        
    plt.style.use('default')
    
    plt.figure()
    plt.plot(time, max_spectrum, 'o')
    plt.plot(time_prom, max_spectrum_prom, 'k*', label = 'promedio de 3 smooth 21')
    plt.plot(time_prom, max_spectrum_prom_original, 'm*', label = 'smooth 51 de promedio de 3')
    plt.xlabel('Time (s)')
    plt.ylabel('max [PL] (nm)')
    plt.legend()
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'Max_PL_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    
    return

def process_spectrum_prom_fit(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum):
    
    n_prom = 3

    time_prom = []
    max_spectrum_prom = []
    
    total_frame = len(range(initial, final))
    color_map = plt.cm.coolwarm(np.linspace(0,1,total_frame))
    plt.figure()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    plt.title('%1s'%(NP))
    
    londa_max_pl = []
    londa_max_pl_original = []
    
    #preparo ajuste
    I = 2500#20000#6000
    init_londa = 550
    init_width = 50
    I2 = 100#2500#1000
    I3 = 100#0#0
    C = 450
    
    init_parameters_NP = np.array([I, init_width, init_londa, I2, I3, C], dtype=np.double)
    
    for i in range(initial+2, final, 3):
        
        spectrum_prom = (matrix_spectrum_smooth[i-2, :] + matrix_spectrum_smooth[i-1, :] + matrix_spectrum_smooth[i, :]) / n_prom

        spectrum_prom_original = (matrix_spectrum[i-2, :] + matrix_spectrum[i-1, :] + matrix_spectrum[i, :]) / n_prom

        wavelength_fitted, lorentz_fitted, best_lorentz = fit_signal_raman_test(wavelength, spectrum_prom, ends_notch, last_wave, init_parameters_NP)
        
        wavelength_fitted_o, lorentz_fitted_o, best_lorentz_o = fit_signal_raman_test(wavelength, spectrum_prom_original, ends_notch, last_wave, init_parameters_NP)
        
    #    full_lorentz_fitted = three_lorentz(wavelength_fitted, *best_lorentz)
    #    fitted =  three_lorentz(wavelength_stokes, *best_lorentz)
    #    r2_lorentz[i] = calc_r2(spectrum_stokes, fitted)

        #I, gamma, x0, I_2, I_3, C = p
        
        time_prom.append(spectrum_time*i)
        
        londa_max_pl.append(np.round( best_lorentz[2], 3))
        londa_max_pl_original.append(np.round( best_lorentz_o[2], 3))
        
        index_max = np.argmax(spectrum_prom)
        max_spectrum_prom.append(np.round(wavelength[index_max],3))
            
        plt.plot(wavelength, spectrum_prom)
        plt.plot(wavelength_fitted, lorentz_fitted, 'r--')
        plt.plot(wavelength_fitted_o, lorentz_fitted_o, 'm--')
        
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity Normalized')
        
    figure_name = os.path.join(common_path, 'Fit_Liveview_spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
        
    plt.style.use('default')
    
    plt.figure()
    plt.plot(time, max_spectrum, 'o')
    plt.plot(time_prom, max_spectrum_prom, 'k*', label = 'promedio de 3 smooth')
    plt.plot(time_prom, londa_max_pl, 'r*', label = 'fit promedio de 3 smooth ')
    plt.plot(time_prom, londa_max_pl_original, 'm*', label = 'fit promedio de 3 original')
    
    plt.xlabel('Time (s)')
    plt.ylabel('max [PL] (nm)')
    plt.legend()
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'Fit_PL_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    
    return

def process_spectrum_fit(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum):
    
    total_frame = len(range(initial, final))
    color_map = plt.cm.coolwarm(np.linspace(0,1,total_frame))
    plt.figure()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    plt.title('%1s'%(NP))
    
    time_londa = []
    londa_max_pl = []
    londa_max_pl_original = []
    
    #preparo ajuste
    I = 2500#20000#6000
    init_londa = 550
    init_width = 50
    I2 = 100#2500#1000
    I3 = 100#0#0
    C = 450
    
    init_parameters_NP = np.array([I, init_width, init_londa, I2, I3, C], dtype=np.double)
    
    for i in range(final-20, final):
        
        spectrum_prom = matrix_spectrum_smooth[i, :]

        spectrum_prom_original = matrix_spectrum[i, :]

        wavelength_fitted, lorentz_fitted, best_lorentz = fit_signal_raman_test(wavelength, spectrum_prom, ends_notch, last_wave, init_parameters_NP)
        
        wavelength_fitted_o, lorentz_fitted_o, best_lorentz_o = fit_signal_raman_test(wavelength, spectrum_prom_original, ends_notch, last_wave, init_parameters_NP)
        
    #    full_lorentz_fitted = three_lorentz(wavelength_fitted, *best_lorentz)
    #    fitted =  three_lorentz(wavelength_stokes, *best_lorentz)
    #    r2_lorentz[i] = calc_r2(spectrum_stokes, fitted)

        #I, gamma, x0, I_2, I_3, C = p
        
        time_londa.append(spectrum_time*i)
        
        londa_max_pl.append(np.round( best_lorentz[2], 3))
        londa_max_pl_original.append(np.round( best_lorentz_o[2], 3))
            
        plt.plot(wavelength, spectrum_prom)
        plt.plot(wavelength_fitted, lorentz_fitted, 'r--')
        plt.plot(wavelength_fitted_o, lorentz_fitted_o, 'm--')
        
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity Normalized')
        
    figure_name = os.path.join(common_path, 'one_Fit_Liveview_spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
        
    plt.style.use('default')
    
    plt.figure()
    plt.plot(time, max_spectrum, 'o')
    plt.plot(time_londa, londa_max_pl, 'r*', label = 'fit de smooth ')
    plt.plot(time_londa, londa_max_pl_original, 'm*', label = 'fit de original')
    
    plt.xlabel('Time (s)')
    plt.ylabel('max [PL] (nm)')
    plt.legend()
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'one_Fit_PL_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    
    return

def three_lorentz(x, *p):

    pi = np.pi

    I, gamma, x0, I_2, I_3, C = p
    
    a = (1/pi) * I_2 * (15.5/2)**2 / ((x - 649)**2 + (15.2/2)**2) 
    b = (1/pi) * I_3 * (183/2)**2 / ((x - 702)**2 + (183/2)**2) 
    
    return  (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + (1/pi) * I_2 * a + (1/pi) * I_3 * b + C

def process_spectrum_polynomial(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum):
    
    total_frame = len(range(initial, final))
    color_map = plt.cm.coolwarm(np.linspace(0,1,total_frame))
    plt.figure()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    plt.title('%1s'%(NP))
    
    time_londa = []
    max_pl_diff = []
    max_pl_original = []
    max_pl_poly3 = []
    
    first_wave = 545
    end_wave = 640
    
    desired_range = np.where((wavelength >= first_wave) & (wavelength <=end_wave))    
    desired_wave = wavelength[desired_range]
    
    x = np.linspace(desired_wave[0], desired_wave[-1], 1000)
    npol = 5
    
    npol3 = 3
    
    for i in range(initial, final):
        
        spectrum_original = matrix_spectrum[i, :]
        desired_spectrum_original = spectrum_original[desired_range]
    
        p3 = np.polyfit(desired_wave, desired_spectrum_original, npol3)
        poly3 = np.polyval(p3, x)
        max_wave_poly3 = round(x[np.argmax(poly3)],3)
        max_pl_poly3.append(max_wave_poly3)
  
        p_o = np.polyfit(desired_wave, desired_spectrum_original, npol)
        poly_o = np.polyval(p_o, x)
        max_wave_poly_o = round(x[np.argmax(poly_o)],3)
        
        diff = np.diff(poly_o, 1, prepend = poly_o[0])
        diff2 = np.diff(poly_o, 2, prepend = poly_o[0])
        
        a = np.where(diff2 <= 0)[0]
        maximus = diff[a]
        
        out = np.argmin(np.abs(maximus))
        max_wave_poly_o_diff = round(x[out],3)
        
       # print(max_wave_poly_o_diff, max_wave_poly_o )
        
        time_londa.append(spectrum_time*i)
        
        max_pl_diff.append(max_wave_poly_o_diff )
        max_pl_original.append(max_wave_poly_o)
            
       # plt.plot(x, desired_spectrum, 'b')
       # plt.plot(x, poly, 'r--')
        
       # plt.plot(desired_wavelength, desired_spectrum_original)
       
     #   plt.plot(x[1:], diff, 'y--')
        plt.plot(x[out], poly_o[out], 'yo')
        plt.plot(x[np.argmax(poly_o)], poly_o[np.argmax(poly_o)], 'bo')
        plt.plot(x, poly_o, 'k--')
        
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity Normalized')
        
    figure_name = os.path.join(common_path, 'one_Poly_Liveview_spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
        
    plt.style.use('default')
    
    plt.figure()
    plt.plot(time, max_spectrum, 'o', label = 'max de smooth')
    plt.plot(time_londa, max_pl_diff, 'y*', label = 'max de poly5 con diff 0')
    plt.plot(time_londa, max_pl_original, 'k*', label = 'max de poly5 de original')
    plt.plot(time_londa, max_pl_poly3, 'm*', label = 'max de poly3 de original')
    
    plt.xlabel('Time (s)')
    plt.ylabel('max [PL] (nm)')
    plt.legend()
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'one_Poly_PL_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    
    return

def process_spectrum_prom_polynomial(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum):
    
    total_frame = len(range(initial, final))
    color_map = plt.cm.coolwarm(np.linspace(0,1,total_frame))
    plt.figure()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    plt.title('%1s'%(NP))
    
    time_londa = []
    max_pl = []
    max_pl_original = []
    
    first_wave = 545
    end_wave = 640
    
    desired_range = np.where((wavelength >= first_wave) & (wavelength <=end_wave))    
    desired_wave = wavelength[desired_range]
    
    x = np.linspace(desired_wave[0], desired_wave[-1], 1000)
    npol = 5
    
    nprom = 10
    
    for i in range(final-50, final, nprom):
        
        spectrum = matrix_spectrum[i, :]
     
        spectrum_original_prom = (matrix_spectrum[i, :] + matrix_spectrum[i-1, :] +  matrix_spectrum[i-2, :] 
        
        +  matrix_spectrum[i-3, :] +  matrix_spectrum[i-4, :] +  matrix_spectrum[i-5, :] 
        
        +  matrix_spectrum[i-6, :] +  matrix_spectrum[i-7, :] +  matrix_spectrum[i-8, :] +  matrix_spectrum[i-9, :])/nprom
        
        desired_spectrum = spectrum[desired_range]
        desired_spectrum_original = spectrum_original_prom[desired_range]
    
        p = np.polyfit(desired_wave, desired_spectrum, npol)
        poly = np.polyval(p, x)
        max_wave_poly = round(x[np.argmax(poly)],3)
  
        p_o = np.polyfit(desired_wave, desired_spectrum_original, npol)
        poly_o = np.polyval(p_o, x)
        max_wave_poly_o = round(x[np.argmax(poly_o)],3)
        
        time_londa.append(spectrum_time*i)
        
        max_pl.append(max_wave_poly)
        max_pl_original.append(max_wave_poly_o)
            
        plt.plot(wavelength, spectrum, 'r')
       # plt.plot(x, poly, 'r--')
        
        plt.plot(wavelength, spectrum_original_prom, 'k')
     #   plt.plot(x, poly_o, 'k--')
        
        
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity Normalized')
        
    figure_name = os.path.join(common_path, 'prom_Liveview_spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
        
    plt.style.use('default')
    
    plt.figure()
    plt.plot(time, max_spectrum, 'o', label = 'max de smooth')
    plt.plot(time_londa, max_pl, 'r*', label = 'max de poly')
    plt.plot(time_londa, max_pl_original, 'k*', label = 'max de poly de 10prom original')
    
    plt.xlabel('Time (s)')
    plt.ylabel('max [PL] (nm)')
    plt.legend()
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'prom_Poly_PL_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    return

def process_spectrum_prom6_polynomial(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum):
    
    total_frame = len(range(initial, final))
    color_map = plt.cm.coolwarm(np.linspace(0,1,total_frame))
    plt.figure()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    plt.title('%1s'%(NP))
    
    time_londa = []
    max_pl = []
    max_pl_original = []
    
    first_wave = 545
    end_wave = 640
    
    desired_range = np.where((wavelength >= first_wave) & (wavelength <=end_wave))    
    desired_wave = wavelength[desired_range]
    
    x = np.linspace(desired_wave[0], desired_wave[-1], 1000)
    npol = 5
    
    nprom = 6
    
    for i in range(initial-nprom-1, final, nprom):
        
        spectrum = matrix_spectrum[i, :]
     
        spectrum_original_prom = (matrix_spectrum[i, :] + matrix_spectrum[i-1, :] +  matrix_spectrum[i-2, :] 
        
        +  matrix_spectrum[i-3, :] +  matrix_spectrum[i-4, :] + matrix_spectrum[i-5, :])/nprom
        
        desired_spectrum = spectrum[desired_range]
        desired_spectrum_original = spectrum_original_prom[desired_range]
    
        p = np.polyfit(desired_wave, desired_spectrum, npol)
        poly = np.polyval(p, x)
        max_wave_poly = round(x[np.argmax(poly)],3)
  
        p_o = np.polyfit(desired_wave, desired_spectrum_original, npol)
        poly_o = np.polyval(p_o, x)
        max_wave_poly_o = round(x[np.argmax(poly_o)],3)
        
        time_londa.append(spectrum_time*i)
        
        max_pl.append(max_wave_poly)
        max_pl_original.append(max_wave_poly_o)
            
       # plt.plot(wavelength, spectrum, 'r')
        plt.plot(x, poly, 'r--')
        
      #  plt.plot(wavelength, spectrum_original_prom, 'k')
        plt.plot(x, poly_o, 'k--')
        
        
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity Normalized')
        
    figure_name = os.path.join(common_path, 'prom6_Poly_Liveview_spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
        
    plt.style.use('default')
    
    plt.figure()
    plt.plot(time, max_spectrum, 'o', label = 'max de smooth')
    plt.plot(time_londa, max_pl, 'r*', label = 'max de poly')
    plt.plot(time_londa, max_pl_original, 'k*', label = 'max de poly de 6prom original')
    
    plt.xlabel('Time (s)')
    plt.ylabel('max [PL] (nm)')
    plt.legend()
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'prom6_Poly_PL_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    return

def cut_range(wavelength, intensity, initial_wave, end_wave):
    
    desired_range = np.where((wavelength >= initial_wave) & (wavelength <=end_wave))    
    wavelength = wavelength[desired_range]
    intensity = intensity[desired_range]
    
    return wavelength, intensity

def process_prom(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum):
    
    first_wave = 550
    end_wave = 590
    
    desired_range = np.where((wavelength >= first_wave) & (wavelength <=end_wave))    
    desired_wave = wavelength[desired_range]
    
    n= 10 #grupos de a n
    
    final = initial + n*5
    
    
    for k in range(initial, final, n):
        
        total_frame = len(range(initial, final))
        color_map = plt.cm.coolwarm(np.linspace(0,1,total_frame))
        plt.figure()
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
        plt.title('%1s'%(NP))
        
        nprom = []
        std = []
        integrate = []
    
        j = 2
        spectrum_o = matrix_spectrum[k, :]
        spectrum_o = spectrum_o[desired_range]
    
        for i in range(k, k + n):
            
            spectrum = matrix_spectrum[i, :]
            spectrum = spectrum[desired_range]
         
            spectrum_sum = spectrum_o + spectrum
            
            spectrum_prom = spectrum_sum/j
        
            nprom.append(j)
            std.append(round(np.std(spectrum_prom),3))
            integrate.append(round(np.sum(spectrum_prom),3))
            
            j = j + 1
            spectrum_o = spectrum_sum
        
        #    desired_spectrum = spectrum_prom[desired_range]
        
            plt.plot(desired_wave, spectrum_prom)
            
        plt.axvline(532, color = 'g', linestyle = '--')
        plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity Normalized')
            
        figure_name = os.path.join(common_path, 'prom_Liveview_spectrum_%s_%s.png' % (NP, k)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        plt.style.use('default')
        
        plt.figure()
        plt.plot(nprom, np.array(std)/np.array(integrate), 'o')
        
        plt.xlabel('N promedio')
        plt.ylabel('STD/SUM')
      #  plt.legend()
      #  plt.xlim(-2, 242)
        figure_name = os.path.join(common_path, 'STD_prom_%s_%s.png' % (NP, k)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
    
    return


if __name__ == '__main__':

    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'

  #  daily_folder = '2021-03-26 (Growth)/20210326-132548_Growth_12x2_560'
   # daily_folder = '2021-03-26 (Growth)/20210326-141841_Growth_12x2_570'
    daily_folder = '2021-03-26 (Growth)/20210326-152533_Growth_12x2_580'
    
    parent_folder = os.path.join(base_folder, daily_folder)
    list_of_folders = os.listdir(parent_folder)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
    list_of_folders = [f for f in list_of_folders if re.search('Liveview_Spectrum',f)]
    
    list_of_folders.sort()
    
    L_folder = len(list_of_folders)
    
    #INPUTS
    
    sustrate_bkg_bool = True #True Resta la señal de Background (region de mismo size pero fuera de la fibra optica, en pixel 200)
    
    path_to = os.path.join(parent_folder, 'PROM_STD_processed_data_sustrate_bkg_%s'%str(sustrate_bkg_bool))
    
    if not os.path.exists(path_to):
        os.makedirs(path_to)
        
    first_wave = 501 #region integrate_antistokes
    starts_notch = 521 #region integrate_antistokes
    
    ends_notch = 546 #  #550 gold  #sustrate 700    #water  625 #antistokes 500
    last_wave = 640 # #570 gold  #sustrate 725   #water 670  #antistokes 521
    
    window, deg, repetitions = 21, 0, 1
    
    exposure_time = 1 #s
    spectrum_time = exposure_time #1.5*exposure_time en caso de Run till Abort
    
    for f in list_of_folders:
        folder = os.path.join(parent_folder,f)
        common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum = process_spectrum(folder, path_to, first_wave, starts_notch, ends_notch, last_wave, window, deg, repetitions, spectrum_time, sustrate_bkg_bool)
     #   process_spectrum_prom(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum)
        
       # process_spectrum_prom_fit(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum)
        
      #  process_spectrum_fit(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum)
        
      #  process_spectrum_polynomial(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum)
        
     #   process_spectrum_prom_polynomial(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum)
        
     #   process_spectrum_prom6_polynomial(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum)
     
        process_prom(common_path, NP, initial, final, wavelength, matrix_spectrum, matrix_spectrum_smooth, time, max_spectrum)
        