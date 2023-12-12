#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:28:16 2021

@author: luciana
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:53:47 2019

@author: Luciana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

def process_spectrum(folder, common_path, starts_notch, ends_notch):
    
    NP = folder.split('NP_')[-1]

    specs = []
    name_spectrum = []
    
    list_of_files = os.listdir(folder)
    list_of_files.sort()
        
    for file in list_of_files:
        
        if file.startswith('wavelength'):
            name_file = os.path.join(folder, file)
            wavelength = np.loadtxt(name_file)
            
        if file.startswith('Spectrum'):
            name_file = os.path.join(folder, file)
            specs.append(np.loadtxt(name_file))
            
            a = file.split('Spectrum_')[-1]
            b = a.split('.txt')[0]
            name_spectrum.append(b)
        
    specs = [specs for _,specs in sorted(zip(name_spectrum,specs))]      
        
    L = len(specs)
            
    print('Number:', NP, 'Spectra acquired:', L)
    
    initial = 0
    final = L
    
    total_frame = len(range(initial, final))
        
    index = []
    max_pl = []
    
    color_map = plt.cm.coolwarm(np.linspace(0,1,total_frame))
    plt.figure()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    plt.title('%1s'%(NP))
    
    first_wave = 545
    end_wave = 620
    desired_range = np.where((wavelength >= first_wave) & (wavelength <=end_wave))    
    desired_wave = wavelength[desired_range]
    x = np.linspace(desired_wave[0], desired_wave[-1], 1000)
    npol = 5
    
    matrix_spectrum = np.zeros((total_frame, len(wavelength)))
    matrix_stokes = np.zeros((total_frame, len(desired_wave)))
    matrix_spectrum_poly = np.zeros((total_frame, len(x)))
    
    intensity = np.zeros(total_frame)
    
    for i in range(initial, final):
        
        spectrum = specs[i]
        matrix_spectrum[i, :] = spectrum
        
        desired_spectrum = spectrum[desired_range]
        matrix_stokes[i, :] = desired_spectrum
        
        p = np.polyfit(desired_wave, desired_spectrum, npol)
        poly = np.polyval(p, x)
        
        matrix_spectrum_poly[i, :] = poly
        max_wave_poly = round(x[np.argmax(poly)],3)
        
        desired = np.where((x >= 550) & (x <=590))    
        intensity[i] = np.sum(poly[desired])
        
        index.append(i)
        
        max_pl.append(max_wave_poly)
            
        plt.plot(wavelength, spectrum)
        plt.plot(x, poly, 'k--')
        
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
        
    figure_name = os.path.join(common_path, 'Confocal_spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
        
    plt.style.use('default')
    
    plt.figure()
    plt.plot(index, max_pl, 'r*', label = 'max wavelegnth de poly 5')
    plt.xlabel('Index')
    plt.ylabel('max wavelength [PL] (nm)')
    plt.legend()
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'wavelength_PL_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
   # plt.figure()
  #  plt.plot(index, intensity, 'r*', label = 'integrate de poly 5')
   # plt.xlabel('Index')
   # plt.ylabel('Integrate 550 nm to 590 nm [PL]')
   # plt.legend()
   # figure_name = os.path.join(common_path, 'integrate_PL_%s.png' % (NP)) 
   # plt.savefig(figure_name , dpi = 400)
   # plt.close()
    
    index_max = np.argmax(intensity)
    specs_max = matrix_spectrum[index_max, :]
    specs_max_poly = matrix_spectrum_poly[index_max, :]
    
    index_min = np.argmin(intensity)
    specs_min = matrix_spectrum[index_min, :]
    specs_min_poly = matrix_spectrum_poly[index_min, :]
    
    plt.figure()
    plt.plot(wavelength, specs_max, 'r')
    plt.plot(x, specs_max_poly, 'k--')
    
    plt.plot(wavelength, specs_min, 'b')
    plt.plot(x, specs_min_poly, 'k--')
    
    plt.plot(wavelength, specs_max - specs_min, 'g')
    plt.plot(x, specs_max_poly - specs_min_poly, 'k--')
        
    plt.xlabel('Wavelength')
    plt.ylabel('PL')
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(path_to, 'max_min_PL_%s.png'% (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    name = os.path.join(common_path, 'max_min_PL_%s.txt' % (NP)) 
    data = np.array([wavelength, specs_max, specs_min]).T
    header_txt = 'wavelength, specs_max, specs_min'
    np.savetxt(name, data, header = header_txt)
    
    name = os.path.join(common_path, 'poly_max_min_PL_%s.txt' % (NP)) 
    data =  np.array([x, specs_max_poly, specs_min_poly]).T
    header_txt =  'wavelength_poly, specs_max_poly, specs_min_poly'
    np.savetxt(name, data, header = header_txt)
    
    return NP, common_path, wavelength, matrix_spectrum, x, matrix_spectrum_poly, specs_max_poly, specs_min_poly

def analyze_last_bin(NP, common_path, wavelength, matrix_spectrum, x, matrix_spectrum_poly, specs_max_poly, specs_min_poly):
    
    L = matrix_spectrum.shape[0]
    
    last_bin = np.sum(specs_max_poly)*0.8
    first_bin = np.sum(specs_min_poly)*1.2
    
    max_wavelength = []
    
    specs_max_bin = np.zeros(len(wavelength))
    specs_min_bin = np.zeros(len(wavelength))
    
    plt.figure()
    plt.title('%1s'%(NP))
    
    j_max = 0
    j_min = 0
    
    for i in range(L):
        
        stokes = matrix_spectrum_poly[i,:]

        
        if np.sum(stokes) > last_bin:
            
            j_max = j_max + 1
            
          #  print(j_max)
            
            specs_max_bin = specs_max_bin + matrix_spectrum[i, :]
            
            max_wavelength.append(round(x[np.argmax(stokes)], 2))
        
            plt.plot(x, stokes, 'r--')
            plt.plot(wavelength, matrix_spectrum[i, :])
            
        if np.sum(stokes) < first_bin:
            
          #  print(j_min)
            
            j_min = j_min + 1
            
            specs_min_bin = specs_min_bin + matrix_spectrum[i, :]
        
            plt.plot(x, stokes, 'b--')
            plt.plot(wavelength, matrix_spectrum[i,:])
        
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
        
    figure_name = os.path.join(common_path, 'bin_Spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
        
    plt.figure()
    plt.plot(max_wavelength, 'r*')
    plt.ylabel('max wavelength [PL] (nm)')
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'last_bin_wavelength_PL_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    specs_max_bin = specs_max_bin/j_max
    specs_min_bin = specs_min_bin/j_min
    
    name = os.path.join(common_path, 'bin_max_min_PL_%s.txt' % (NP)) 
    data = np.array([wavelength, specs_max_bin, specs_min_bin]).T
    header_txt = 'wavelength, specs_max_bin, specs_min_bin'
    np.savetxt(name, data, header = header_txt)
    
    plt.figure()
    plt.plot(wavelength, specs_max_bin, 'r')
    plt.plot(wavelength, specs_min_bin, 'b')
    plt.plot(wavelength, specs_max_bin - specs_min_bin, 'g')
    plt.ylabel('Intensity')
    plt.xlabel('Wavelength (nm)')
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'bin_max_min_PL_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()

    return

def compare(common_path):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if re.search('max_min_PL',f) and re.search('txt',f) 
    and not re.search('bin',f) and not re.search('poly',f) ]
    list_of_folders.sort()
    
    
    for file in list_of_folders:
        
        NP = file.split('PL_')[-1]
        NP = NP.split('.')[0]
        
        name = os.path.join(common_path, file)
        data = np.loadtxt(name, skiprows=1)
        wavelength = data[:, 0]
        max_spectrum = data[:, 1]
        min_spectrum = data[:, 2]
        
        file_bin = 'bin_max_min_PL_%s.txt'%NP
        name_bin = os.path.join(common_path, file_bin)
        data_bin = np.loadtxt(name_bin, skiprows=1)
    #    wavelength = data[:, 0]
        bin_max_spectrum = data_bin[:, 1]
        bin_min_spectrum = data_bin[:, 2]
        
        file_poly = 'poly_max_min_PL_%s.txt'%NP
        name_poly = os.path.join(common_path, file_poly)
        data_poly = np.loadtxt(name_poly,skiprows=1)
        wavelength_poly = data_poly[:, 0]
        poly_max_spectrum = data_poly[:, 1]
        poly_min_spectrum = data_poly[:, 2]
        
        plt.figure()
        plt.title('%1s'%(NP))
        
        plt.plot(wavelength, max_spectrum, 'r')
        plt.plot(wavelength_poly, poly_max_spectrum, 'k--')
        
        plt.plot(wavelength, min_spectrum, 'b')
        plt.plot(wavelength_poly, poly_min_spectrum, 'k--')
        
        plt.plot(wavelength, bin_max_spectrum, 'm--')
        plt.plot(wavelength, bin_min_spectrum, 'g--')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
            
        figure_name = os.path.join(common_path, 'compare_Spectrum_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
    return

def compare_PL(common_path, path_PL):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if re.search('max_min_PL',f) and re.search('txt',f) 
    and not re.search('bin',f) and not re.search('poly',f) ]
    list_of_folders.sort()
    
    list_of_folders = list_of_folders[7:]
    
    print(list_of_folders)
    
    list_of_folders_PL = os.listdir(path_PL)
    list_of_folders_PL = [f for f in list_of_folders_PL if re.search('Luminescence_Steps_Spectrum',f)]
    list_of_folders_PL.sort()
    
 #   list_of_folders_PL = list_of_folders_PL[:8]
    
    print(list_of_folders_PL)
    
    for file in list_of_folders:
        
        NP = file.split('PL_')[-1]
        NP = NP.split('.')[0]
        
        print('NP:', NP)
        
        name = os.path.join(common_path, file)
        data = np.loadtxt(name, skiprows=1)
        wavelength = data[:, 0]
        max_spectrum = data[:, 1]
        min_spectrum = data[:, 2]
        
        file_bin = 'bin_max_min_PL_%s.txt'%NP
        name_bin = os.path.join(common_path, file_bin)
        data_bin = np.loadtxt(name_bin, skiprows=1)
    #    wavelength = data[:, 0]
        bin_max_spectrum = data_bin[:, 1]
        bin_min_spectrum = data_bin[:, 2]
        
        file_poly = 'poly_max_min_PL_%s.txt'%NP
        name_poly = os.path.join(common_path, file_poly)
        data_poly = np.loadtxt(name_poly,skiprows=1)
        wavelength_poly = data_poly[:, 0]
        poly_max_spectrum = data_poly[:, 1]
        poly_min_spectrum = data_poly[:, 2]
        
        n = int(NP) - 10 + 48
        file_PL = 'Luminescence_Steps_Spectrum_Col_005_NP_%03d.txt'%n
        name_PL = os.path.join(path_PL, file_PL)
        data_PL = np.loadtxt(name_PL, skiprows=1)
        wavelength_PL = data_PL[:, 0]
        PL = data_PL[:, 2]
        
        desired = np.where((wavelength >= 546) & (wavelength <=620))
        wavelength = wavelength[desired]
        max_spectrum = max_spectrum[desired]
        
        desired_PL = np.where((wavelength_PL >= 546) & (wavelength_PL <=620))
        wavelength_PL = wavelength_PL[desired_PL]
        PL = PL[desired_PL]
        
        max_spectrum = (max_spectrum - max_spectrum[0])/ (max(max_spectrum) - max_spectrum[0])
        PL = (PL- PL[0])/ (max(PL) - PL[0])
        
        plt.figure()
        plt.title('%1s'%(NP))
        
        plt.plot(wavelength, max_spectrum, 'r', label = 'Confocal PL')
        plt.plot(wavelength_PL, PL, 'g', label = 'PL steps')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Intensity')
        plt.legend()
            
        figure_name = os.path.join(common_path, 'compare_PLsteps_Confocal_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
    return

def compare_PL_RAW(common_path, path_PL, path_PL_processed):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if re.search('max_min_PL',f) and re.search('txt',f) 
    and not re.search('bin',f) and not re.search('poly',f) ]
    list_of_folders.sort()
    
    list_of_folders = list_of_folders[7:]
    
    print(list_of_folders)
    
    list_of_folders_PL = os.listdir(path_PL)
    list_of_folders_PL = [f for f in list_of_folders_PL if re.search('Spectrum_luminescence_Col_005',f)]
    list_of_folders_PL.sort()
    
    print(list_of_folders_PL)
    
    for file in list_of_folders:
        
        NP = file.split('PL_')[-1]
        NP = NP.split('.')[0]
        
        print('NP:', NP)
        
        name = os.path.join(common_path, file)
        data = np.loadtxt(name, skiprows=1)
        wavelength = data[:, 0]
        max_spectrum = data[:, 1]
        min_spectrum = data[:, 2]
        
        file_bin = 'bin_max_min_PL_%s.txt'%NP
        name_bin = os.path.join(common_path, file_bin)
        data_bin = np.loadtxt(name_bin, skiprows=1)
    #    wavelength = data[:, 0]
        bin_max_spectrum = data_bin[:, 1]
        bin_min_spectrum = data_bin[:, 2]
        
        file_poly = 'poly_max_min_PL_%s.txt'%NP
        name_poly = os.path.join(common_path, file_poly)
        data_poly = np.loadtxt(name_poly,skiprows=1)
        wavelength_poly = data_poly[:, 0]
        poly_max_spectrum = data_poly[:, 1]
        poly_min_spectrum = data_poly[:, 2]
        
        n = int(NP) - 10 + 48
        folder_PL = os.path.join(path_PL, 'Spectrum_luminescence_Col_005_NP_%03d'%n)
        name_PL = os.path.join(folder_PL, 'Line_Spectrum_step_0005.txt')
        PL = np.loadtxt(name_PL)
        
        file_PL_wave = os.listdir(folder_PL)
        file_PL_wave = [f for f in file_PL_wave  if re.search('Calibration',f)][0]
        file_PL_wave = os.path.join(folder_PL, file_PL_wave)
        wavelength_PL = np.loadtxt(file_PL_wave)
        
        file_PL_processed = 'Luminescence_Steps_Spectrum_Col_005_NP_%03d.txt'%n
        name_PL_processed = os.path.join(path_PL_processed, file_PL_processed)
        data_PL_processed = np.loadtxt(name_PL_processed, skiprows=1)
        wavelength_PL_processed = data_PL_processed[:, 0]
        PL_processed = data_PL_processed[:, 2]
        
        max_spectrum = max_spectrum - min(max_spectrum)
        PL = PL - min(PL)
        PL_processed = PL_processed - min(PL_processed)
        
        wave, spectrum, wave_pl, pl, wave_pl_processed, pl_processed = plot_desired_range(wavelength, max_spectrum, 
        wavelength_PL, PL, wavelength_PL_processed, PL_processed, first_wave = 546, end_wave = 620)
        
        plt.figure()
        plt.title('%1s'%(NP))
        
        plt.plot(wave, spectrum, 'r', label = 'Confocal PL')
        plt.plot(wave_pl, pl, 'g', label = 'PL steps')
        plt.plot(wave_pl_processed, pl_processed, 'k--', label = 'PL steps processed')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Intensity')
        plt.legend()
            
        figure_name = os.path.join(common_path, 'PLstepsRAW_Confocal_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        wavelength, max_spectrum, wavelength_PL, PL, wavelength_PL_processed, PL_processed = plot_desired_range(wavelength, max_spectrum, 
        wavelength_PL, PL, wavelength_PL_processed, PL_processed, first_wave = 500, end_wave = 620)
        
        plt.figure()
        plt.title('%1s'%(NP))
        
        plt.plot(wavelength, max_spectrum, 'r', label = 'Confocal PL')
        plt.plot(wavelength_PL, PL, 'g', label = 'PL steps')
        plt.plot(wavelength_PL_processed, PL_processed, 'k--', label = 'PL steps processed')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Intensity')
        plt.legend()
            
        figure_name = os.path.join(common_path, 'All_PLstepsRAW_Confocal_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
    return

def plot_desired_range(wavelength, max_spectrum, wavelength_PL, PL, wavelength_PL_processed, PL_processed, first_wave, end_wave):
            
    desired = np.where((wavelength >= first_wave) & (wavelength <=end_wave))
    wavelength = wavelength[desired]
    max_spectrum = max_spectrum[desired]
    
    desired_PL = np.where((wavelength_PL >= first_wave) & (wavelength_PL <=end_wave))
    wavelength_PL = wavelength_PL[desired_PL]
    PL = PL[desired_PL]
    
    desired_PL_processed = np.where((wavelength_PL_processed >= first_wave) & (wavelength_PL_processed <=end_wave))
    wavelength_PL_processed = wavelength_PL_processed[desired_PL_processed]
    PL_processed = PL_processed[desired_PL_processed]
    
    max_spectrum = (max_spectrum - max_spectrum[0])/ (max(max_spectrum) - max_spectrum[0])
    PL_processed = (PL_processed- PL[0])/ (max(PL) - PL[0])
    PL = (PL- PL[0])/ (max(PL) - PL[0])
   
   # max_spectrum = max_spectrum/max_spectrum[0]
   # PL_processed = PL_processed/PL_processed[0]
   # PL = PL/PL[0]
    
    return wavelength, max_spectrum, wavelength_PL, PL, wavelength_PL_processed, PL_processed

def smooth_Signal(signal, window, deg, repetitions):
    
    k = 0
    while k < repetitions:
        signal = sig.savgol_filter(signal, window, deg, mode = 'mirror')
        k = k + 1
        
    return signal


if __name__ == '__main__':

    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'

    daily_folder = '2021-03-26 (Growth)/Confocal_Juli/20210422-155516_Luminescence_10x2_Columna5_580'
    
    #%%
    
    parent_folder = os.path.join(base_folder, daily_folder)
    list_of_folders = os.listdir(parent_folder)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
    list_of_folders = [f for f in list_of_folders if re.search('Slow_Confocal',f)]
    
    list_of_folders.sort()
    
    L_folder = len(list_of_folders)
    
    #INPUTS
 
    path_to = os.path.join(parent_folder, 'TEST_Poly5_Confocal')
    
    if not os.path.exists(path_to):
        os.makedirs(path_to)
        
    starts_notch = 521 #region integrate_antistokes
    ends_notch = 546 #  #550 gold  #sustrate 700    #water  625 #antistokes 500
    
  #  exposure_time = 2 #s
    
    for f in list_of_folders:
        folder = os.path.join(parent_folder,f)
        NP, common_path, wavelength, matrix_spectrum, x, matrix_spectrum_poly, specs_max_poly, specs_min_poly = process_spectrum(folder, path_to, starts_notch, ends_notch) 
        analyze_last_bin(NP, common_path, wavelength, matrix_spectrum, x, matrix_spectrum_poly, specs_max_poly, specs_min_poly)
     
    
    compare(path_to)
    
    #%%
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'

    daily_folder = '2021-03-26 (Growth)/Confocal_Juli/20210422-155516_Luminescence_10x2_Columna5_580'
    parent_folder = os.path.join(base_folder, daily_folder)
    path_to = os.path.join(parent_folder, 'TEST_Poly5_Confocal')
    
    daily_folder_PL = '2021-03-26 (Growth)/20210326-184459_Luminescence_Steps_12x10/'
    sub_file = 'processed_data_luminiscence_sustrate_bkg_False/Col_005_01/luminescence_steps'
    path_PL_processed = os.path.join(base_folder, daily_folder_PL, sub_file)

    compare_PL(path_to, path_PL, path_PL_processed)
    
    #%%
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'

    daily_folder = '2021-03-26 (Growth)/Confocal_Juli/20210422-155516_Luminescence_10x2_Columna5_580'
    parent_folder = os.path.join(base_folder, daily_folder)
    path_to = os.path.join(parent_folder, 'TEST_Poly5_Confocal')
    
    daily_folder_PL = '2021-03-26 (Growth)/20210326-184459_Luminescence_Steps_12x10/'
    path_PL = os.path.join(base_folder, daily_folder_PL)
    
    sub_file = 'processed_data_luminiscence_sustrate_bkg_False/Col_005_01/luminescence_steps'
    path_PL_processed = os.path.join(base_folder, daily_folder_PL, sub_file)
    
    compare_PL_RAW(path_to, path_PL, path_PL_processed)