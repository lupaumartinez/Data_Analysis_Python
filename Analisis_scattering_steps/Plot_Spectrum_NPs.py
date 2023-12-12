# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:53:47 2019

@author: Luciana
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from Correct_Step_and_Glue import glue_steps

from scipy.optimize import curve_fit


def choose_lamp(file):

    lampara = np.loadtxt(file, comments = '#')
    wave_lampara = np.array(lampara[:, 0])
    espectro_lampara = np.array(lampara[:, 1])
    
    smooth_espectro_lampara  =  signal.savgol_filter(espectro_lampara, 7, 0, mode = 'mirror') 
    
  #  plt.figure()
    
  #  plt.plot(wave_lampara, espectro_lampara)
   # plt.plot(wave_lampara, smooth_espectro_lampara)
   # plt.show()
    
    lamp = [wave_lampara, espectro_lampara]
    
    return lamp
    

def line_spectrum_normalized_lamp(wavelength, spectrum_NP, spectrum_bkg, lamp, window_smooth):
    
    smooth_spectrum_NP =  signal.savgol_filter(spectrum_NP, window_smooth, 0, mode = 'mirror')
    
    smooth_spectrum_bkg = signal.savgol_filter(spectrum_bkg, window_smooth, 0, mode = 'mirror')
   # smooth_spectrum_bkg = signal.savgol_filter(spectrum_bkg, 3, 0, mode = 'mirror')
   # smooth_spectrum_bkg = signal.savgol_filter(spectrum_bkg, 3, 0, mode = 'mirror')
   # smooth_spectrum_bkg = signal.savgol_filter(spectrum_bkg, 3, 0, mode = 'mirror')
    
    smooth_spectrum_lamp =  signal.savgol_filter(lamp[1], window_smooth, 0, mode = 'mirror')
    spectrum_lamp, wavelength_lamp = interpolated_lamp(wavelength, [lamp[0], smooth_spectrum_lamp])
 
    normalized_spectrum_lamp = spectrum_lamp/max(spectrum_lamp)
   
    spectrum_NP_normalized = (smooth_spectrum_NP - smooth_spectrum_bkg)/normalized_spectrum_lamp
   # spectrum_NP_normalized = smooth_spectrum_NP/normalized_spectrum_lamp
    
    return wavelength, spectrum_NP_normalized, wavelength_lamp, spectrum_lamp
    
def interpolated_lamp(calibration, lamp):

    lower_lambda = calibration[0]
    upper_lambda = calibration[-1]
    step = len(calibration)
    
    wavelength_lamp = lamp[0]
    spectrum_lamp = lamp[1]
        
    wavelength_new = np.linspace(lower_lambda, upper_lambda, step)

    desired_range = np.where((wavelength_lamp>=lower_lambda) & (wavelength_lamp<=upper_lambda))
    wavelength_lamp = wavelength_lamp[desired_range]
    spectrum_lamp = spectrum_lamp[desired_range]

    new_lamp_spectrum = np.interp(wavelength_new, wavelength_lamp, spectrum_lamp)
        
    return new_lamp_spectrum, wavelength_new

def average(n, arr):
    
    end = n*int(len(arr)/n)
    arr_mean = np.mean(arr[:end].reshape(-1,n),1)
    
    return arr_mean


def procces_data_scattering(folder, col_folders, lamp):

    for number_col in col_folders:
        
        number_folder = 'Col_%03d'%number_col
        
        direction = os.path.join(folder, number_folder)
        
        save_folder = os.path.join(direction, 'fig_normalized_%s'%(number_folder))
        folder_norm = os.path.join(direction, 'normalized_%s'%(number_folder))
       # folder_fit = os.path.join(direction, 'fit_normalized_%s'%(number_folder))
        folder_fit_polynomial = os.path.join(direction, 'fit_polynomial_normalized_%s'%(number_folder))
        
        if not os.path.exists(folder_norm):
            os.makedirs(folder_norm)
            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
    #    if not os.path.exists(folder_fit):
    #        os.makedirs(folder_fit)
            
        if not os.path.exists(folder_fit_polynomial):
            os.makedirs(folder_fit_polynomial)
        
        data_spectrum = []
        name_spectrum =  []
        
        data_bkg = 0
        
        for files in os.listdir(direction):
            if files.startswith("NP"):
                name = os.path.join(direction,  files)
                data_spectrum.append(np.loadtxt(name))
                name_spectrum.append(files)
            if files.startswith("background"):
                name = os.path.join(direction,  files)
                data_bkg = np.loadtxt(name)
        
        data_spectrum = [data_spectrum for _,data_spectrum in sorted(zip(name_spectrum,data_spectrum))]            
        name_spectrum = sorted(name_spectrum)
        
        print('Col_%03d'%number_col, name_spectrum)
        
        exceptions_NP = []
        exceptions_NP_array = np.array(['NP_%03d.txt' % i for i in exceptions_NP])
        
        i = 0    
        
        NP_list = []
        max_wavelength = []
        max_wavelength_poly = []
   #     londa_fit = []
        
        window_smooth = 3#3
        window_smooth_2 = 41#11 #21 #21 #antes 21, 41
        
        plt.figure()
        
        for i in range(len(data_spectrum)):
            if not np.where(exceptions_NP_array == name_spectrum[i])[0].size == 0: continue
            
            wavelength_steps = data_spectrum[i][:,0]
            spectrum_NP_steps = data_spectrum[i][:,1]
            spectrum_bkg_steps = data_bkg[:,1]
            
          #  spectrum_NP_steps = spectrum_NP_steps - spectrum_bkg_steps
            
            wavelength_final, spectrum_final = glue_steps(wavelength_steps, spectrum_NP_steps, number_pixel = 1002, grade = 2, plot_all_step = False)
            wavelength_final, spectrum_bkg_final = glue_steps(wavelength_steps, spectrum_bkg_steps, number_pixel = 1002, grade = 2, plot_all_step = False)
            
            desired_range = np.where((wavelength_final >= 500) & (wavelength_final <=850))    
            wavelength_final = wavelength_final[desired_range]
            spectrum_final = spectrum_final[desired_range]
            spectrum_bkg_final = spectrum_bkg_final[desired_range]
            
            wavelength, spectrum_NP_original, x_lamp, y_lamp = line_spectrum_normalized_lamp(wavelength_final, spectrum_final, spectrum_bkg_final, lamp, window_smooth)
            
            #Original and Smooth:
            spectrum_NP_original_smooth = signal.savgol_filter(spectrum_NP_original, window_smooth_2, 0, mode = 'mirror')
            
            #Normalized and Smooth:
            
         #   offset_noise = np.min(spectrum_NP_original)
         #   spectrum_NP = spectrum_NP_original - offset_noise
         #   spectrum_NP = spectrum_NP/max(spectrum_NP)
            
          #  spectrum_NP = signal.savgol_filter(spectrum_NP_original, window_smooth_2, 0, mode = 'mirror')
          #  offset_noise = np.min(spectrum_NP)
          #  spectrum_NP = spectrum_NP - offset_noise
          #  spectrum_NP = spectrum_NP/max(spectrum_NP)
          
            s = spectrum_NP_original_smooth
            spectrum_NP = (s-min(s))/(max(s)-min(s))
            
            NP = name_spectrum[i].split('NP_')[-1]
            NP = int(NP.split('.')[0])
            NP_list.append(NP)
            
            max_wavelength.append(round(wavelength[np.argmax(spectrum_NP)], 3))
            
            name = os.path.join(folder_norm, name_spectrum[i])
            data = np.array([wavelength, spectrum_NP, spectrum_NP_original_smooth]).T
            np.savetxt(name, data)
            
            #preparo ajuste
            
          #  first_wave = wavelength[np.argmax(spectrum_NP)-50]
          #  end_wave = wavelength[np.argmax(spectrum_NP)+80]
          #  I = 0.9
          #  init_londa = 560
          #  init_width = 100
          #  C = 0.5
          #  init_parameters = np.array([I, init_width, init_londa, C], dtype=np.double)
          #  lorentz_fitted, wave_fitted, londa_max = fit_spr(wavelength, spectrum_NP, first_wave, end_wave, init_param = init_parameters)
            
          #  londa_fit.append(londa_max)
            
          #  name = os.path.join(folder_fit, name_spectrum[i])
          #  data = np.array([wave_fitted, lorentz_fitted]).T
          #  header_text = 'wavelength, fit lorentz'
          #  np.savetxt(name, data, header = header_text)
            
            #Fitting polynomial grade 3
            
            try:
                first_wave = wavelength[np.argmax(spectrum_NP)-30]
                end_wave = wavelength[np.argmax(spectrum_NP)+ 70] #70]
                
                desired_wave, desired_spectrum = cut_range(wavelength, spectrum_NP, first_wave, end_wave)
                npol = 3
                x = np.linspace(desired_wave[0], desired_wave[-1], 1000)
                p = np.polyfit(desired_wave, desired_spectrum, npol)
                poly = np.polyval(p, x)
                max_wave_poly  = round(x[np.argmax(poly)],2)
                plt.plot(x, poly, 'r--')
            except:
                pass
                max_wave_poly = 0
                
            max_wavelength_poly.append(max_wave_poly)
            
            name = os.path.join(folder_fit_polynomial, name_spectrum[i])
            data = np.array([x, poly]).T
            header_text = 'wavelength, fit polynomial'
            np.savetxt(name, data, header = header_text)
            
            plt.plot(wavelength, spectrum_NP, '-', label = name_spectrum[i].split('.')[0])
           # plt.plot(wave_fitted, lorentz_fitted, 'k--')
            
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Intensity')
            plt.xlim(450, 850)
            plt.legend(loc = 'upper right')
                   
        plt.savefig(os.path.join(save_folder, 'NP_Spectrums.png'))
        plt.close()
     #   plt.show()
        
        plt.figure()
        plt.plot(NP_list, max_wavelength, 'o', label = 'max')
        plt.plot(NP_list, max_wavelength_poly, '*', label = 'max polynomial')
       # plt.plot(NP_list, londa_fit, '*', label = 'fit lorentz')
        plt.xlabel('NPs')
        plt.ylabel('Max Wavelength [nm]')
        plt.legend()
        plt.savefig(os.path.join(save_folder, 'Max_wavelength_NP_Spectrums.png'))
        plt.close()
      #  plt.show()
        
        plt.figure()
        plt.hist(max_wavelength, bins=10, rwidth=0.9, color='C1')
        plt.xlabel('Max Wavelength [nm]')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(save_folder, 'hist_Max_wavelength_NP_Spectrums.png'))#, dpi = 500)
        plt.close()
       # plt.show()
        
        plt.figure()
        plt.hist(max_wavelength_poly, bins=10, rwidth=0.9, color='C1')
        plt.xlabel('Max Wavelength Polynomial [nm]')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(save_folder,'hist_Max_Poly_wavelength_NP_Spectrums.png'))#, dpi = 500)
        #plt.show()
        plt.close()
        
       # plt.hist(londa_fit, bins=10, normed = True, rwidth=0.9, color='C1')
       # plt.xlabel('Fit Wavelength LSPR [nm]')
       # plt.ylabel('Frequency')
       # plt.savefig(save_folder + '/hist_Fit_wavelength_NP_Spectrums.png', dpi = 500)
       # plt.show()
        
        name = os.path.join(save_folder, 'max_wavelength.txt')
        data = np.array([NP_list, max_wavelength, max_wavelength_poly]).T
        header_text = 'NP, Max Wavelength (nm), Max Wavelength Polynomial (nm)'
        np.savetxt(name, data, header = header_text)
        
        
def center_spectrum(wavelength, intensity, initial_wave, end_wave):
    
    desired_range = np.where((wavelength >= initial_wave) & (wavelength <=end_wave))    
    wavelength = wavelength[desired_range]
    intensity = intensity[desired_range]
    
    I = np.sum(intensity)
    
    center_wave = round(np.sum(wavelength*intensity)/I, 3)
        
    return center_wave

def fit_lorentz(p, x, y):
    
    try:
        A = curve_fit(lorentz, x, y, p0 = p)

    except RuntimeError:
        print("Error - curve_fit failed")
        A =  np.zeros(4), np.zeros(4)
    
    return A

def lorentz(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = np.pi
    I, gamma, x0, C = p
    return (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def fit_spr(wavelength, intensity, initial_wave, end_wave, init_param):
    
    desired_range = np.where((wavelength >= initial_wave) & (wavelength <=end_wave))    
    wavelength = wavelength[desired_range]
    intensity = intensity[desired_range]

    best_lorentz, err = fit_lorentz(init_param, wavelength, intensity)

    if best_lorentz[0] != 0:

        full_lorentz_fitted = lorentz(wavelength, *best_lorentz)
        londa_max = round(best_lorentz[2], 3)
        
    else: 
        
        full_lorentz_fitted = np.zeros(len(wavelength))
        londa_max = 0
    
    return full_lorentz_fitted, wavelength, londa_max

def cut_range(wavelength, intensity, initial_wave, end_wave):
    
    desired_range = np.where((wavelength >= initial_wave) & (wavelength <=end_wave))    
    wavelength = wavelength[desired_range]
    intensity = intensity[desired_range]
    
    return wavelength, intensity

                 
if __name__ == '__main__':
    
    base_folder = r'C:\Users\lupau\OneDrive\Documentos'
    
    daily_folder = r'2023-03-17 80 nm SS Scattering post PL'
    folder_NPs = r'photos'
    
    folder = os.path.join(base_folder, daily_folder, folder_NPs)
    
    col_folders = [1,2,3,4,5,8,9,10]
    
    mode = 'unpol' #'py' # 'py'  
# folder_lamp = os.path.join(base_folder, 'Luciana Martinez/Programa_Python/Analize Spectrum Growth/Analize_Scattering')
    folder_lamp = os.path.join(base_folder, daily_folder, 'lamparaIR')
 
    if mode == 'unpol':
        #file = os.path.join(folder_lamp, 'Resumen_lamparaIR_new', 'lamparaIR_grade_2.txt')  
        
        file = os.path.join(folder_lamp, 'lamparaIR_grade_2.txt')
        
    if mode == 'px':
        file = os.path.join(folder_lamp, 'Resumen_lamparaIR', 'lamp_IR_PX_grade_2_PySpectrum.txt') 
    if mode == 'py':
        file = os.path.join(folder_lamp, 'Resumen_lamparaIR', 'lamp_IR_PY_grade_2_PySpectrum.txt')
        
 #   file = os.path.join(base_folder, '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)',
  #                      'Scattering_pol_max_laser', 'lampara_slit330_450-950nm_overlap_0.2',
   #                     'lamparaIR_grade_2.txt')
                        
    lamp = choose_lamp(file)
    
    procces_data_scattering(folder, col_folders, lamp)

    
    