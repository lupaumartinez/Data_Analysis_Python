#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:11:59 2019

@author: luciana

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import scipy.signal as sig

def smooth_Signal(signal, window, deg, repetitions):
    
    k = 0
    while k < repetitions:
        signal = sig.savgol_filter(signal, window, deg, mode = 'mirror')
        k = k + 1
        
    return signal

    
def lorentz(x, *p):
    
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = np.pi
    I, gamma, x0, C = p
    
    return (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def fit_lorentz(p, x, y):
    return curve_fit(lorentz, x, y, p0 = p)


def fit_two_lorentz(p, x, y):
    return curve_fit(two_lorentz, x, y, p0 = p, bounds = (0, [2000, 500, 700, 2000, 500, 700, 500]))#, method = 'trf')

def fit_three_lorentz(p, bounds, x, y):
    return curve_fit(three_lorentz, x, y, p0 = p, bounds = bounds)#bounds = ([0, 0, 530, 0, 0, 0], [2000, 100, 800, 100, 100, 500]))

def two_lorentz(x, *p):

    pi = np.pi

    I, gamma, x0, I_2, gamma_2, x0_2, C = p
    return  (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + (1/pi) * I_2 * (gamma_2/2)**2 / ((x - x0_2)**2 + (gamma_2/2)**2) + C

def three_lorentz(x, *p):

    pi = np.pi

    I, gamma, x0, I_2, I_3, C = p
    
    a = (1/pi) * I_2 * (15.5/2)**2 / ((x - 649)**2 + (15.2/2)**2) 
    b = (1/pi) * I_3 * (183/2)**2 / ((x - 702)**2 + (183/2)**2) 
    
    return  (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + (1/pi) * I_2 * a + (1/pi) * I_3 * b + C

def calc_r2(observed, fitted):
    # Calculate coefficient of determination
    avg_y = observed.mean()
    ssres = ((observed - fitted)**2).sum()
    sstot = ((observed - avg_y)**2).sum()
    return 1.0 - ssres/sstot

def fit_lorentz_signal(spectrum_S, wavelength_S, init_params):

    best_lorentz, err = fit_two_lorentz(init_params, wavelength_S, spectrum_S)

    lorentz_fitted = two_lorentz(wavelength_S, *best_lorentz)
    r2_coef_pearson = calc_r2(spectrum_S, lorentz_fitted)

    return lorentz_fitted, best_lorentz, r2_coef_pearson

def fit_lorentz_signal_NP(spectrum_S, wavelength_S, init_params, bounds):

    best_lorentz, err = fit_three_lorentz(init_params, bounds, wavelength_S, spectrum_S)

    lorentz_fitted = three_lorentz(wavelength_S, *best_lorentz)
    r2_coef_pearson = calc_r2(spectrum_S, lorentz_fitted)

    return lorentz_fitted, best_lorentz, r2_coef_pearson

def stokes_signal(wavelength, spectrum, ends_notch, final_wave):
    
    desired_range_stokes = np.where((wavelength > ends_notch + 2) & (wavelength <= final_wave -2))
    wavelength_S = wavelength[desired_range_stokes]
    spectrum_S = spectrum[desired_range_stokes]
    
    return wavelength_S, spectrum_S

def plot_signal(wavelength_NP, signal_NP, wavelength_substrate, signal_substrate, window, deg, repetitions, ends_notch, final_wave):
    
    specs_substrate = [spectrum_all for _,spectrum_all in sorted(zip(wavelength_substrate, signal_substrate))]
    wavelength_substrate = np.sort(wavelength_substrate)
    specs_substrate = np.array(specs_substrate)
    
    specs_NP = [spectrum_all for _,spectrum_all in sorted(zip(wavelength_NP, signal_NP))]
    wavelength_NP = np.sort(wavelength_NP)
    specs_NP = np.array(specs_NP)
    
   # spectrum_substrate = smooth_Signal(specs_substrate, window, deg, repetitions)
   # spectrum_NP = smooth_Signal(specs_NP, window, deg, repetitions)
    
    spectrum_substrate = specs_substrate
    spectrum_NP = specs_NP

    wavelength_substrate_S, spectrum_substrate_S = stokes_signal(wavelength_substrate, spectrum_substrate, ends_notch, final_wave)
    wavelength_NP_S, spectrum_NP_S = stokes_signal(wavelength_NP, spectrum_NP, ends_notch, final_wave)
    
    I = 1200
    init_londa = 649
    init_width = 15
    init_londa2 = 700
    init_width2 = 180
    I2 = 200
    C = 450
    init_parameters = np.array([I, init_width, init_londa, I2, init_width2, init_londa2, C], dtype=np.double)
    lorentz_fitted, best_lorentz, r2_coef_pearson = fit_lorentz_signal(spectrum_substrate_S, wavelength_substrate_S, init_parameters)
     
    print('Fit Lorentz', best_lorentz, 'r', r2_coef_pearson)
    
    I = 2500#20000#6000
    init_londa = 550
    init_width = 50
    I2 = 100#2500#1000
    I3 = 100#0#0
    C = 450
    init_parameters_NP = np.array([I, init_width, init_londa, I2, I3, C], dtype=np.double)
    bounds = ([0, 0, 500, 0, 0, 0], [10000, 300, 1000, 5000, 5000, 700])
    lorentz_fitted_NP, best_lorentz_NP, r2_coef_pearson_NP = fit_lorentz_signal_NP(spectrum_NP_S, wavelength_NP_S, init_parameters_NP, bounds)  
   
    print('Fit Lorentz NP', best_lorentz_NP, 'r', r2_coef_pearson_NP)
    print('Ploteo signal')
    
    plt.figure()      
   # plt.plot(wavelength_substrate, specs_substrate, label = 'signal on substrate')
    plt.plot(wavelength_substrate, spectrum_substrate, 'k-', label = 'smooth_w%d_d%d_r%d'%(window, deg, repetitions))
    plt.plot(wavelength_substrate_S, lorentz_fitted, 'r--', label = 'fit lorentz substrate')
    
    plt.plot(wavelength_NP, spectrum_NP, 'g-', label = 'smooth signal NP')
    plt.plot(wavelength_NP_S, lorentz_fitted_NP, 'b--', label = 'fit lorentz NP')
    
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(521, ends_notch, facecolor='grey', alpha=0.2)
  #  plt.ylim(0, 1500)
    plt.legend(loc='upper right') 
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Counts')
    
    plt.show()

if __name__ == '__main__':

    base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2019/Mediciones_PySpectrum/'
       
    daily_substrate = '2019-10-18 (growth HAuCl4 1 mW 532nm on SS 50nm PDDA)/20191018-201803_Spectrum_Measurment_step_04_sustrato_laser_high/'
    file_substrate = os.path.join(base_folder, daily_substrate, 'Line_Spectrum_step_0004.txt')
    signal_substrate = np.loadtxt(file_substrate)
    calibration_wavelength = os.path.join(base_folder, daily_substrate, '20191018_201803_Calibration_Shamrock_Spectrum_step_0004.txt')
    wavelength_substrate = np.loadtxt(calibration_wavelength)
 
    NP = 'Col_002_NP_019'#'Col_005_NP_054'#'Col_007_NP_072'
    daily_folder = '2019-10-18 (growth HAuCl4 1 mW 532nm on SS 50nm PDDA)/20191018-191609_Luminescence_Steps_12x10/Spectrum_luminescence_%s'%NP
    file_NP = os.path.join(base_folder, daily_folder, 'Line_Spectrum_step_0004.txt') 
    calibration_wavelength_NP = calibration_wavelength
   
  #  NP= 'Col_001_NP_002'
  #  daily_folder = '2019-09-20 (growth HAul4 on NPC Au 60 nm PSS)/20190920-175045_Growth_10x1/Liveview_Spectrum_%s'%NP 
  #  file_NP = os.path.join(base_folder, daily_folder, 'Line_Spectrum_step27.txt')
   # calibration_wavelength_NP = os.path.join(base_folder, daily_folder, 'Calibration_Shamrock_Spectrum.txt')
    
    signal_NP = np.loadtxt(file_NP)
    wavelength_NP = np.loadtxt(calibration_wavelength_NP)
    
    window, deg, repetitions = 51, 0, 1
    
    ends_notch = 545
    final_wave = 800
    
    plot_signal(wavelength_NP, signal_NP, wavelength_substrate, signal_substrate, window, deg, repetitions, ends_notch, final_wave)


        
