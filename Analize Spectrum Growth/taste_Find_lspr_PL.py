#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:35:28 2021

@author: luciana
"""


#Pruebas busqueda de lspr: interpolo la señal para que tenga mas datos, a eso la suavizo y a eso le busco el maximo.


import os
import re
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import scipy.signal as signal

def find_lspr(file, save_folder):
    
    
    data = np.loadtxt(file)
    
    wavelength = data[:,0]
    
    spectrum = data[:,1]
    
    print(len(spectrum))
    
    wave = wavelength
    
 #   wave = np.linspace(wavelength[0], wavelength[-1], 200)
 #   spectrum = np.interp(wave, wavelength, spectrum)
    
    print(len(spectrum))
   
    n = 21
    
    spectrum = signal.savgol_filter(spectrum, n, 0, mode = 'mirror')
    
    center_wave, center_intensity = center_spectrum(wave, spectrum, 520, 600)
    
    max_wavelength = round(wave[np.argmax(spectrum)], 3)
    
    #preparo ajuste
    I = 0.9
    init_londa = 560
    init_width = 100
    C = 0.5
    init_parameters = np.array([I, init_width, init_londa, C], dtype=np.double)
    lorentz_fitted, wave_fitted, londa_max, intensity_max = fit_spr(wave, spectrum, 530, 640, init_param = init_parameters)
    
    plt.figure()
    plt.plot(wave, spectrum)
    plt.plot(wave_fitted, lorentz_fitted, 'k--')
    
    plt.plot(max_wavelength, spectrum[np.argmax(spectrum)], 'ro')
    plt.plot(center_wave, center_intensity, 'bo')
    plt.plot(londa_max, intensity_max, 'ko')
    
    
    plt.show()
    
    
def center_spectrum(wavelength, intensity, initial_wave, end_wave):
    
    desired_range = np.where((wavelength >= initial_wave) & (wavelength <=end_wave))    
    wavelength = wavelength[desired_range]
    intensity = intensity[desired_range]
    
    I = np.sum(intensity)
    
    center_wave = round(np.sum(wavelength*intensity)/I, 3)
    
    intensity = intensity[closer(intensity, center_wave)]
        
    return center_wave, intensity

def closer(x,value):
    # returns the index of the closest element to value of the x array
    out = np.argmin(np.abs(x-value))
    return out

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
        
    intensity_max = lorentz(londa_max, *best_lorentz)
    
    return full_lorentz_fitted, wavelength, londa_max, intensity_max
    

if __name__ == '__main__':
    
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
    
    daily_folder = '2021-03-26 (Growth)/20210326-184459_Luminescence_Steps_12x10/processed_data_luminiscence_sustrate_bkg_False/'
    
    subfile = 'Col_005_01/luminescence_steps/Luminescence_Steps_Spectrum_Col_005_NP_058.txt'
    
    file = os.path.join(base_folder, daily_folder, subfile)
    
    save_folder = os.path.join(base_folder, daily_folder, 'pruebas_maximo')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    
    find_lspr(file, save_folder)