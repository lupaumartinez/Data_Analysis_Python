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

def find_lspr(file, save_folder, fit_lorentz):
    
    
    data = np.loadtxt(file)
    wavelength = data[:,0]
    spectrum = data[:,1]
    
    wave = wavelength
   
  #  n =  3
  #  spectrum = signal.savgol_filter(spectrum, n, 0, mode = 'mirror')
    
    max_wavelength = round(wave[np.argmax(spectrum)], 0)
    
     #cut range
    first_wave = wave[np.argmax(spectrum)-70]
    end_wave = wave[np.argmax(spectrum)+70]
    
   # first_wave = 550
   # end_wave = 585
    desired_wave, desired_spectrum = cut_range(wave, spectrum, first_wave, end_wave)
    
    if fit_lorentz:
    #preparo ajuste lorentz
        I = 0.90
        init_londa = sum(desired_wave * desired_spectrum)
        init_width = np.sqrt(sum(desired_spectrum*(desired_wave - init_londa)**2))
        C = 0.5
        init_parameters = np.array([I, init_width, init_londa, C], dtype=np.double)
        lorentz_fitted, wave_fitted, londa_max, intensity_max = fit_spr(desired_wave, desired_spectrum, init_param = init_parameters)
       
    #preparo ajuste F
   # I = 0.9
   # init_londa = sum(desired_wave * desired_spectrum)
   # init_width = np.sqrt(sum(desired_spectrum*(desired_wave - init_londa)**2))
   # a = 0#0.15 #-0.4
   # m = 0#-6
   # C = 0.5
    #init_parameters = np.array([I, init_width, init_londa, a, m, C], dtype=np.double)
    #F_fitted, wave_F_fitted, londa_max_F, intensity_max_F = fit_spr_F(desired_wave, desired_spectrum, init_param = init_parameters)
    
    #preparo ajuste polinomial
    npol = 3
    x = np.linspace(desired_wave[0], desired_wave[-1], 1000)
    
    pp = np.polyfit(desired_wave, desired_spectrum, npol)
    ppoli = np.polyval(pp, x)
    ap = round(x[np.argmax(ppoli)],0)
    
    print('r2 polinomio de grado ', npol, ':', calc_r2(desired_spectrum, np.polyval(pp, desired_wave)))
    
    plt.figure()
    plt.plot(wave, spectrum)
    plt.show()
    
    plt.figure()
    
    plt.plot(desired_wave, desired_spectrum, 'b')
    plt.plot(max_wavelength, spectrum[np.argmax(spectrum)], 'bo')

    plt.plot(x, ppoli, 'r--', label = 'polinomio')
    plt.plot(ap, ppoli[np.argmax(ppoli)], 'ro')
    
    if fit_lorentz:
        if londa_max != 0:
            plt.plot(wave_fitted, lorentz_fitted, 'k--', label = 'lorentz')
            plt.plot(londa_max, intensity_max, 'ko')
        
    plt.legend()
    plt.show()
    
    print('max:', max_wavelength)
    print('max poli', ap)
    if fit_lorentz:
        print('londa lorentz:', londa_max)
    
    
def cut_range(wavelength, intensity, initial_wave, end_wave):
    
    desired_range = np.where((wavelength >= initial_wave) & (wavelength <=end_wave))    
    wavelength = wavelength[desired_range]
    intensity = intensity[desired_range]
    
    return wavelength, intensity
    
    
def center_spectrum(wavelength, intensity, initial_wave, end_wave):
    
    desired_range = np.where((wavelength >= initial_wave) & (wavelength <=end_wave))    
    wavelength = wavelength[desired_range]
    intensity = intensity[desired_range]
    
    I = np.sum(intensity)
    
    center_wave = round(np.sum(wavelength*intensity)/I, 3)
    
    intensity = intensity[closer(intensity, center_wave)]
        
    return center_wave, intensity

def calc_r2(observed, fitted):
    # Calculate coefficient of determination
    avg_y = observed.mean()
    ssres = ((observed - fitted)**2).sum()
    sstot = ((observed - avg_y)**2).sum()
    return round(1.0 - ssres/sstot, 3)
    

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

def fit_spr(wavelength, intensity, init_param):

    best_lorentz, err = fit_lorentz(init_param, wavelength, intensity)

    if best_lorentz[0] != 0:

        full_lorentz_fitted = lorentz(wavelength, *best_lorentz)
        londa_max = round(best_lorentz[2], 3)
        
    else: 
        
        full_lorentz_fitted = np.zeros(len(wavelength))
        londa_max = 0
        
    intensity_max = lorentz(londa_max, *best_lorentz)
    
    return full_lorentz_fitted, wavelength, londa_max, intensity_max

def lor(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # A = amplitude
    # w0 = center
    pi = np.pi
    A, gamma, w0 = p
    
    return (1/pi) * A * (gamma/2)**2 / ((x - w0)**2 + (gamma/2)**2)
    #return  A/(2*pi)* gamma / ((x - w0)**2 + (gamma/2)**2)

def gauss(x, *p):
    # Gauss fitting function with an offset
    # gamma = FWHM
    # A = amplitude
    # w0 = center
    pi = np.pi
    A, gamma, w0 = p
    return  (A/gamma)*np.sqrt(4*np.log(2)/pi)*np.exp(-4*np.log(2)*((x - w0)/gamma)**2)

def P(x, a, *p):
    
    A, gamma, w0 = p
    
    return 1 - a*((x-w0)/gamma)*np.exp(-(1/2)*((x-w0)/(2*gamma))**2)

def F(x, *parameters):
    
    A, gamma, w0, a, m, cte = parameters
    
    p = A, gamma, w0
    
    return m*gauss(x*P(x, a, *p), *p) + (1-m)*lor(x*P(x, a, *p), *p) + cte

def fit_F(parameters, x, y):
    
    bounds = ([0,80, 550, -1, -np.inf, -np.inf], [np.inf, np.inf, 600, 1, 0, np.inf])
    
    try:
        A = curve_fit(F, x, y, p0 = parameters)#, bounds = bounds)

    except RuntimeError:
        print("Error - curve_fit failed")
        A =  np.zeros(6), np.zeros(6)
    
    return A

def fit_spr_F(wavelength, intensity, init_param):
    
   # wavelength = np.array(wavelength)
   # intensity = np.array(intensity)
    
    best_F, err = fit_F(init_param, wavelength, intensity)
    
    print(best_F)

    if best_F[0] != 0:

        full_F_fitted = F(wavelength, *best_F)
        londa_max = round(best_F[2], 3)
        
    else: 
        
        full_F_fitted = np.zeros(len(wavelength))
        londa_max = 0
        
    intensity_max = F(londa_max, *best_F)
    
   # print(best_F, err)
    
    return full_F_fitted, wavelength, londa_max, intensity_max


if __name__ == '__main__':
    
    scattering = False
    PL = True
    
    if scattering:
        
    
        base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
        daily_folder = '2021-03-26 (Growth)/Scattering_growth/'
       # subfile = 'PX/photos/Col_005/normalized_Col_005/NP_004.txt'
        subfile = 'PX/photos/Col_005/normalized_Col_005/NP_001.txt'
        
        file = os.path.join(base_folder, daily_folder, subfile)
        
        save_folder = os.path.join(base_folder, daily_folder, 'pruebas_maximo')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        find_lspr(file, save_folder, fit_lorentz = True)

    if PL:
        
        base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
        
        daily_folder = '2021-03-26 (Growth)/20210326-184459_Luminescence_Steps_12x10/processed_data_luminiscence_sustrate_bkg_False/'
        
        subfile = 'Col_005_01/luminescence_steps/Luminescence_Steps_Spectrum_Col_005_NP_058.txt'
        
        file = os.path.join(base_folder, daily_folder, subfile)
        
        save_folder = os.path.join(base_folder, daily_folder, 'pruebas_maximo')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        find_lspr(file, save_folder,fit_lorentz = False)