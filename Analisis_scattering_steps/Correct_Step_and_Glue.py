#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:49:18 2020

@author: luciana
"""

import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

import scipy.signal as sig

def smooth_Signal(signal, window, deg, repetitions):
    
    k = 0
    while k < repetitions:
        signal = sig.savgol_filter(signal, window, deg, mode = 'mirror')
        k = k + 1
        
    return signal

def glue_steps(wave_PySpectrum, spectrum_py, number_pixel, grade, plot_all_step):

    L = int(len(spectrum_py)/number_pixel) #cantidad de steps
    
    n_skip_points = 30
    n = int(n_skip_points/2)
    
    spec_steps = np.zeros((number_pixel-n_skip_points, L))
    wave_steps = np.zeros((number_pixel-n_skip_points, L))
    
    spec_steps_glue = np.zeros((number_pixel-n_skip_points, L))
    wave_steps_glue = np.zeros((number_pixel-n_skip_points, L))
    
    list_of_inf = np.zeros(L)
    list_of_sup = np.zeros(L)
    
    for i in range(L):
        
        spec = spectrum_py[i*number_pixel:number_pixel*(1+i)]
        wave = wave_PySpectrum[i*number_pixel:number_pixel*(i+1)]
        
        spec_steps[:, i] = spec[n:-n]
        wave_steps[:, i] = wave[n:-n]
        
        spec_steps_glue[:, i] = spec[n:-n]
        wave_steps_glue[:, i] = wave[n:-n]
        
        list_of_inf[i] = wave_steps[0, i] # wave[0]
        list_of_sup[i] = wave_steps[-1, i]
    
    for j in range(L-1):
        
        inf = list_of_inf[j+1]
       # sup = list_of_sup[i]
        
        wave_tail = wave_steps[:, j]   
        desired_range_tail = np.where(wave_tail >= inf)[0]
        
        m = int( len(desired_range_tail))
        
        weigth_h = np.linspace(0, 1, m)**grade
        weigth_t = np.flip(weigth_h)
        
        coef = weigth_h + weigth_t
        
        weigth_h =  weigth_h/coef
        weigth_t =  weigth_t/coef
        
        desired_range_tail = range(number_pixel-n_skip_points - m,  number_pixel-n_skip_points)
        spec_tail = spec_steps[desired_range_tail , j]
        wave_tail = wave_steps[desired_range_tail , j]
        
        desired_range_head = range(0,  m)
        spec_head = spec_steps[desired_range_head, j+1]
        wave_head = wave_steps[desired_range_head, j+1]
        
        spec_weigth = weigth_h*spec_head + weigth_t*spec_tail
        
       # spec_weigth = smooth_Signal(spec_weigth, window = 21, deg = 0, repetitions = 1)

        spec_steps_glue[desired_range_tail, j] = spec_weigth
        wave_steps_glue[desired_range_tail, j] = wave_tail
        
        spec_steps_glue[desired_range_head, j+1] = spec_weigth
        wave_steps_glue[desired_range_head, j+1] = wave_head
        
    wave_final = np.reshape(wave_steps_glue, [1,wave_steps_glue.size])[0]
    spectrum_final = np.reshape(spec_steps_glue, [1,wave_steps_glue.size])[0]
    
    spectrum_final = [spectrum_all for _,spectrum_all in sorted(zip(wave_final,spectrum_final))]
    wave_final = np.sort(wave_final)
    spectrum_final = np.array(spectrum_final)
    
    wave_final, spectrum_final = interpole_spectrum(wave_final, spectrum_final, number_pixel)
    
    if plot_all_step:
    
        plt.figure()
        plt.plot(wave_final, spectrum_final, 'ko-')
        
        for i in range(L):
            plt.plot(wave_steps[:,i], spec_steps[:,i], 'o')
            
        plt.show()

    return wave_final, spectrum_final


def select_ROI(image, center_row, spot_size):
    
    down_row = center_row - int((spot_size-1)/2)
    up_row = center_row + int((spot_size-1)/2) + 1  
    roi_rows = range(down_row, up_row)
    
    spectrum = np.round(np.mean(image[:,roi_rows], axis=1),2)
    
    return spectrum

def glue_photos(wavelength, image, number_pixel, grade, plot_all_step):
    
    large = image.shape[1]
    
    #image_glue = np.zeros((image.shape[0], image.shape[1]))
    
    # factor = number_pixel/window_wavelength #103
    
  #  desired_points = round(factor*(wavelength[-1] - wavelength[0]))
  
    desired_points = number_pixel
    
    image_glue = np.zeros((desired_points, desired_points))
    
  #  plt.figure()
    
    for i in range(large):
        spectrum_row = image[:, i]
        wave_final, spectrum_final = glue_steps(wavelength, spectrum_row, number_pixel, grade, plot_all_step)
  # plt.plot(wave_final, spectrum_final)
        image_glue[:, i] = spectrum_final
  #  plt.show()
    
    return wave_final, image_glue

def interpole_spectrum(wavelength, spectrum, number_pixel):
    
   # factor = number_pixel/window_wavelength
    
  #  desired_points = round(factor*(wavelength[-1] - wavelength[0]))
    
    desired_points = number_pixel
    
    lower_lambda = wavelength[0]
    upper_lambda = wavelength[-1]
    
    wavelength_new = np.linspace(lower_lambda, upper_lambda, desired_points)

    spectrum_new = np.interp(wavelength_new, wavelength, spectrum)
    
    return wavelength_new, spectrum_new