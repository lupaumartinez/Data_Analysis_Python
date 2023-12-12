# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:35:11 2022

@author: Luciana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import shutil

from scipy.optimize import curve_fit

def create_spectrum(folder, savefolder, center_row, roi):
    
    list_of_files = os.listdir(folder)
    list_of_files = [f for f in list_of_files if re.search('Picture_Andor',f)]
    list_of_files.sort()

    for f in list_of_files:
        
        f_name = f.split('.')[0]
        f_name = f_name.split('Picture_Andor_')[1]
        
        file = os.path.join(folder, f)
        image = skimage.io.imread(file)
    
        spot_size = roi
       # center_row = int(image.shape[0]/2) 
        
        down_row = center_row - int((spot_size-1)/2)
        up_row = center_row + int((spot_size-1)/2) + 1  
        roi_rows = range(down_row, up_row)
        
        im = image[roi_rows]
        spectrum = np.sum(im, axis=0)
        
        filename ='Spectrum_%s.txt'%f_name

        full_filename = os.path.join(savefolder, filename)
        np.savetxt(full_filename, spectrum, fmt='%.3f')
        
    return

def watch_spectrum(folder, center_row_array, roi_array, pixel_i, pixel_j, wavelength_file):
    
    w = os.path.join(folder, wavelength_file)
    wavelength = np.loadtxt(w)
    
    f = 'Picture_Andor_i%04d_j%04d.tiff'%(pixel_i, pixel_j)
        
    f_name = f.split('.')[0]
    f_name = f_name.split('Picture_Andor_')[1]
    
    file = os.path.join(folder, f)
    image = skimage.io.imread(file)
    plot_roi(image, center_row_array, roi_array)
    print('medio', int(image.shape[0]/2) )
             
    spec = np.mean(image, axis=0)
    
    profile_v = np.mean(image, axis=1)
    i_max = np.argmax(profile_v)
    print('argmax profile v', i_max)
    i_array =  np.arange(0, len(profile_v), 1) - i_max
    
    fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True)
    
    ax0.plot(i_array, profile_v)
    ax0.set_xlabel('Pixels')
    ax0.set_ylabel('Mean Counts')
    
    ax1.plot(wavelength, spec)
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Mean Counts')
    
    plt.show()
    
    for roi in roi_array:

        spot_size = roi
        
        for center_row in center_row_array:
            
            down_row = center_row - int((spot_size-1)/2)
            up_row = center_row + int((spot_size-1)/2) + 1  
            roi_rows = range(down_row, up_row)
            
            im = image[roi_rows]
            spectrum = np.mean(im, axis=0)
            profile_v_im = np.mean(im, axis=1)
            i_max_im = np.argmax(profile_v_im)
            i_array_im = np.arange(0, len(profile_v_im), 1) - i_max_im
            
            ax0.plot(i_array_im, profile_v_im, '--')
            ax1.plot(wavelength, spectrum, '--', label = 'in_row_%s_size%s'%(center_row, spot_size))
            
            plt.legend()
            
            plt.show()
        
    return

def plot_roi(image, center_row_array, roi_array):
    
    plt.figure()
    plt.imshow(image)
    
    plt.figure()
    plt.imshow(image)
    
    for roi in roi_array:

        spot_size = roi
        
        for center_row in center_row_array:
            
            down_row = center_row - int((spot_size-1)/2)
            up_row = center_row + int((spot_size-1)/2) + 1
            
            print(spot_size, up_row-down_row)
            
            plt.hlines(up_row, 0, 1002, linestyle = '--')
            plt.hlines(down_row, 0, 1002, linestyle = '-')
            
            plt.show()
            
    return
    
def find_best_row(folder, best_center_row):
    
    list_of_f = os.listdir(folder)
    list_of_f = [f for f in list_of_f if re.search('Picture_Andor',f)]
    list_of_f.sort()
    
    file = os.path.join(folder, list_of_f[0])
    image = skimage.io.imread(file)
    
    profile = np.zeros(image.shape[0])
    
  #  plt.figure()
  #  plt.xlabel('Wavelength (nm)')
  #  plt.ylabel('Mean Counts')

    for f in list_of_f:
        
        f_name = f.split('.')[0]
        f_name = f_name.split('Picture_Andor_')[1]
        
        file = os.path.join(folder, f)
        image = skimage.io.imread(file)
        
        profile_v = np.mean(image, axis=1)
            
    #    center_row = 20
    #    spot_size = 19
    #    down_row = center_row - int((spot_size-1)/2)
    #    up_row = center_row + int((spot_size-1)/2) + 1
    #    plt.vlines(up_row, 0, 5000, linestyle = '--')
    #    plt.vlines(down_row, 0, 5000, linestyle = '-')
     #   plt.plot(profile_v)
     #   plt.show()
        
        profile = profile_v + profile
        
    profile = profile/len(list_of_files)
    
    profile_norm = (profile -np.min(profile))/(np.max(profile)-np.min(profile))
    
    i_max = np.argmax(profile_norm)
    
    best_center_row.append(i_max)
        
    i_array = np.arange(0, len(profile), 1) #- i_max
     
    size = find_size(i_array, profile_norm)
  
    center_row = i_max
    spot_size = size
  
    down_row = center_row - int((spot_size-1)/2)
    up_row = center_row + int((spot_size-1)/2) + 1
    
    bsize = up_row-down_row
    
    best_size_roi.append(bsize)
    
   # plt.vlines(up_row, 0, 1, linestyle = '--')
   # plt.vlines(down_row, 0, 1, linestyle = '-')
   
   # plt.plot(i_array, profile) 
   
    plt.plot(i_array, profile_norm)
    plt.show()
        
    return

def find_size(x, y):
    
    medium = (np.max(y) - np.min(y))/100
    
    d = np.where(y> medium)
    
    xmedium = x[d]
    size = xmedium[-1] - xmedium[0]
    
   # w = FWHM/(np.sqrt(2*np.log(2)))
    
    return size
        
        
def gaussian(x, Io, xo, wo, C):

    return Io*np.exp(-2*(x-xo)**2/(wo**2)) + C

def fit_gaussian(gaussian, x, y):
    
   # mean = sum(x * y)
  #  sigma = sum(y * (x - mean)**2)
    #sigma = np.sqrt(sum(y*(x - mean)**2))
    
    mean = 0
    sigma = 6
    
    bounds = ([0, -3, 1, 0], [1, 3, 30, 1])

    popt, pcov = curve_fit(gaussian, x, y, p0 = [1, mean, sigma, 0], bounds = bounds)
        
    x_fit = np.linspace(-12, 12, 100)
    
    y_fit = gaussian(x_fit, popt[0], popt[1], popt[2], popt[3])
    
    return x_fit, y_fit, popt

def lorentz(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = np.pi
    I, x0, gamma, C = p
    return (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def fit_lorentz(lorentz, x, y):
    
   # mean = sum(x * y)
  #  sigma = sum(y * (x - mean)**2)
    #sigma = np.sqrt(sum(y*(x - mean)**2))
    
    x0 = 0
    gamma = 10
    
   # bounds = ([0, -3, 1, 0], [1, 3, 30, 1])

    popt, pcov = curve_fit(lorentz, x, y, p0 = [1, x0, gamma, 0])#, bounds = bounds)
        
    x_fit = np.linspace(-12, 12, 100)
    
    y_fit = lorentz(x_fit, *popt)
    
    return x_fit, y_fit, popt
        
if __name__ == '__main__':

    plt.close('all')
    base_folder = r'C:\Users\lupau\OneDrive\Documentos'#'C:\Ubuntu_archivos\Printing'
    daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL'
    base_folder = os.path.join(base_folder, daily_folder)
    folder = '20220715-174344_Luminescence_Load_grid'
    
    direction = os.path.join(base_folder, folder)
        
    list_of_all_files = os.listdir(direction)
    list_of_files = [f for f in list_of_all_files if re.search('Slow_Confocal',f)]
    list_of_files.sort()
    
    center_row_array = np.array([20])
    roi_array = np.array([17, 19, 21]) 
   
    find_best_row_bool = True
   
    if find_best_row_bool:
    
        plt.figure()
        plt.xlabel('Pixels')
        plt.ylabel('Normalized Profile')
       # plt.ylabel('Mean Counts')
        best_center_row = []
        best_size_roi = []
    
    for folder_NP in list_of_files[:]:
        
        f = os.path.join(direction, folder_NP)
        print(folder_NP)
        
        list_of_f = os.listdir(f)
        
        wavelength_file = [f for f in list_of_f if re.search('wavelength',f)][0]
       # watch_spectrum(f, center_row_array, roi_array, 5, 5, wavelength_file)
    
        if find_best_row_bool:
            find_best_row(f, best_center_row)
       
    if find_best_row_bool:
        
        best_center_row = np.array(best_center_row)
        print('best size', round(np.median(best_size_roi)))
            
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
        ax1.plot(np.arange(0, len(list_of_files), 1), best_center_row, 'o')
        ax2.plot(np.arange(0, len(list_of_files), 1), best_size_roi, 'o')
        ax1.set_xlabel('Folder NP')
        ax2.set_xlabel('Folder NP')
        ax1.set_ylabel('Best center row')
        ax2.set_ylabel('Best size')
        plt.show()
        
    
    #%%
    
    plt.close('all')
    
    #center_row = 26 #
    roi = 13 #poner impares, los pares los tira para abajo 1 #19
    
    save_folder = os.path.join(base_folder, '%s_in_best_center_size_%s'%(folder, roi))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    for i in range(len(list_of_files)):
        
        folder_NP = list_of_files[i]
        
        f = os.path.join(direction, folder_NP)
        print(folder_NP)
        
        savefolder = os.path.join(save_folder, '%s'%folder_NP)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
            
        list_of_f = os.listdir(f)
        
        wavelength_file = [f for f in list_of_f if re.search('wavelength',f)][0]
        original = os.path.join(f, wavelength_file)
        target = os.path.join(savefolder, wavelength_file)
        shutil.copyfile(original, target)
        
        trace_file = [f for f in list_of_f if re.search('Trace_BS',f)][0]
        original = os.path.join(f, trace_file)
        target = os.path.join(savefolder, trace_file)
        shutil.copyfile(original, target)
        
        center_row = best_center_row[i]
            
        create_spectrum(f, savefolder, center_row, roi)