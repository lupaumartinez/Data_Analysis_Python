#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:37:31 2019

@author: luciana
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:53:47 2019

@author: Luciana
"""

from tkinter import Tk, filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def open_data(direction, save_folder):
        
    all_data = []
    
    for files in os.listdir(direction):
        if files.startswith("Col"):
            new_direction = os.path.join(direction, files)
            for files in os.listdir(new_direction):
                if files.startswith("fig_normalized"):
                    name_data = os.path.join(new_direction, files, "max_wavelength.txt")
                    data = np.loadtxt(name_data)
                    maxpoly = np.array(data[:,2]) #
                    all_data.extend(maxpoly) #extend
                     
    name = os.path.join(save_folder, 'all_max_wavelength_polynomial.txt')
    all_data = np.array(all_data).T
    np.savetxt(name, all_data)
    
    return


def plot_data(save_folder, name):
    
    file = os.path.join(save_folder, '%s.txt'%name)
    max_wavelength = np.loadtxt(file)
    
    L = len(max_wavelength)
    print(max_wavelength)
    
    mean = round(np.mean(max_wavelength),2)
    std = round(np.std(max_wavelength),2)
    
    print('valor medio', mean, 'desviasión estándar', std)
    
    plt.figure()
    
    plt.hist(max_wavelength, bins=7, rwidth=0.9, color='C1', label = 'N = %d'%L)
    plt.xlabel('Max Wavelength (nm)')
    plt.ylabel('Frequency')
    plt.axvspan(mean - std, mean + std, color = 'green', alpha = 0.3)
    plt.axvline(mean, color = 'red', linestyle = '--')
    plt.text(mean + 0.001*mean, 0.008, ' %2d + %2d nm'%(mean, std), color = 'red')
    plt.xlim(mean - 2*std, mean + 2*std)
    #plt.ylim(0, 0.1)
    plt.legend()
    plt.savefig(save_folder + '/'+ 'hist_%s.png'%name, dpi = 500)
    plt.close()
    
    return

def plot_all_spectrums(direction, save_folder):
    
    plt.figure()
    
    for files in os.listdir(direction):
        if files.startswith("Col"):
            new_direction = os.path.join(direction, files)
            for files in os.listdir(new_direction):
                if files.startswith("normalized"):
                    folder = os.path.join(new_direction, files)
                    list_of_NPs = os.listdir(folder)
                    
                    for f in list_of_NPs:
                        file = os.path.join(folder, f)
                        data = np.loadtxt(file)
                        wave = data[:, 0]
                        spec = data[:, 1]
                        plt.plot(wave, spec)
                        plt.show()
                     
    name = os.path.join(save_folder, 'all_spectrums.png')
    plt.savefig(name)
    plt.close()
    
    return len(spec)

def mean_all_spectrums(direction, save_folder, n):
    
    mean_all = np.zeros(n)
    count = 0
    
    max_londa = []
    
    plt.figure()
    
    for files in os.listdir(direction):
        if files.startswith("Col"):
            new_direction = os.path.join(direction, files)
            for files in os.listdir(new_direction):
                if files.startswith("normalized"):
                    folder = os.path.join(new_direction, files)
                    list_of_NPs = os.listdir(folder)
                    
                    for f in list_of_NPs:
                        file = os.path.join(folder, f)
                        data = np.loadtxt(file)
                        wave = data[:, 0]
                        spec = data[:, 1]
                        
                        count =  count + 1
                        
                        spec = signal.savgol_filter(spec, 29, 0, mode = 'mirror')
                        max_londa.append(wave[np.argmax(spec)])
                        
                        mean_all = spec + mean_all
                        plt.plot(wave, spec)
                        plt.show()
                        
    name = os.path.join(save_folder, 'all_max_wavelength_new_smooth.txt')
    all_data = np.array(max_londa).T
    np.savetxt(name, all_data)
                        
    mean_all = mean_all/count
    #smooth = signal.savgol_filter(mean_all, 21, 0, mode = 'mirror')
    
    plt.plot(wave, mean_all, 'k-')
   # plt.plot(wave, smooth, 'r--')
   
    print(wave[np.argmax(mean_all)])
    
    name = os.path.join(save_folder, 'all_spectrums.png')
    plt.savefig(name)
    
    plt.close()
    
    return

    
if __name__ == '__main__':
    
   # root = Tk()
   # root.withdraw()
   # direction = filedialog.askdirectory()
   # print(direction)
    
    base_folder = r'C:\Users\lupau\OneDrive\Documentos'
    
    daily_folder = r'2022-06-24 Au60 satelites Pd rendija'
    folder_NPs = r'20220624-185257_Scattering_Steps_Load_grid\more_NPs_per_photo'
    
    direction = os.path.join(base_folder, daily_folder, folder_NPs)
    
    save_folder = os.path.join(direction, 'fig_normalized_all_col')  
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    open_data(direction, save_folder)
    plot_data(save_folder, 'all_max_wavelength_polynomial')
    n = plot_all_spectrums(direction, save_folder)
    
    mean_all_spectrums(direction, save_folder, n)
    plot_data(save_folder, 'all_max_wavelength_new_smooth')