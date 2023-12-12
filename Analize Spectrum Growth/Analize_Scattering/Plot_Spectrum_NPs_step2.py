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

def open_data(direction, save_folder):
        
    all_data = []
    
    for files in os.listdir(direction):
        if files.startswith("Col"):
            new_direction = os.path.join(direction, files)
            for files in os.listdir(new_direction):
                if files.startswith("fig_normalized"):
                    name_data = os.path.join(new_direction, files, "max_wavelength.txt")
                    data = np.loadtxt(name_data)
                    all_data.extend(data[:,2])
                     
    name = os.path.join(save_folder, 'all_max_wavelength_polynomial.txt')
    all_data = np.array(all_data).T
    np.savetxt(name, all_data)
    
    return


def plot_data(save_folder):
    
    name = os.path.join(save_folder, 'all_max_wavelength_polynomial.txt')
    max_wavelength = np.loadtxt(name)
    
    L = len(max_wavelength)
    print(max_wavelength)
    
    mean = round(np.mean(max_wavelength),2)
    std = round(np.std(max_wavelength),2)
    
    print('valor medio', mean, 'desviasión estándar', std)
    
    plt.hist(max_wavelength, bins=7, normed = True, rwidth=0.9, color='C1', label = 'N = %d'%L)
    plt.xlabel('Max Wavelength (nm)')
    plt.ylabel('Frequency')
    plt.axvspan(mean - std, mean + std, color = 'green', alpha = 0.3)
    plt.axvline(mean, color = 'red', linestyle = '--')
    plt.text(mean + 0.001*mean, 0.008, ' %2d + %2d nm'%(mean, std), color = 'red')
    plt.xlim(mean - 2*std, mean + 2*std)
    #plt.ylim(0, 0.1)
    plt.legend()
    plt.savefig(save_folder + '/all_hist_Max_Poly_wavelength_NP_Spectrums.png', dpi = 500)
    plt.show()

    return

    
if __name__ == '__main__':
    
    root = Tk()
    root.withdraw()
    direction = filedialog.askdirectory()
    print(direction)
    
    save_folder = os.path.join(direction, 'fig_normalized_all_col')  
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    open_data(direction, save_folder)
    plot_data(save_folder)
