# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:16:39 2022

@author: lupau
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sig
from Correct_Step_and_Glue import glue_steps

base_folder = r'C:\Users\lupau\OneDrive\Documentos'
daily_folder = r'2022-07-08 Nanostars P2R20 impresion\scattering'

folder_lamp = os.path.join(base_folder, daily_folder, 'lampara_6s')

file_lamp = os.path.join(folder_lamp, 'lamparaIR_grade_2.txt')
data = np.loadtxt(file_lamp, comments = '#')
wave_lamp = data[:, 0]
spec_lamp = data[:, 1]

folder_NPs = r'G4\photos'

parent_folder = os.path.join(base_folder, daily_folder, folder_NPs)

#%%

cols = np.arange(1, 11, 1)

totalNP = 10
large_data = 861

for col in cols:
    
    column = 'Col_%03d'%col
    folder = os.path.join(parent_folder, column)
        
    savefolder = os.path.join(folder, 'fig_spectrums')
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    
    list_of_folders = os.listdir(folder)
    list_of_bkg =  [f for f in list_of_folders if re.search('background',f)][0]
    list_of_NPs =  [f for f in list_of_folders if re.search('NP_',f)]
    
    print(list_of_bkg, list_of_NPs)
    
    data = np.loadtxt(os.path.join(folder,list_of_bkg ))
        
    wavelength = data[:,0]
    signal_bkg = data[:,1]
    grade = 2  #correct_step_and_glue grado de weights del glue
    
    wavelength_gs, signal_bkg_gs = glue_steps(wavelength, signal_bkg, number_pixel = 1002, grade = grade, plot_all_step = False)
         
    matrix = np.zeros((large_data, totalNP +1))  
    
    plt.figure()
    
    for f in list_of_NPs:
        
        NP = f.split('NP_')[1]
        NP = int(NP.split('.txt')[0])
        
        data = np.loadtxt(os.path.join(folder, f))
        signal = data[:,1]
    
        wavelength_gs, signal_gs = glue_steps(wavelength, signal, number_pixel = 1002, grade = grade, plot_all_step = False)
       
        desired = np.where((wavelength_gs >= 450) & (wavelength_gs <= 950))
        londa = wavelength_gs[desired]
        
        spectrum = signal_gs[desired] 
        spectrum_bkg = signal_bkg_gs[desired]
        lamp = spec_lamp[desired]
        
        s = spectrum - spectrum_bkg
        s = s/lamp
        
       # plt.plot(londa, spectrum-spectrum_bkg)
       # plt.plot(londa, spectrum_bkg)
       # plt.plot(londa, lamp*factor)
       # plt.plot(londa, s)
        
        desired_norm = np.where((londa >= 480) & (londa <= 850))
        s2 = s[desired_norm]
        s_norm = (s2 - min(s2))/ (max(s2)-min(s2))
        londa_norm = londa[desired_norm]
        
        print(len(londa_norm))
        
        matrix[:, 0] = londa_norm
        matrix[:, NP+1] = s2
        
        name = os.path.join(savefolder,'NP_%03d.txt'%NP)
        data = np.array([londa_norm, s2, s_norm]).T
        np.savetxt(name, data)#, header = header_text)
        
        plt.plot(londa_norm, s2)#_norm)
        plt.show()
        
    name = os.path.join(savefolder, 'fig_spectrums.png')
    plt.savefig(name)
    plt.close()
    
    name = os.path.join(savefolder, 'matrix_%s.txt'%column)
    np.savetxt(name, matrix)

#%%
n = 0

plt.figure()

for col in cols:
    
    name_col = 'Col_%03d'%col
    print(name_col)
    
    folder_col = os.path.join(parent_folder, name_col, 'fig_spectrums')
    
    list_files = os.listdir(folder_col)
    list_files =  [f for f in list_files if re.search('NP',f)]
    list_files.sort()
    
    for NP_file in list_files:
        
        print(NP_file)
        
        file = os.path.join(folder_col, NP_file)
        
        data = np.loadtxt(file)
        
        londa = data[:, 0]
        spec = data[:, 2]
        
        plt.plot(londa, spec)
        
        n = n + 1
        
plt.show()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Scattering')
name = os.path.join(parent_folder, 'fig_all_spectrums.png')
plt.savefig(name)
plt.close()
print(n)
    
    
    
    
