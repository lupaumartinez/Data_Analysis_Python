# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:22:25 2022

@author: Luciana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sig
from Correct_Step_and_Glue import glue_steps

if __name__ == '__main__':

    base_folder = r'C:\Ubuntu_archivos\Printing'
    
    daily_folder = r'2022-07-07 Nanostars R20 drop and cast'
    
    folder_NPs = '20220707-150626_Scattering_Steps_Load_grid_final_150s'
    
    folder_lamp = os.path.join(base_folder, '2022-07-06 Nanostars R20 drop and cast', 'lampara_2seg')
    
    file_lamp = os.path.join(folder_lamp, 'lamparaIR_grade_2.txt')
    data = np.loadtxt(file_lamp, comments = '#')
    wave_lamp = data[:, 0]
    spec_lamp = data[:, 1]

    factor = 5000
    
    plt.figure()
    
    plt.plot(wave_lamp, spec_lamp*factor, 'y--')
    
    parent_folder = os.path.join(base_folder, daily_folder, folder_NPs)

    list_of_folders = os.listdir(parent_folder)
    list_of_folders =  [f for f in list_of_folders if re.search('_NP_',f)]
    
    savefolder = os.path.join(parent_folder, 'Col_1000')
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    
    save_folder = savefolder
    
    steps = 6
    sum_bkg = np.zeros(steps*1002)
    count_bkg = 0

    for folder_original in list_of_folders:
    
        NP = folder_original.split('NP_')[1]
        
       # save_folder = os.path.join(savefolder, 'Col_%s'%NP)
       # if not os.path.exists(save_folder):
       #     os.makedirs(save_folder)
     
        folder = os.path.join(parent_folder, folder_original)
         
        list_of_files = os.listdir(folder)
         
        file1 = [f for f in list_of_files if re.search('Calibration',f)][0]
         
        calibration_wavelength = os.path.join(folder, file1)
        wavelength0 = np.loadtxt(calibration_wavelength) 
         
        file2 = [f for f in list_of_files if re.search('Line_Spectrum',f) and not re.search('Background',f)][0]
        line_spectrum = os.path.join(folder, file2)
        specs = np.loadtxt(line_spectrum)
        
        file3 = [f for f in list_of_files if re.search('Background',f)][0]
        line_spectrum_bkg = os.path.join(folder, file3)
        specs_bkg = np.loadtxt(line_spectrum_bkg)
         
        wavelength = wavelength0
        signal = specs
        signal_bkg = specs_bkg
     
        name = os.path.join(save_folder,'NP_%s.txt'%NP)
        data = np.array([wavelength, signal]).T
        np.savetxt(name, data)#, header = header_text)
        
        if max(signal_bkg)>500:
            print('no tomar bkg de NP', NP)
        else:
            sum_bkg = signal_bkg + sum_bkg
            count_bkg = count_bkg + 1
            
           # name = os.path.join(save_folder, 'background_row_%s.txt'%NP)
           #data = np.array([wavelength, signal_bkg]).T
           # np.savetxt(name, data)#, header = header_text)
            plt.plot(wavelength, signal_bkg)
            
        plt.plot(wavelength, signal)
        plt.show()
        
    sum_bkg = sum_bkg/count_bkg
    plt.plot(wavelength, sum_bkg, 'k-')
    name = os.path.join(save_folder, 'background_row_mean.txt')
    data = np.array([wavelength, sum_bkg]).T
    np.savetxt(name, data)#, header = header_text)    
    
    #plt.xlim(500, 650)
  #  plt.ylim(100, 3000)
  
#%%

savefolder = os.path.join(save_folder, 'fig_spectrums')
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

list_of_folders = os.listdir(save_folder)
list_of_bkg =  [f for f in list_of_folders if re.search('background_row_mean',f)][0]
list_of_NPs =  [f for f in list_of_folders if re.search('NP_',f)]

print(list_of_bkg, list_of_NPs)

data = np.loadtxt(os.path.join(save_folder,list_of_bkg ))
    
wavelength = data[:,0]
signal_bkg = data[:,1]
grade = 2  #correct_step_and_glue grado de weights del glue

wavelength_gs, signal_bkg_gs = glue_steps(wavelength, signal_bkg, number_pixel = 1002, grade = grade, plot_all_step = False)
     
plt.figure()

for f in list_of_NPs:
    
    NP = f.split('NP_')[1]
    
    data = np.loadtxt(os.path.join(save_folder, f))
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
    
    plt.plot(londa_norm, s2, label = 'NP_%s'%NP)#_norm)
    plt.legend()
    
    name = os.path.join(savefolder, 'fig_spectrums.png')
    plt.savefig(name)
    
    name = os.path.join(savefolder,'NP_%s.txt'%NP)
    data = np.array([londa_norm, s2, s_norm]).T
    np.savetxt(name, data)#, header = header_text)
    
plt.show()