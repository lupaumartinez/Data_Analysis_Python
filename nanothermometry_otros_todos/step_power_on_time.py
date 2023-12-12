# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:25:17 2022

@author: Luciana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use('for_confocal.mplstyle')
except:
    print('Pre-defined matplotlib style was not loaded.')

plt.ioff()
plt.close('all')

def calibration(voltage, poly):
    
    power = np.polyval(poly, voltage)
    
    return power

def process_control_power(parent_folder, path_to, NP, poly, save_each_plot):
    
    
    folder = os.path.join(parent_folder, f)
    file_trace_BS = os.path.join(folder, 'Trace_BS.txt')
    
    data = np.loadtxt(file_trace_BS)
    time = data[:, 0][10:-10]
    voltage = data[:, 1][10:-10]
    
    time = np.array(time)
    power = calibration(voltage, poly)
    
    if save_each_plot:
    
        plt.figure()
        plt.title('%s'%NP)
        plt.plot(time, power)
        plt.ylim(0.6, 1)
        plt.xlabel('Time (s)')
        plt.ylabel('Power BFP (mW)')
        nameplot = os.path.join(path_to, 'Power_%s'%NP)
        plt.savefig(nameplot)
        plt.close()
    
    return np.mean(power), np.std(power), time[-1]
       
if __name__ == '__main__':
    
    base_folder =  r'C:\Users\lupau\OneDrive\Documentos'#r'C:\Ubuntu_archivos\Printing'
    
    daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-174344_Luminescence_Load_grid_in_best_center_size_13'
  
    poly = 2.28, 0.001 #3.12, -0.009
    
    save_each_plot = False
        
    parent_folder = os.path.join(base_folder, daily_folder)
    
    path_to = os.path.join(parent_folder,'processed_data_power_time')
    if not os.path.exists(path_to):
        os.makedirs(path_to)
            
    list_of_folders = os.listdir(parent_folder)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
    list_of_folders_slow = [f for f in list_of_folders if re.search('Slow_Confocal_Spectrum',f)]
    list_of_folders_slow.sort()
    
    list_time = []
    list_mean_power = []
    list_std_power = []
    
    for f in list_of_folders_slow[:]:
    
        NP = f.split('Spectrum_')[-1]
        NP = NP.split('.txt')[0]
        
        number_NP = NP.split('NP_')[-1]
        number_NP = int(number_NP)
        
        print('Plot trace power on time:', NP)
        
        mean, std, time_confocal = process_control_power(parent_folder, path_to, NP, poly, save_each_plot)
        
        list_mean_power.append(mean)
        list_std_power.append(std)
        list_time.append(number_NP*time_confocal)
        
    list_time = np.array(list_time)
    list_mean_power = np.array(list_mean_power)
    list_std_power = np.array(list_std_power)
    
    list_time  = list_time /3600
    
    plt.figure()
    plt.errorbar(list_time , list_mean_power, yerr = list_std_power, fmt = 'o')
    #     plt.plot(time, voltage0, 'k')
    #plt.ylim(0.20, 0.35)
    plt.xlabel('Time (hour)')
    plt.ylabel('Power BFP (mW)')
    nameplot = os.path.join(path_to, 'mean_Power_time.png')
    plt.savefig(nameplot)
    plt.close()
    
    
    header = 'time (hour), mean power BS (mW), std (mW)'
    data = np.array([list_time, list_mean_power, list_std_power]).T
    name =  os.path.join(path_to,'mean_Power_time.txt')
    np.savetxt(name, data, header = header)