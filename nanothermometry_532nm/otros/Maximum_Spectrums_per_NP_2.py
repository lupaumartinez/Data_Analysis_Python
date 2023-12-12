# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:45:30 2022

@author: lupau
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.close('all')

plt.style.use('default')
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams ["axes.labelsize"] = 16
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16

base_folder = r'C:\Users\lupau\OneDrive\Documentos\2022-06-30 PL de Au y shells Pd'

#%%
#Au60

list_p_60 = []
list_irradiance = []

exposure_time = 2.5
EM = 150
size_roi = 19
factor = size_roi*EM*exposure_time

T = 0.47

daily_folder = r'Au60\20220615-182446_Luminescence_Load_grid_in_best_center'
parent_folder = os.path.join(base_folder, daily_folder)
list_p_60.append(parent_folder)
Power= 0.72
PSF= 328
I = (2*Power*T/(np.pi * PSF**2))
list_irradiance.append(I)


daily_folder = r'Au60@satelitesPd\20220622-140802_Luminescence_Load_grid_in_best_center'
parent_folder = os.path.join(base_folder, daily_folder)
list_p_60.append(parent_folder)
Power= 0.72
PSF= 298
I = (2*Power*T/(np.pi * PSF**2))
list_irradiance.append(I)

daily_folder = r'Au60Pd2nm\20210622-164652_Aupb4-Luminescence_5x10_l8_1,6mW'
parent_folder = os.path.join(base_folder, daily_folder)
list_p_60.append(parent_folder)
Power=0.6
PSF=331
I = (2*Power*T/(np.pi * PSF**2))
list_irradiance.append(I)

factor_norm = 0.7*10**7

colors = ['g', 'b', 'r']

n = 20

plot_each_one = True

plt.figure(figsize=(5,4))
plt.xlabel('Wavelength (nm)')
plt.ylabel('PL (a.u.)')

for i in range(len(colors)):
    
    folder = os.path.join(list_p_60[i], 'pl_maximum')
    
    list_of_files = os.listdir(folder)
    list_of_files = [f for f in list_of_files if re.search('txt',f)]
    list_of_files.sort()
    
    color = colors[i]
    irradiance = list_irradiance[i]
    
    mean = np.zeros(1002)
    
    for f in list_of_files[:n]:
        
        file = os.path.join(folder, f)
        data = np.loadtxt(file, skiprows = 1)
        londa = data[:, 0]
        spec = data[:, 1]
        
        spec = spec/factor
        
        spec = spec/irradiance
        
        spec = spec/factor_norm
        
        mean = mean + spec
        
        if plot_each_one:
            plt.plot(londa, spec, color = color, alpha = 0.2, linewidth=1)
            plt.show()
      
    mean = mean/n
    
    if color == 'g':
        max_mean = max(mean)
    
    if not plot_each_one:
        mean = mean/max_mean
        
    plt.plot(londa, mean, color = color, alpha = 0.6, linewidth=3)
    plt.show()
    
plt.xticks(np.arange(500, 620, 20))
plt.yticks(np.arange(0, 1.2, 0.2))
plt.ylim(-0.05, 1.10)
plt.xlim(495, 605)

plt.savefig(os.path.join(base_folder, 'Au60_%s.png'%plot_each_one), dpi = 400, bbox_inches='tight')
plt.savefig(os.path.join(base_folder, 'Au60_%s.pdf'%plot_each_one), dpi = 400, bbox_inches='tight')


#%%

#Au67

list_p_67 = []
list_irradiance = []

exposure_time = 2.5
EM = 150
size_roi = 19
factor = size_roi*EM*exposure_time

T = 0.47

daily_folder =  r'Au67\20220614-165226_Luminescence_10x10_3.0umx0.0um'
parent_folder = os.path.join(base_folder, daily_folder)
Power=0.74
PSF=326
I = (2*Power*T/(np.pi * PSF**2))
list_irradiance.append(I)
list_p_67.append(parent_folder)

daily_folder =  r'Au67Pd2nm\20211124-112922_Luminescence_10x10'
parent_folder = os.path.join(base_folder, daily_folder)
Power=0.8
PSF=290
I = (2*Power*T/(np.pi * PSF**2))
list_irradiance.append(I)
list_p_67.append(parent_folder)

daily_folder =  r'Au67Pd3.5nm\20211112-131317_Luminescence_5x5_Au67Pd6_linea1'
parent_folder = os.path.join(base_folder, daily_folder)
Power=0.78 
PSF=294
I = (2*Power*T/(np.pi * PSF**2))
list_irradiance.append(I)
list_p_67.append(parent_folder)

colors = ['g', 'b', 'r']

factor_norm = 10**7

n = 20

plot_each_one = True

plt.figure(figsize=(5,4))
plt.xlabel('Wavelength (nm)')
plt.ylabel('PL (a.u.)')

for i in range(len(colors)):
    
    folder = os.path.join(list_p_67[i], 'pl_maximum')
    
    list_of_files = os.listdir(folder)
    list_of_files = [f for f in list_of_files if re.search('txt',f)]
    list_of_files.sort()
    
    print(len(list_of_files))
    
    color = colors[i]
    irradiance = list_irradiance[i]
    
    print('N', len(list_of_files[:15]))
    
    mean = np.zeros(1002)
    
    for f in list_of_files[:n]:
        
        file = os.path.join(folder, f)
        data = np.loadtxt(file, skiprows = 1)
        londa = data[:, 0]
        spec = data[:, 1]
        
        spec = spec/factor
        
        spec = spec/irradiance
        
        spec = spec/factor_norm
        
        mean = mean + spec
        
        if plot_each_one:
            plt.plot(londa, spec, color = color, alpha = 0.2, linewidth=1)
            plt.show()
            
    mean = mean/n
    
    if color == 'g':
        max_mean = max(mean)
        print(max_mean)
    
    if not plot_each_one:
        mean = mean/max_mean
        
    plt.plot(londa, mean, color = color, alpha = 0.6, linewidth=3)
    plt.show()

plt.xticks(np.arange(500, 620, 20))
plt.yticks(np.arange(0, 1.2, 0.2))
plt.ylim(-0.05, 1.10)
plt.xlim(495, 605)

plt.savefig(os.path.join(base_folder, 'Au67_%s.png'%plot_each_one), dpi = 400, bbox_inches='tight')
plt.savefig(os.path.join(base_folder, 'Au67_%s.pdf'%plot_each_one), dpi = 400, bbox_inches='tight')