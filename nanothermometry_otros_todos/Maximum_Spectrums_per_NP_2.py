# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:45:30 2022

@author: lupau
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

plt.style.use('default')
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams ["axes.labelsize"] = 16
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16

base_folder = r'C:\Users\lupau\OneDrive\Documentos\2022-07-15 Nanostars R20 drop and cast PL'

#%%
#Au60

list_p_60 = []
list_irradiance = []
list_factor = []

T = 0.47

daily_folder = r'20220715-161433_Luminescence_Load_grid_in_best_center_size_13'
parent_folder = os.path.join(base_folder, daily_folder)
list_p_60.append(parent_folder)
Power= 0.78
PSF= 306
I = (2*Power*T/(np.pi * PSF**2))
list_irradiance.append(I)
EM = 80
size_roi = 13
exposure_time = 2.5
factor = size_roi*EM*exposure_time
list_factor.append(factor)


daily_folder = r'20220715-174344_Luminescence_Load_grid_in_best_center_size_13'
parent_folder = os.path.join(base_folder, daily_folder)
list_p_60.append(parent_folder)
Power= 0.71
PSF= 338
I = (2*Power*T/(np.pi * PSF**2))
list_irradiance.append(I)
EM = 80
size_roi = 13
exposure_time = 1.2
factor = size_roi*EM*exposure_time
list_factor.append(factor)

factor_norm = 10**7

colors = ['g', 'orange']

n = 10

plot_each_one = False

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
    factor = list_factor[i]
    
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
            plt.plot(londa, spec, color = color, alpha = 0.3, linewidth=1)
            plt.show()
      
    mean = mean/n
    
    if not plot_each_one:
        mean = mean/max(mean)
       # if color == 'orange':
          #  londa = londa - (610-550)
    
    plt.plot(londa, mean, color = color, alpha = 0.7, linewidth=3)
    plt.show()
    
#plt.xticks(np.arange(500, 620, 20))
#plt.yticks(np.arange(0, 1.2, 0.2))
#plt.ylim(-0.05, 1.10)
#plt.xlim(495, 605)

plt.savefig(os.path.join(base_folder, 'R20_%s.png'%plot_each_one), dpi = 400, bbox_inches='tight')
plt.savefig(os.path.join(base_folder, 'R20_%s.pdf'%plot_each_one), dpi = 400, bbox_inches='tight')