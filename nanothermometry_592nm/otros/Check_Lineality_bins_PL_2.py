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
from scipy.optimize import curve_fit

def lorentz2(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = 3.141592653589793
    I, gamma, x0, C = p
    return (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def fit_lorentz2(p, x, y):
    return curve_fit(lorentz2, x, y, p0 = p)
#%%

plt.close('all')

base_folder = r'C:\Users\lupau\OneDrive\Documentos'

#Au67
list_p = []

daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220615-124348_Luminescence_10x5_3.0umx0.0um_size_ROI_19_in_best_center'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_319nm_hasta600')
#list_p.append(parent_folder)

daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220614-165226_Luminescence_10x10_3.0umx0.0um'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_326nm_hasta600')
#list_p.append(parent_folder)

daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220614-165226_Luminescence_10x10_3.0umx0.0um'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_326nm')
list_p.append(parent_folder)

daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220615-124348_Luminescence_10x5_3.0umx0.0um_size_ROI_19_in_best_center'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_319nm')
list_p.append(parent_folder)

daily_folder = r'2022-06-15 AuNP 60 control satelites Pd\20220615-182446_Luminescence_Load_grid_size_ROI_19_in_best_center\processed_data_328nm'
parent_folder = os.path.join(base_folder, daily_folder)
list_p.append(parent_folder)

colors = ['k', 'k', 'r']

fig, ax = plt.subplots()
ax.set_ylabel('Londa SPR (nm)')
ax.set_xlabel('Cross (nm)')

ax2 = ax.twinx()
ax2.set_ylabel('Width SPR (nm)')

for i in range(len(colors)):
    
    parent_folder = list_p[i]
    
    color = colors[i]
        
    folder = os.path.join(parent_folder, 'Mean Spectrums per NP')
    
    list_of_files = os.listdir(folder)
    list_of_files = [f for f in list_of_files if re.search('txt',f)]
    list_of_files.sort()
    
    folder2 = os.path.join(parent_folder, 'PL stokes normalized from bins')
    
    list_of_files2 = os.listdir(folder2)
    list_of_files2 = [f for f in list_of_files2 if re.search('txt',f)]
    list_of_files2.sort()
    
    print(len(list_of_files), len(list_of_files2))
    
    #ax2.set_ylim(0,1)
    
    for i in range(len(list_of_files)):
        
        file = os.path.join(folder, list_of_files[i])
        data = np.loadtxt(file, skiprows = 1)
        londa = data[:, 0]
        spec = data[:, 1]
        
        desired = np.where(londa > 543)
        londa_s = londa[desired]
        s = spec[desired]
        s = (s)#/(max(s))
        londa_max = londa_s[np.argmax(s)]
        
        try:        
            init_params = np.array([max(s), 56, londa_max, 0], dtype=np.double)
            # Get the fitting parameters for the best lorentzian
            best_lorentz, err = fit_lorentz2(init_params, londa_s, s)
            # calculate the errors
            lorentz_fitted = lorentz2(s, *best_lorentz)
            full_lorentz_fitted = lorentz2(londa_s, *best_lorentz)
            londa_max_pl = best_lorentz[2]
            width_pl = best_lorentz[1]
            print('SPR wavelenth (max) = %.2f nm, Width = %.2f nm' % (londa_max_pl, width_pl))
        except RuntimeError:
            print('SPR fitting did not converge. Analysis must go on... ignoring NP\'s SPR.')
            full_lorentz_fitted = np.zeros(len(londa))
            londa_max_pl = 0
            width_pl = 0
        
       # ax2.plot(londa, spec, '--', color = color)
        a = full_lorentz_fitted/max(full_lorentz_fitted)
        ax.plot(londa_s, a, '-', color = color)
        
        file = os.path.join(folder2, list_of_files2[i])
        data = np.loadtxt(file, skiprows = 1)
        londa_stokes = data[:, 0]
        norm_spec1 = data[:, 1]
        norm_spec2 = data[:, 2]
        
        b = norm_spec1/max(norm_spec1)
        c = norm_spec2/max(norm_spec2)
        ax2.plot(londa_stokes, b, '--', color = color)
      #  ax2.plot(londa_stokes, c, '-', color = color)
        
        cross = londa_stokes[np.argmin(np.abs(norm_spec2 - norm_spec1))]
        print(cross, londa_max_pl)
        
     #   ax2.plot(cross, width_pl, '*', color = color)
      #  ax.plot(cross, londa_max_pl, 'o', color = color)
        
    plt.show()