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

list_p = []

base_folder = r'C:\Ubuntu_archivos\Printing'

#Au67

daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220615-124348_Luminescence_10x5_3.0umx0.0um_size_ROI_19_in_best_center'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_319nm')
#list_p.append(parent_folder)
daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220614-165226_Luminescence_10x10_3.0umx0.0um'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_326nm')
list_p.append(parent_folder)

#Au60

daily_folder = r'2022-06-15 AuNP 60 control satelites Pd\20220615-182446_Luminescence_Load_grid_size_ROI_19_in_best_center'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_328nm')
#list_p.append(parent_folder)

daily_folder =  r'2022-05-12 (Au NP 60 nm control satelites paladio)\20220512-184652_Luminescence_Load_grid'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data')
list_p.append(parent_folder)
daily_folder =  r'2022-05-12 (Au NP 60 nm control satelites paladio)\20220513-105850_Luminescence_Load_grid'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data')
list_p.append(parent_folder)

daily_folder =  r'2022-05-12 (Au NP 60 nm control satelites paladio)\20220513-155112_Luminescence_Load_grid'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data')
list_p.append(parent_folder)

#Au60-satelite Pd

daily_folder = r'2022-05-11 (Au NP 60 nm satelites paladio)\20220510-171340_Luminescence_Load_grid'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data')
list_p.append(parent_folder)
daily_folder = r'2022-05-11 (Au NP 60 nm satelites paladio)\20220511-125259_Luminescence_Load_grid'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data')
list_p.append(parent_folder)

daily_folder = r'2022-06-22 Au60 satelites Pd\20220621-185503_Luminescence_Load_grid_in_best_center'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_315nm')
list_p.append(parent_folder)

daily_folder = r'2022-06-22 Au60 satelites Pd\20220622-115036_Luminescence_Load_grid_in_best_center'
parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_298nm')
list_p.append(parent_folder)

colors = ['k', 'r', 'r', 'r', 'g', 'g', 'b', 'b']

plt.figure()

for i in range(len(colors)):
    
    folder = os.path.join(list_p[i], 'Mean Spectrums per NP')
    
    list_of_files = os.listdir(folder)
    list_of_files = [f for f in list_of_files if re.search('txt',f)]
    
    color = colors[i]
    
    for f in list_of_files:
        
        file = os.path.join(folder, f)
        data = np.loadtxt(file, skiprows = 1)
        londa = data[:, 0]
        spec = data[:, 1]
        
        if color == 'b':
            londa = londa - 0.4
        
        plt.plot(londa, spec/max(spec), color = color)
        plt.show()