# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:51:34 2022

@author: Luciana
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def irradiance(power, omega):
    
    factor = 0.47
    
    I = 2*factor*power/(np.pi*(omega/1000)**2)
    
    return I

base_folder = r'C:\Ubuntu_archivos\Printing'

list_folders = []
list_folders_name = []
list_folders_photothermal = []
list_folders_power = []

name = 'Au 67 impresas Juli'

daily_folder = r'2022-05-17 Au-Paladio\Test\processed_data_luciana_version_0.71'
betha = 63
power = 0.71
omega = 342
I = irradiance(power, omega)
list_folders_power.append(I)
list_folders.append(daily_folder)
list_folders_name.append(name)
list_folders_photothermal.append(betha)

name = 'Au 67 impresas'

daily_folder = r'2022-05-26 AuNP 67 impresas\20220526-182536_Luminescence_10x10_3.0umx0.0um\processed_data_p'
betha = 87
power = 0.74
omega = 342
I = irradiance(power, omega)
list_folders_power.append(I)
list_folders.append(daily_folder)
list_folders_name.append(name)
list_folders_photothermal.append(betha)
daily_folder = r'2022-05-26 AuNP 67 impresas\20220526-164518_Luminescence_10x10_3.0umx0.0um\processed_data_p'
betha = 83
power = 0.77 #0.75
omega = 342
I = irradiance(power, omega)
list_folders_power.append(I)
list_folders.append(daily_folder)
list_folders_name.append(name)
list_folders_photothermal.append(betha)
#daily_folder = r'2022-05-24 AuNP 67 impresas\20220524-173839_Luminescence_10x1_3.0umx3.0um\processed_data_BS_2.17, 0.033'
#betha = 71.6
#power = 0.72
#omega = 342
#I = irradiance(power, omega)
#list_folders_power.append(I)
#list_folders.append(daily_folder)
#list_folders_name.append(name)
#list_folders_photothermal.append(betha)

name = 'Au 60 impresas'

daily_folder = r'2022-05-12 (Au NP 60 nm control satelites paladio)\20220513-155112_Luminescence_Load_grid\processed_data'
betha = 89
list_folders.append(daily_folder)
list_folders_name.append(name)
list_folders_photothermal.append(betha)
power = 0.66
omega = 342
I = irradiance(power, omega)
list_folders_power.append(I)

name = 'Au 60'

daily_folder = r'2022-05-12 (Au NP 60 nm control satelites paladio)\20220513-105850_Luminescence_Load_grid\processed_data'
betha = 85
power = 0.67
omega = 342
I = irradiance(power, omega)
list_folders_power.append(I)
list_folders.append(daily_folder)
list_folders_name.append(name)
list_folders_photothermal.append(betha)
daily_folder = r'2022-05-12 (Au NP 60 nm control satelites paladio)\20220512-184652_Luminescence_Load_grid\processed_data'
betha = 81
power = 0.67
omega = 342
I = irradiance(power, omega)
list_folders_power.append(I)
list_folders.append(daily_folder)
list_folders_name.append(name)
list_folders_photothermal.append(betha)

name = 'Au 60 - satelites Pd'

#daily_folder = r'2022-05-11 (Au NP 60 nm satelites paladio)\20220511-125259_Luminescence_Load_grid\processed_data'
#betha = 77
#power = 0.76
#omega = 342
#I = irradiance(power, omega)
#list_folders_power.append(I)
#list_folders.append(daily_folder)
#list_folders_name.append(name)
#list_folders_photothermal.append(betha)

daily_folder = r'2022-05-11 (Au NP 60 nm satelites paladio)\20220510-171340_Luminescence_Load_grid\processed_data'
betha = 84
power = 0.77
omega = 342
I = irradiance(power, omega)
list_folders_power.append(I)
list_folders.append(daily_folder)
list_folders_name.append(name)
list_folders_photothermal.append(betha)

#%%

plt.close('all')


plt.figure()

step = 1/100
x = np.arange(0, 1+step, step)
desired = np.where((x > 0.4)&(x<0.6))

powers = np.zeros(len(list_folders))
bethas = np.zeros(len(list_folders))
delta_T = np.zeros(len(list_folders))
y_sum =  np.zeros(len(list_folders))

for i in range(len(list_folders)):
    
    f = list_folders[i]
    name = list_folders_name[i]
    betha = list_folders_photothermal[i]
    power = list_folders_power[i]
    
    dt =  betha*power
    
    powers[i] = power
    bethas[i] = betha
    delta_T[i] = dt
    
    print(name, delta_T[i])
                    
    parent_folder = os.path.join(base_folder, f)

    file = os.path.join(parent_folder, 'Pruebas Señal Anti y Stokes', 'poly2_stokes_antistokes.dat')
    
    poly = np.loadtxt(file)
    
    yf = np.polyval(poly, x)
    
    ysum = np.mean(yf[desired])
    
    y_sum[i] = ysum
    
    plt.plot(dt, ysum, 'o', label = '%s'%(name))
    
   # plt.plot(x, yf, '--', label = '%s_betha_power_deltaT = %2d, %.2f, %.2f'%(name,betha, power, delta_T[i]))

plt.legend()   
plt.xlabel('Delta Temperatura')
plt.ylabel('Y sum') 
plt.show()

#%%

plt.figure()

for i in range(len(list_folders)):
    
    f = list_folders[i]
    name = list_folders_name[i]
    betha = list_folders_photothermal[i]
    power = list_folders_power[i]
    
    dt =  betha*power
                    
    parent_folder = os.path.join(base_folder, f)

    file = os.path.join(parent_folder, 'Pruebas Señal Anti y Stokes', 'poly2_stokes_antistokes.dat')
    
    poly = np.loadtxt(file)
    
    yf = np.polyval(poly, x)
    
    ysum = np.mean(yf[desired])
    
    plt.plot(betha, ysum, 'o', label = '%s'%(name))
    
   # plt.plot(x, yf, '--', label = '%s_betha_power_deltaT = %2d, %.2f, %.2f'%(name,betha, power, delta_T[i]))

plt.legend()   
plt.xlabel('betha')
plt.ylabel('Y sum') 
plt.show()