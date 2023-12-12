# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 20:34:06 2023

@author: lupau
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')

base_folder = r'C:\Users\lupau\OneDrive\Documentos'

daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220615-124348_Lum_size_ROI_19_in_best_center\processed_data_319nm_hasta600'
parent_folder = os.path.join(base_folder, daily_folder)

folderS = os.path.join(parent_folder,r'Integral Spectrums per NP\Desired_S_nm_543_inf\Desired_AS_nm_500-505')
folderAS = os.path.join(parent_folder,r'Integral Spectrums per NP\Desired_AS_nm_500-520\Desired_S_nm_543-548')

file = r'Integral Spectrums per NP\Desired_S_nm_543_inf\Desired_AS_nm_500-505\ajuste_AS_per_NP_9.txt'
file = os.path.join(parent_folder, file)
data = np.loadtxt(file, skiprows = 1)
lspr = data[:,0]
width = data[:,1]

Qf = lspr/width
NP = 10

Qy_S = np.zeros(NP)
Qy_AS = np.zeros(NP)

fig, ax0 = plt.subplots()

for i in range(NP):
 #   ax.plot(matrixS[0,:], matrixS[1+i,:], 'o')
 
    fileS = os.path.join(folderS, 'mean_integral_S_per_NP_%s.txt'%i)
    dataS = np.loadtxt(fileS, skiprows = 1)
    irradianceS = dataS[:, 0]
    S = dataS[:, 1]
    
    fileAS = os.path.join(folderAS, 'mean_integral_AS_per_NP_%s.txt'%i)
    
    dataAS = np.loadtxt(fileAS, skiprows = 1)
    irradianceAS = dataAS[:, 0]
    AS = dataAS[:, 1]

    ax0.plot(irradianceS, S, 'o', color = 'C%s'%i)
    ax0.plot(irradianceAS, AS, '*', color = 'C%s'%i)
    
    Qy_S[i] = np.mean(S[1:]/irradianceS[1:])
    Qy_AS[i] = np.mean(AS[1:]/irradianceAS[1:])

fig, (ax, ax1) = plt.subplots(2, 1)
ax.plot(lspr, Qy_S, 'ro')
ax.plot(lspr, Qy_AS, 'bo')
ax1.plot(lspr, Qf, 'go')