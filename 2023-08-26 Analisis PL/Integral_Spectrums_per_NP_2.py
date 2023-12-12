# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 20:17:39 2023

@author: lupau
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')

base_folder = r'C:\Users\lupau\OneDrive\Documentos'

##Nanostars R20 532 nm
daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-161433_Luminescence_Load_grid_in_best_center_size_13\processed_data'
parent_folder = os.path.join(base_folder, daily_folder,)

file = os.path.join(parent_folder,'Integral Spectrums per NP\matrixS_power_law.txt')
matrixS = np.loadtxt(file, skiprows = 1).T
file = os.path.join(parent_folder,'Integral Spectrums per NP\matrixAS_power_law.txt')
matrixAS = np.loadtxt(file, skiprows = 1).T

fig, ax = plt.subplots()

for i in range(matrixS.shape[0]-1):
 #   ax.plot(matrixS[0,:], matrixS[1+i,:], 'o')

    ax.plot(matrixS[0,:], matrixS[1+i,:], '-o', color = 'g')
    ax.plot(matrixAS[0,:], matrixAS[1+i,:], '-*', color = 'g')
    
n = i
    
##Nanostars R20 532 nm
daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-161433_Luminescence_Load_grid_in_best_center_size_13\processed_data'
parent_folder = os.path.join(base_folder, daily_folder,)

##Nanostars R20 594 nm

daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-174344_Luminescence_Load_grid_in_best_center_size_13\processed_data'
parent_folder = os.path.join(base_folder, daily_folder)

file = os.path.join(parent_folder,'Integral Spectrums per NP\matrixS_power_law.txt')
matrixS = np.loadtxt(file, skiprows = 1).T
file = os.path.join(parent_folder,'Integral Spectrums per NP\matrixAS_power_law.txt')
matrixAS = np.loadtxt(file, skiprows = 1).T


for i in range(matrixS.shape[0]-1):
 #   ax.plot(matrixS[0,:], matrixS[1+i,:], 'o')

    ax.plot(matrixS[0,:], matrixS[1+i,:], '-o', color = 'orange')
    ax.plot(matrixAS[0,:], matrixAS[1+i,:], '-*', color = 'orange')