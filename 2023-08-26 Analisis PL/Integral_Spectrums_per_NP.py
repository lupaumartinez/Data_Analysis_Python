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

base_folder = r'C:\Users\lupau\OneDrive\Documentos'

#daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220615-124348_Luminescence_10x5_3.0umx0.0um_size_ROI_19_in_best_center'
#exceptions_NP = []
#parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_319nm_hasta600')
#np_col = 10

#daily_folder = r'2022-06-14 AuNP 67 PM 460 HP\20220614-165226_Luminescence_10x10_3.0umx0.0um'
#exceptions_NP = []
#parent_folder = os.path.join(base_folder, daily_folder, 'processed_data_326nm_hasta600')
#np_col = 10

#daily_folder = r'2022-06-15 AuNP 60 control satelites Pd\20220615-182446_Luminescence_Load_grid_size_ROI_19_in_best_center\processed_data_328nm'
#parent_folder = os.path.join(base_folder, daily_folder)
#exceptions_NP = []
#np_col = 10

#daily_folder = r'2022-05-11 Au-Paladio\2022-05-12 (Au NP 60 nm control)\20220513-155112_Luminescence_Load_grid\processed_data'
#parent_folder = os.path.join(base_folder, daily_folder)
#exceptions_NP = []
#np_col = 10

#daily_folder = r'2022-05-11 Au-Paladio\2022-05-11 (Au NP 60 nm satelites paladio)\20220511-125259_Luminescence_Load_grid\processed_data'
#parent_folder = os.path.join(base_folder, daily_folder)
#exceptions_NP = ['Col_001_NP_041']#'Col_001_NP_016', 'Col_001_NP_055', 'Col_001_NP_065']
#np_col = 8

#size_roi = 19

##Nanostars R20 532 nm

#daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-161433_Luminescence_Load_grid_in_best_center_size_13\processed_data'
#parent_folder = os.path.join(base_folder, daily_folder)
#exceptions_NP = ['Col_001_NP_009']
#np_col = 9
#size_roi = 13

##Nanostars R20 594 nm

daily_folder = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-174344_Luminescence_Load_grid_in_best_center_size_13\processed_data'
parent_folder = os.path.join(base_folder, daily_folder)
exceptions_NP = []
np_col = 10
size_roi = 13

save_folder0 = os.path.join(parent_folder, 'Integral Spectrums per NP')
if not os.path.exists(save_folder0):
    os.makedirs(save_folder0)

list_of_folders = os.listdir(parent_folder)
list_of_folders_NP = [f for f in list_of_folders if re.search('Col',f)]

print(len(list_of_folders_NP)-len(exceptions_NP))

NP = []
Col = []

color = []
for e in sns.color_palette():
    color.append(e)

file = os.path.join(parent_folder, list_of_folders_NP[0], 'pl_in_bins', 'londa_%s.dat'%list_of_folders_NP[0])
londa = np.loadtxt(file)
N = len(londa)

file = os.path.join(parent_folder, list_of_folders_NP[0], 'pl_in_bins', 'all_bins_%s.dat'%list_of_folders_NP[0])
matrix_specs_bins = np.loadtxt(file)
totalbins = matrix_specs_bins.shape[1]

# file = os.path.join(folder, 'pl_in_bins', 'londa_%s.dat'%NP_folder)

#dS = np.arange(543, 601, 5)
#dAS = np.arange(500, 525, 5)

dS = np.arange(606, 660, 5)
dAS = np.arange(562, 587, 5)

S_bool = False

if not S_bool:

    save_folder1 = os.path.join(save_folder0, 'Desired_AS_nm_%s-%s'%(dAS[0], dAS[-1]))
    if not os.path.exists(save_folder1):
        os.makedirs(save_folder1)
        
    desired_antistokes = np.where((londa >= dAS[0]) & (londa <= dAS[-1]))
    
    n = len(dS)-1
    
else:
    
    save_folder1 = os.path.join(save_folder0, 'Desired_S_nm_%s-%s'%(dS[0], dS[-1]))
    if not os.path.exists(save_folder1):
        os.makedirs(save_folder1)

    desired_stokes = np.where((londa >= dS[0]) & (londa <= dS[-1]))
    
    n = len(dAS)-1

for i in range(n):
    
    if not S_bool:
        desired_stokes = np.where((londa >= dS[i]) & (londa <= dS[i+1]))
    
        save_folder = os.path.join(save_folder1, 'Desired_S_nm_%s-%s'%(dS[i], dS[i+1]))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
    else:
        desired_antistokes = np.where((londa >= dAS[i]) & (londa <= dAS[i+1]))
    
        save_folder = os.path.join(save_folder1, 'Desired_AS_nm_%s-%s'%(dAS[i], dAS[i+1]))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
    stokes_int_per_NP = np.zeros((np_col, totalbins))
    antistokes_int_per_NP = np.zeros((np_col, totalbins))
    i_per_NP = np.zeros((np_col, totalbins))
    count = np.zeros(np_col)
    LSPR = np.zeros(np_col)
    WIDTH = np.zeros(np_col)
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    for NP_folder in list_of_folders_NP:
    
        if NP_folder in exceptions_NP:
            print('salteo', NP_folder)
            continue
        
        name_col = NP_folder.split('_NP')[0]
        name_col = int(name_col.split('Col_')[1])
         
        name_NP = int(NP_folder.split('NP_')[1])
        
        if name_NP >= np_col:
            name_col = int(name_NP/np_col) + 1
            print(name_NP, name_col)
        
        NP_unique = int(name_NP-np_col*(name_col-1))
        
        Col.append(name_col)
        NP.append(name_NP)
       
        folder = os.path.join(parent_folder, NP_folder)
        
        londa_stokes = londa[desired_stokes]
        londa_antistokes = londa[desired_antistokes]
        
        file = os.path.join(folder, 'pl_in_bins', 'all_bins_%s.dat'%NP_folder)
        matrix_specs_bins = np.loadtxt(file)
        totalbins = matrix_specs_bins.shape[1]
        
        file = os.path.join(folder, 'pl_in_bins', 'bin_irradiance_%s.dat'%NP_folder)
        data = np.loadtxt(file)
        irradiance = data[:, 0]
        
        file = os.path.join(folder, 'spr', 'spr_fitted_parameters_%s.dat'%NP_folder)
        data = np.loadtxt(file)
        lspr = data[0]
        width = data[1]
        
       # matrix_stokes = np.zeros((len(londa_stokes), totalbins))
       # matrix_antistokes = np.zeros((len(londa_antistokes), totalbins))
        
        stokes_int = np.zeros(totalbins)
        antistokes_int = np.zeros(totalbins)
        bines = np.arange(0, totalbins, 1)
        
        sum_bin_stokes = np.zeros(len(desired_stokes))
        sum_bin_antistokes = np.zeros(len(desired_antistokes))
        sum_spec = np.zeros(len(londa))
        
        for i in range(totalbins):
            
            stokes = matrix_specs_bins[desired_stokes, i]/size_roi
            antistokes = matrix_specs_bins[desired_antistokes, i]/size_roi
            
            stokes_int[i] = np.sum(stokes)
            antistokes_int[i] = np.sum(antistokes)
            
        if np.where(NP_unique == range(np_col)):
            count[NP_unique] = count[NP_unique] + 1
            stokes_int_per_NP[NP_unique, :] = stokes_int_per_NP[NP_unique, :] + stokes_int
            antistokes_int_per_NP[NP_unique, :] = antistokes_int_per_NP[NP_unique, :] + antistokes_int
            i_per_NP[NP_unique, :] = i_per_NP[NP_unique, :] + irradiance
            
            LSPR[NP_unique] = LSPR[NP_unique] + lspr
            WIDTH[NP_unique] = WIDTH[NP_unique] + width
            
        ax1.plot(irradiance, stokes_int, 'o', color = color[NP_unique])
        ax2.plot(irradiance, antistokes_int, '*', color = color[NP_unique])
        
        fig1.savefig(os.path.join(save_folder, 'integral_Stokes_per_NP.png'))
        fig2.savefig(os.path.join(save_folder, 'integral_AntiStokes_per_NP.png'))
  
    lspr = LSPR/count
    width = WIDTH/count
    
    print('lspr', lspr, 'width', width)
    
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
    
    pAS_list = np.zeros(np_col)
    pS_list = np.zeros(np_col)
    
    for i in range(np_col):
        
        I = i_per_NP[i, :]/count[i]
        S = stokes_int_per_NP[i, :]/count[i]
        AS = antistokes_int_per_NP[i, :]/count[i]
        
        logI = np.log(I[1:])
        logS = np.log(S[1:])
        logAS = np.log(AS[1:])
        
        pS = [0,0]
        yS = np.zeros(len(logI))
        
                
        pAS = [0,0]
        yAS = np.zeros(len(logI))
        
        try:
            pS = np.polyfit(logI, logS, 1)
            yS = np.polyval(pS, logI)
        
        except: pass
            
        try:
            pAS = np.polyfit(logI, logAS, 1)
            yAS = np.polyval(pAS, logI)
            
        except: pass
        
        pS_list[i] = round(pS[0],3)
        
        pAS_list[i] = round(pAS[0],3)
        
        print('NP', i, 'ajuste S', pS, 'ajuste AS', pAS)
        
        data = np.array([I, AS]).T
        np.savetxt(os.path.join(save_folder, 'mean_integral_AS_per_NP_%d.txt'%i), data, header = 'irradiance, Int. AS')
        
        data = np.array([I, S]).T
        np.savetxt(os.path.join(save_folder, 'mean_integral_S_per_NP_%d.txt'%i), data, header = 'irradiance, Int. S')
        
        ax3.plot(I, S, 'o')
        ax4.plot(I, AS, '*')
        
        ax5.plot(logI, logS, 'o')
        ax5.plot(logI, yS, 'r--')
        ax6.plot(logI, logAS, '*')
        ax6.plot(logI, yAS, 'r--')
        
    fig3.savefig(os.path.join(save_folder, 'mean_integral_Stokes_per_NP.png'))
    fig4.savefig(os.path.join(save_folder, 'mean_integral_AntiStokes_per_NP.png'))
    
    fig5.savefig(os.path.join(save_folder, 'ajuste_Stokes_per_NP.png'))
    fig6.savefig(os.path.join(save_folder, 'ajuste_AntiStokes_per_NP.png'))
    
    data = np.array([lspr, width, pS_list]).T
    np.savetxt(os.path.join(save_folder, 'ajuste_S_per_NP.txt'), data, header = 'lspr (nm), width (nm), power law')
        
    data = np.array([lspr, width, pAS_list]).T
    np.savetxt(os.path.join(save_folder, 'ajuste_AS_per_NP.txt'), data, header = 'lspr (nm), width (nm), power law')
    
    #fig7, ax7 = plt.subplots()
    #ax7.plot(lspr, pS_list, 'o')
    #ax7.plot(lspr/width, pAS_list, '*')
    #fig7.savefig(os.path.join(save_folder, 'ajuste_S_vs_lspr.png'))
    
    #fig8, ax8 = plt.subplots()
    #ax8.plot(lspr, pAS_list, '*')
    #ax7.plot(lspr/width, pAS_list, '*')
    #fig8.savefig(os.path.join(save_folder, 'ajuste_AS_vs_lspr.png'))

plt.close('all')   
    
#%%

h = 4.135667516e-15  # in eV*s
c = 299792458  # in m/s

def lambda_to_energy(londa):
    # Energy in eV and wavelength in nm
    hc = 1239.84193  # Plank's constant times speed of light in eV*nm
    energy = hc/londa
    return energy

#dS = np.arange(543, 601, 5)
#dAS = np.arange(500, 525, 5)
#londaS = (dS[1:] + dS[0:-1])/2
#londaAS = (dAS[1:] + dAS[0:-1])/2

folder = os.path.join(save_folder0, 'Desired_S_nm_%s-%s'%(dS[0], dS[-1]))
#folder2 = os.path.join(save_folder0, 'Desired_AS_nm_500-520')

list_of_folders = os.listdir(folder)

wexc = 592

matrixAS = np.zeros(((len(pAS_list)+1),len(list_of_folders)))

for i in range(len(list_of_folders)):
    
    f = list_of_folders[i]
    
    a = f.split('nm_')[-1]
    w = (float(a.split('-')[-1])+ float(a.split('-')[0]))/2
    
    energy_shift = lambda_to_energy(w)-lambda_to_energy(wexc)
    
    matrixAS[0,i] = energy_shift
       
    fol = os.path.join(folder, f, 'ajuste_AS_per_NP.txt')
    
    data = np.loadtxt(fol, skiprows = 1)
    
  #  londa = data[:,0]
  #  width = data[:,1]
    pAS = data[:,2]
    
    for j in range(len(pAS)):
    
        matrixAS[1+j,i] = pAS[j]
    
   # data = np.array([energy_shift_list, pAS[i, :]])

folder = os.path.join(save_folder0, 'Desired_AS_nm_%s-%s'%(dAS[0], dAS[-1]))
list_of_folders = os.listdir(folder)

wexc = 532

matrixS = np.zeros(((len(pS_list)+1),len(list_of_folders)))

for i in range(len(list_of_folders)):
    
    f = list_of_folders[i]
    
    a = f.split('nm_')[-1]
    w = (float(a.split('-')[-1])+ float(a.split('-')[0]))/2
    
    energy_shift = lambda_to_energy(w)-lambda_to_energy(wexc)
    
    matrixS[0,i] = energy_shift
       
    fol = os.path.join(folder, f, 'ajuste_S_per_NP.txt')
    
    data = np.loadtxt(fol, skiprows = 1)
    
  #  londa = data[:,0]
  #  width = data[:,1]
    pS = data[:,2]
    
    for j in range(len(pAS)):
    
        matrixS[1+j,i] = pS[j]
    
   # data = np.array([energy_shift_list, pAS[i, :]])
    

NP = matrixAS.shape[0] - 1
W = matrixAS.shape[1] + matrixS.shape[1]
matrix = np.zeros((NP + 1, W))

for i in range(matrixAS.shape[0]):

    matrix[i, :matrixAS.shape[1]] = matrixAS[i,:]
    matrix[i, matrixAS.shape[1]:W] = matrixS[i,:]
 
fig, ax = plt.subplots()

for i in range(NP):
 #   ax.plot(matrixS[0,:], matrixS[1+i,:], 'o')

    ax.plot(matrixS[0,:], matrixS[1+i,:], '-o', color = 'C%s'%i)
    ax.plot(matrixAS[0,:], matrixAS[1+i,:], '-*', color = 'C%s'%i)
    
#ax.set_xlim(-0.3,0.2)
#ax.set_xticks(np.arange(-0.3, 0.3, 0.1))
#ax.set_yticks(np.arange(0.5, 2.5, 0.5))
#ax.set_ylim(0.5, 2.0)

np.savetxt(os.path.join(save_folder0, 'matrixAS_power_law.txt'), matrixAS.T, header = 'energy shift, power-law')
np.savetxt(os.path.join(save_folder0, 'matrixS_power_law.txt'), matrixS.T, header = 'energy shift, power-law')
np.savetxt(os.path.join(save_folder0, 'matrix_power_law.txt'), matrix.T, header = 'energy shift, power-law')

fig.savefig(os.path.join(save_folder0, 'matrix_power_law.png'))