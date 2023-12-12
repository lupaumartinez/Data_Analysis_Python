# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:16:10 2022

@author: Luciana
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

base_folder = r'C:\Ubuntu_archivos\Printing'

daily_folder = r'2022-05-17 Au-Paladio\Test\processed_data_luciana_version_0.71'

daily_folder = r'2022-05-26 AuNP 67 impresas\20220526-182536_Luminescence_10x10_3.0umx0.0um\processed_data_p'
daily_folder = r'2022-05-26 AuNP 67 impresas\20220526-164518_Luminescence_10x10_3.0umx0.0um\processed_data_p'
daily_folder = r'2022-05-24 AuNP 67 impresas\20220524-173839_Luminescence_10x1_3.0umx3.0um\processed_data_BS_2.17, 0.033'

daily_folder = r'2022-05-12 (Au NP 60 nm control satelites paladio)\20220513-155112_Luminescence_Load_grid\processed_data'
daily_folder = r'2022-05-12 (Au NP 60 nm control satelites paladio)\20220513-105850_Luminescence_Load_grid\processed_data'
daily_folder = r'2022-05-12 (Au NP 60 nm control satelites paladio)\20220512-184652_Luminescence_Load_grid\processed_data'

daily_folder = r'2022-05-11 (Au NP 60 nm satelites paladio)\20220511-125259_Luminescence_Load_grid\processed_data'
daily_folder = r'2022-05-11 (Au NP 60 nm satelites paladio)\20220510-171340_Luminescence_Load_grid\processed_data'

parent_folder = os.path.join(base_folder, daily_folder)

save_folder = os.path.join(parent_folder, 'Pruebas Señal Anti y Stokes')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

list_of_folders = os.listdir(parent_folder)
list_of_folders_NP = [f for f in list_of_folders if re.search('Col',f)]

NP = len(list_of_folders_NP)

plt.figure()
name_fig = os.path.join(save_folder, 'pruebas_integral_antivsstokes_2.png')

for NP_folder in list_of_folders_NP:

    folder = os.path.join(parent_folder, NP_folder)
    
    file = os.path.join(folder, 'pl_in_bins', 'londa_%s.dat'%NP_folder)
    londa = np.loadtxt(file)
    N = len(londa)
    
    desired_stokes = np.where((londa >= 543) & (londa <= 560))
    desired_antistokes = np.where((londa >= 515) & (londa <= 520))
    
    londa_stokes = londa[desired_stokes]
    londa_antistokes = londa[desired_antistokes]
    
    file = os.path.join(folder, 'pl_in_bins', 'all_bins_%s.dat'%NP_folder)
    matrix_specs_bins = np.loadtxt(file)
    totalbins = matrix_specs_bins.shape[1]
    
   # matrix_stokes = np.zeros((len(londa_stokes), totalbins))
   # matrix_antistokes = np.zeros((len(londa_antistokes), totalbins))
    
    stokes_int = np.zeros(totalbins)
    antistokes_int = np.zeros(totalbins)
    
    for i in range(totalbins):
        
        spec = matrix_specs_bins[:, i]
        
       # plt.plot(londa, spec)
        
        stokes = spec[desired_stokes]
        antistokes = spec[desired_antistokes]
        
       # matrix_stokes[:, i] = stokes
       # matrix_antistokes[:, i] = antistokes
        
        stokes_int[i] = np.sum(stokes)/(560 - 543)
        antistokes_int[i] = np.sum(antistokes)/(520 - 515)
        
    data = np.array([stokes_int, antistokes_int]).T
    name =  os.path.join(save_folder, 'integrales_stokes_antistokes_%s.dat'%NP_folder)
    header = 'stokes, antistokes'
  #  np.savetxt(name, data, header = header)
        
    plt.plot(stokes_int*10**-6, antistokes_int*10**-6, 'o')   
    
plt.title('NP = %s'%NP)
plt.xlabel('Integral Stokes 543-560 nm')
plt.ylabel('Integral Anti-Stokes 515-520 nm')
plt.xlim(0, 1.6)
plt.ylim(0, 0.3)
#plt.savefig(name_fig)
plt.show()

#%%               
                       
    
list_of_folders = os.listdir(save_folder)
list_of_folders_NP = [f for f in list_of_folders if re.search('Col',f)]
NP = len(list_of_folders_NP)

plt.figure()
plt.title('Referencia Last Bin')

x = []
y = []

for f in list_of_folders_NP:
    
    file = os.path.join(save_folder, f)
    integrales = np.loadtxt(file, skiprows = 1)
    
    stokes_int = integrales[:, 0]
    antistokes_int = integrales[:, 1]
    
    stokes_int_ref_bin = stokes_int[-1]
    antistokes_int_ref_bin = antistokes_int[-1]
    
    qs = stokes_int/stokes_int_ref_bin
    qa = antistokes_int/antistokes_int_ref_bin
    
    x.append(qs)
    y.append(qa)
        
    plt.plot(qs,qa, 'o')

        
plt.title('NP = %s'%NP)
plt.xlabel('Quotient Integral Stokes 543-560 nm')
plt.ylabel('Quotient Integral Anti-Stokes 515-520 nm')
#plt.xlim(0, 1.6)
#plt.ylim(0, 0.3)
#plt.savefig(name_fig)
plt.show()     

#%%


def average(x, y, totalbins):
    
    totalbins = len(stokes_int)
    step = 1/totalbins
    bins = np.arange(0, 1 + step, step)
    
    xbins = np.zeros(totalbins)
    ybins = np.zeros(totalbins)
    
    xbins_std = np.zeros(totalbins)
    ybins_std = np.zeros(totalbins)
    
    for i in range(totalbins):
        
        desired = np.where((x >= bins[i]) & (x <= bins[i+1]))
                           
        xi = x[desired]
        yi = y[desired]

        xbins[i] = np.mean(xi)
        xbins_std[i] = np.std(xi)

        ybins[i] = np.mean(yi)
        ybins_std[i] = np.std(yi)   

    return xbins, ybins, xbins_std, ybins_std

xbins, ybins, xbins_std, ybins_std = average(np.array(x), np.array(y), totalbins)

plt.figure()
plt.plot(x, y, 'o')
plt.errorbar(xbins, ybins, xerr = xbins_std, yerr = ybins_std, fmt = 'o')
plt.xlabel('Quotient Integral Stokes 543-560 nm')
plt.ylabel('Quotient Integral Anti-Stokes 515-520 nm')
#plt.xlim(0, 1.6)
#plt.ylim(0, 0.3)
#plt.savefig(name_fig)
plt.show()     

#%%

name_fig = os.path.join(save_folder, 'pruebas_integral_antivsstokes_3.png')

p = np.polyfit(xbins, ybins, 2)
p = np.round(p, 3)
yf = np.polyval(p, xbins)
print(p)

plt.figure()
plt.plot(x, y, 'o')
plt.errorbar(xbins, ybins, xerr = xbins_std, yerr = ybins_std, fmt = 'o')
plt.plot(xbins, yf, 'r--')
plt.xlabel('Quotient Integral Stokes 543-560 nm')
plt.ylabel('Quotient Integral Anti-Stokes 515-520 nm')
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.text(0.1, 0.9, 'poly = %.3f, %.3f, %.3f'%(p[0], p[1], p[2]))
#plt.savefig(name_fig)
plt.show()

data = np.array([xbins, xbins_std, ybins, ybins_std]).T
name =  os.path.join(save_folder, 'mean_bins_integrales_stokes_antistokes.dat')
header = 'mean int stokes, std in stokes, mean int antistokes, std int antistokes'
#np.savetxt(name, data, header = header)

data = p.T
name =  os.path.join(save_folder, 'poly2_stokes_antistokes.dat')
#np.savetxt(name, data)