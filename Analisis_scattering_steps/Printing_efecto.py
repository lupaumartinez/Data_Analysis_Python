# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:39:41 2022

@author: lupau
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:35:56 2022

@author: Luciana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

base_folder = r'C:\Users\lupau\OneDrive\Documentos'
daily_folder = r'2022-06-29 Nanostars R20'

parent_folder = os.path.join(base_folder, daily_folder)
save_folder = manage_save_directory(parent_folder, 'figures_scattering_2')

file_bulk = os.path.join(base_folder, daily_folder, 'PVP P2R20_eene2022.txt')
data = np.loadtxt(file_bulk, skiprows =2)
londa = data[:, 0]
spec = data[:, 1]
desired = np.where((londa > 500) & (londa < 850))
londa_exc = londa[desired]
spec = spec[desired]
spec = spec-min(spec)
exc = spec/np.max(spec)

file_bulk_2020 = os.path.join(base_folder, daily_folder, 'PVP R20P2.txt')
data_2020 = np.loadtxt(file_bulk_2020, skiprows = 2)
londa_2020 = data_2020[:, 0]
spec_2020 = data_2020[:, 1]
desired_2020 = np.where((londa_2020 > 500) & (londa_2020 < 850))
londa_exc_2020 = londa_2020[desired_2020]
spec_2020 = spec_2020[desired_2020]
spec_2020 = spec_2020-min(spec_2020)
exc_2020 = spec_2020/np.max(spec_2020)

folder_initial = '20220629-143210_Scattering_Steps_Load_grid_INICIAL_4seg'
folder_initial = os.path.join(base_folder, daily_folder, folder_initial)
folder_initial = os.path.join(folder_initial, 'Col_1000',  'fig_spectrums')
    
folder_final = '20220629-152148_Scattering_Steps_Load_grid_FINAL_6seg'
folder_final = os.path.join(base_folder, daily_folder, folder_final)
folder_final = os.path.join(folder_final, 'Col_1000',  'fig_spectrums')

list_f = [folder_initial, folder_final]

colors = ['b', 'm']

mean_spec = np.zeros((723, 2))
mean_londa = np.zeros(723)

n = 5
londa_max = np.zeros((n, 2))

plt.figure()

for j in range(2):
    
    f = list_f[j]
    
    list_files = os.listdir(f)
    list_files =  [f for f in list_files if re.search('txt',f)]
    list_files.sort()
    
    c = colors[j]
    
    for i in range(n):
        
        NP_file = list_files[i]
        
     #   print(NP_file)
        
        file = os.path.join(f, NP_file)
        
        data = np.loadtxt(file)
        
        londa = data[:, 0]
        spec = data[:, 2]
        
        londa_max[i, j] = londa[np.argmax(spec)]
        
        mean_spec[:, j] = mean_spec[:, j] + spec
       
        plt.plot(londa, spec, color = c, alpha = 0.4)

    mean_spec[:, j] = mean_spec[:, j]/n
    mean_spec[:, j] = mean_spec[:, j] - min(mean_spec[:, j])
    mean_spec[:, j] = mean_spec[:, j]/max(mean_spec[:, j])
    mean_londa = londa
    
    plt.plot(londa, mean_spec[:, j], color = c, linewidth = 5)

plt.plot(londa_exc_2020, exc_2020, 'k--', linewidth = 3, alpha = 0.6)    
plt.plot(londa_exc, exc, color = 'grey', linestyle = '--', linewidth = 3, alpha = 0.6)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Scattering')
plt.show()
name = os.path.join(save_folder, 'scattering.png')
plt.savefig(name)

plt.figure()
plt.plot(np.arange(0, n), londa_max[:, 0], 'bo')
plt.plot(np.arange(0, n), londa_max[:, 1], 'mo')
plt.hlines(londa_exc[np.argmax(exc)], 0, n-1, color = 'k')
plt.hlines(londa_exc_2020[np.argmax(exc_2020)], 0, n-1, color = 'grey')
plt.hlines(mean_londa[np.argmax(mean_spec[:, 0])], 0, n-1, color = 'b')
plt.hlines(mean_londa[np.argmax(mean_spec[:, 1])], 0, n-1, color = 'm')

y_star = mean_londa[np.argmax(mean_spec[:, 0])]
y_err_star = np.mean(abs(y_star- londa_max[:, 0]))
plt.fill_between(np.arange(0, n), y_star - 2*y_err_star, y_star + 2*y_err_star, color = 'b', alpha = 0.2)

y_sph = mean_londa[np.argmax(mean_spec[:, 1])]
y_err_sph = np.mean(abs(y_sph- londa_max[:, 1]))
plt.fill_between(np.arange(0, n), y_sph - 2*y_err_sph, y_sph + 2*y_err_sph, color = 'm', alpha = 0.2)

medium =  (y_sph + y_star)/2

plt.ylabel('Max Wavelength')
plt.show()
name =  os.path.join(save_folder, 'wavelength_scattering.png')
#plt.savefig(name)

#%%

parent_folder = os.path.join(base_folder, r'2022-07-08 Nanostars P2R20 impresion\scattering')

save_folder = manage_save_directory(parent_folder, 'figures_scattering_printing')

plt.close('all')

def find_londa_max(parent_folder, folder_printing, save_folder, name, color):
    
    plt.figure()
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Scattering')
        
    f = os.path.join(parent_folder, folder_printing)
    
    list_files = os.listdir(f)
    list_files =  [f for f in list_files if re.search('txt',f)]
    list_files.sort()
    
    list_londa_max = []
    
    for col_file in list_files:
        
        file = os.path.join(f, col_file)
        
        data = np.loadtxt(file)
        
        londa = data[:, 0]
        
        desired = np.where((londa > 500) & (londa < 850))
        londa = londa[desired]
        
        NP = data[:,1:].shape[1]
    
        for k in range(NP):
    
            spec = data[:, k+1]
            spec = spec[desired]
            
            if max(spec)<0.1:
                print('no tiene espectro', col_file, k)
                continue
            
            spec = spec-min(spec)
            spec = spec/np.max(spec)
            
            list_londa_max.append(londa[np.argmax(spec)])
        
            plt.plot(londa, spec, color = color, alpha = 0.4)
        
    plt.plot(mean_londa, mean_spec[:, 0], color = 'b', linewidth = 5)
    plt.plot(mean_londa, mean_spec[:, 1], color = 'm', linewidth = 5) 
    plt.plot(londa_exc, exc, 'k--', linewidth = 3, alpha = 0.6)
    plt.xlabel('Wavelength (nm)')
    plt.show()
    
    name1 = os.path.join(save_folder, 'scattering_printing_%s.png'%name)
    plt.savefig(name1)
    
    bins = 4
    step_bin = ((y_star + 2*y_err_star) - (y_sph - 2*y_err_sph))/bins
    rango = [y_sph - 2*y_err_sph, y_sph - 2*y_err_sph + bins*step_bin]
    ylim = 100
    plt.figure()
    
    plt.axvspan(y_star - 2*y_err_star, y_star + 2*y_err_star, alpha=0.2, color='b')
    plt.axvspan(y_sph - 2*y_err_sph, y_sph + 2*y_err_sph, alpha=0.2, color='m')
    
    plt.vlines(y_star - 2*y_err_star, 0, ylim, color = 'k', alpha = 0.1)
    plt.vlines(y_sph + 2*y_err_sph, 0, ylim, color = 'k', alpha = 0.1)
    plt.vlines(medium, 0, ylim, color = 'k', alpha = 0.1)
    
    plt.hist(list_londa_max, bins = bins, range = rango, rwidth=True, align = 'mid', color = color, alpha = 0.5)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Frequency')
    plt.ylim(0, ylim)

    name2 = os.path.join(save_folder, 'hist_all_scattering_printing_%s.png'%name)
    plt.savefig(name2)

    return list_londa_max

def contador(list_londa_max, save_folder, name):
    
    N = len(list_londa_max)
    n_star = 0
    n_sph = 0
    n_star_close = 0
    n_sph_close = 0
    
    l_star = 0
    l_sph = 0
    l_star_close = 0
    l_sph_close = 0
    
    for l in list_londa_max:
        
        if y_star - 2*y_err_star <= l:# <= y_star + 2*y_err_star:
            
            n_star = n_star + 1
            
            l_star = l_star + l
            
        if medium < l < y_star - 2*y_err_star:
            
            n_star_close = n_star_close + 1
            
            l_star_close = l_star_close + l
    
        if y_sph + 2*y_err_sph < l <= medium:
            
            n_sph_close = n_sph_close + 1
            
            l_sph_close = l_sph_close + l
            
        if l <= y_sph + 2*y_err_sph:
            
            n_sph = n_sph + 1
            
            l_sph = l_sph + l
            
    l_star = l_star/n_star
    l_sph = l_sph/n_sph
    l_star_close = l_star_close/n_star_close
    l_sph_close = l_sph_close/n_sph_close
            
    n_int = N - n_star - n_sph - n_star_close - n_sph_close
            
    print('N total', N)
    print('N star', n_star, 'N sph', n_sph)
    print('N star close', n_star_close, 'N sph close', n_sph_close)
    print('N intermedio', n_int)
    
    plot_histogram(list_londa_max, save_folder, name)
    
    list_l_mean = [l_sph, l_sph_close, l_star_close, l_star]
    list_N = [n_sph, n_sph_close, n_star_close, n_star]
    
    return list_l_mean, list_N

def plot_histogram(list_londa_max, save_folder, name):
     
    l0 = []
    l1 = []
    l2 = []
    l3 = []
    
    for l in list_londa_max:
        
        if y_star - 2*y_err_star <= l:# <= y_star + 2*y_err_star:
            
            l0.append(l)
            
        if medium < l < y_star - 2*y_err_star:
            
            l1.append(l)
            
       # if y_sph - 2*y_err_sph <= l <= y_sph + 2*y_err_sph:
        if l <= y_sph + 2*y_err_sph:
            
            l2.append(l)
        
        if y_sph + 2*y_err_sph < l <= medium:
            
            l3.append(l)
    
    bins = 10
    rango = [550, 850]
    plt.figure()
    plt.hist(l0, bins = bins, range = rango, rwidth=True, align = 'mid', alpha = 0.5, color = 'b')
    plt.hist(l1, bins = bins, range = rango, rwidth=True, align = 'mid', alpha = 0.5)
    plt.hist(l2, bins = bins, range = rango, rwidth=True, align = 'mid', alpha = 0.5, color = 'm')
    plt.hist(l3, bins = bins, range = rango, rwidth=True, align = 'mid', alpha = 0.5)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Frequency')
    plt.ylim(0, 35)
    
    name = os.path.join(save_folder, 'hist_scattering_printing_%s.png'%name)
    plt.savefig(name)
    
    return


folder_printing_red = r'G3\photos\smooth_spectrums'
list_londa_max_red = find_londa_max(parent_folder, folder_printing_red, save_folder, '640nm_1.55mW', color = 'r')
folder_printing_red_2 = r'G2\photos\smooth_spectrums'
list_londa_max_red_2 = find_londa_max(parent_folder, folder_printing_red_2, save_folder, '640nm_1.65mW', color = 'y')
folder_printing_green = r'G4\photos\smooth_spectrums'
list_londa_max_green = find_londa_max(parent_folder, folder_printing_green, save_folder, '532nm_3.20mW', color = 'g')

plt.figure()
plt.plot(list_londa_max_red, 'ro')
plt.plot(list_londa_max_red_2, 'yo')
plt.plot(list_londa_max_green, 'go')
#plt.hlines(londa_exc[np.argmax(exc)], 0, 100, color = 'k')
plt.hlines(mean_londa[np.argmax(mean_spec[:, 0])], 0, 100, color = 'b')
plt.hlines(mean_londa[np.argmax(mean_spec[:, 1])], 0, 100, color = 'm')
plt.hlines(y_star, 0, 100, color = 'b')
plt.hlines(y_sph, 0, 100, color = 'm')
plt.fill_between(np.arange(-1, 101), y_star - 2*y_err_star, y_star + 2*y_err_star, color = 'b', alpha = 0.2)
plt.fill_between(np.arange(-1, 101), y_sph - 2*y_err_sph, y_sph + 2*y_err_sph, color = 'm', alpha = 0.2)
plt.ylabel('Max Wavelength (nm)')
plt.xlabel('NP printing')
plt.show()
name =  os.path.join(save_folder, 'wavelength_scattering_printing.png')
plt.savefig(name)

print('red 1.6 mW')
list_l_mean_r, list_N_r = contador(list_londa_max_red,save_folder, '640nm_1.55mW')
print('red 2 mW')
list_l_mean_r2, list_N_r2 =contador(list_londa_max_red_2,save_folder, '640nm_1.65mW')
print('green 4 mW')
list_l_mean_g, list_N_g = contador(list_londa_max_green, save_folder, '532nm_3.20mW')

plt.figure()
plt.plot(list_l_mean_r, list_N_r/np.sum(list_N_r), 'ro--', label = '640nm_1.55mW')
plt.plot(list_l_mean_r2, list_N_r2/np.sum(list_N_r2), 'yo--', label = '640nm_1.65mW')
plt.plot(list_l_mean_g, list_N_g/np.sum(list_N_g), 'go--', label = '532nm_3.20mW')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Porcentage %')
plt.legend()
name =  os.path.join(save_folder, 'Porcentage_scattering_printing.png')
plt.savefig(name)
