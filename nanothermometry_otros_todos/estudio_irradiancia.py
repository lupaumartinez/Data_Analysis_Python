# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:53:15 2022

@author: lupau
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import skimage.io


plt.style.use('default')
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams ["axes.labelsize"] = 14
plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=14

def irradiance_theoric(power, w0, x, x0):
    
    I = (2*power/(np.pi * w0**2)) * np.exp(-(2*(x-x0)**2/(w0**2))) 

    return I

def image_confocal_stokes(parent_folder, f, plot):
    
    folder = os.path.join(parent_folder, f)
    
    list_of_folders = os.listdir(folder)
    files = [f for f in list_of_folders if re.search('tiff',f)]
    file = os.path.join(folder, files[0])
    
    matrix = skimage.io.imread(file)
    
    if plot:
            
        fig = plt.figure(figsize=(6, 12))
        axes = fig.subplots(ncols=1, nrows=1)
        
        ims = axes.imshow(matrix)#, cmap= 'inferno')
        axes.grid(False)
    
        cax = axes.inset_axes([1.02, 0.0, 0.03, 1])
        cbar = fig.colorbar(ims, ax=axes, cax=cax)
        cbar.set_label("Integrate PL Stokes")
        
        fig.set_tight_layout(True)
        
        plt.show()
    
    return matrix

def classification(value, totalbins, rango):
    # Bin the data. Classify a value into a bin.
    # totalbins = number of bins to divide rango (range)
    bin_max = totalbins - 1
    numbin = 0
    inf = rango[0]
    sup = rango[1]
    if value > sup:
        print('Value higher than max')
        return bin_max
    if value < inf:
        print('Value lower than min')
        return 0
    step = (sup - inf)/totalbins
    # tiene longitud totalbins + 1
    # pero en total son totalbins "cajitas" (bines)
    binned_range = np.arange(inf, sup + step, step)
    while numbin < bin_max:
        if (value >= binned_range[numbin] and value < binned_range[numbin+1]):
            break
        numbin += 1
        if numbin > bin_max:
            break
    return numbin

def binning(path_folder, NP, matrix, px, totalbins, plot):
    
    binned_power = np.zeros((px, px))
    
    pixel_cts_per_sec = (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))
    
    for i in range(px):
       for j in range(px):
           nbin = classification(pixel_cts_per_sec[i,j], totalbins, [0,1])
           binned_power[i,j] = nbin
    
    #print(binned_power)
    
    if plot:
    
        fig=plt.figure(num=1,clear=True)
        ax = fig.add_subplot()
        ax.set_xticks(range(px))
        ax.set_yticks(range(px))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(binned_power, cmap = 'plasma')
        figure_name = os.path.join(path_folder,'binned_power_%s.png' % NP)
        fig.savefig(figure_name, dpi = 300)
        
        plt.close()
    
    return binned_power


def calc_irradiance(factor, w0, err_w0, matrix, power_mean, power_std):
    
    pixel_cts_per_sec = (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))
    
    calc_irrad = factor*power_mean*2/(np.pi*w0**2)
    err_calc_irrad = factor*np.sqrt((2/(np.pi*w0**2)*power_std)**2 +
                             (4*power_mean*err_w0/(np.pi*w0**3))**2 )
    
    matrix_irradiance = calc_irrad*pixel_cts_per_sec
    
    matrix_std_irradiance = err_calc_irrad*pixel_cts_per_sec
    
    return matrix_irradiance, matrix_std_irradiance

def bins_irradiance(path_folder, NP, matrix_irradiance, matrix_std_irradiance, totalbins, binned_power):
    
    mean_irradiance = np.zeros(totalbins)
    std_irradiance = np.zeros(totalbins)
                     
    for s in range(totalbins):
       u, v = np.where(binned_power == s)
       counter = 0
       aux_mean_irr = 0
       aux_std_irr = 0
       for i, j in zip(u, v):
           aux_mean_irr += matrix_irradiance[i,j]
           aux_std_irr  += matrix_std_irradiance[i,j]**2
           counter += 1
       if counter == 0:
           print('Bin %d has NO power to average.' % s)
           aux_mean_irr = aux_mean_irr
           aux_std_irr = aux_std_irr
       else:
         #  print('Bin %d has %d power to average.' % (s, counter))
           aux_mean_irr = aux_mean_irr/counter
           aux_std_irr = np.sqrt(aux_std_irr/counter)
            
       mean_irradiance[s] = aux_mean_irr
       std_irradiance[s] = aux_std_irr
       
    plt.figure()
    plt.errorbar(range(totalbins), mean_irradiance, yerr = std_irradiance, fmt = 'o')
    plt.ylabel(u'Irradiance (mW/nm$^{2}$)')
    plt.xlabel('Bin')
    figure_name = os.path.join(path_folder,'irradiance_in_bins_%s.png' % NP)
    plt.savefig(figure_name, dpi = 300)
    
    plt.close()
    
    return mean_irradiance, std_irradiance

def binning_distance(path_folder, NP, totalbins, binned_power, size_image, pixel):
    
    pixel_size = size_image/pixel
    axe = np.arange(-size_image/2+pixel_size/2, size_image/2+pixel_size/2, pixel_size)
    
    mean_distance = np.zeros(totalbins)
                     
    for s in range(totalbins):
       u, v = np.where(binned_power == s)
       counter = 0
       aux_distance = 0
       for i, j in zip(u, v):
           d = np.sqrt(axe[i]**2 + axe[j]**2)
           aux_distance = aux_distance + d
           counter += 1
       if counter == 0:
           print('Bin %d has NO power to average.' % s)
           aux_distance = aux_distance
       else:
         #  print('Bin %d has %d power to average.' % (s, counter))
           aux_distance = aux_distance/counter
            
       mean_distance[s] = aux_distance
       
    std_distance =  np.zeros(totalbins)
    
    for s in range(totalbins):
        u, v = np.where(binned_power == s)
        counter = 0
        aux_power = 0
        for i, j in zip(u, v):
           d = np.sqrt(axe[i]**2 + axe[j]**2)
           aux_power += (d - mean_distance[s])**2
           counter += 1
        if counter == 0:
           print('Bin %d has NO power to average.' % s)
           aux_power = aux_power
        else:
          # print('Bin %d has %d power to average.' % (s, counter))
           aux_power = np.sqrt(aux_power/(counter-1))
           
        std_distance[s] = aux_power
       
    plt.figure()
    plt.errorbar(range(totalbins), mean_distance, yerr = std_distance, fmt = 'o')
    plt.ylabel(u'Distance (nm)')
    plt.xlabel('Bin')
    figure_name = os.path.join(path_folder,'distance_in_bins_%s.png' % NP)
    plt.savefig(figure_name, dpi = 300)
    
    plt.close()
    
    return mean_distance, std_distance

if __name__ == '__main__':
    
    plt.close('all')
    
    base_folder =r'C:\Users\lupau\OneDrive\Documentos'
    
    daily_folder = '2022-05-24 AuNP 67 impresas/20220524-173839_Luminescence_10x1_3.0umx3.0um'

    parent_folder = os.path.join(base_folder, daily_folder)
    
    save_folder = os.path.join(parent_folder, 'psf_bins10_0.73mW_342nm')
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    size_image = 800 #nm    
    pixel = 10
    totalbins = 10
    
    factor = 0.47
    w0 = 342#342#42 #nm
    err_w0 = 5 #nm
    power_mean = 0.73
    power_std = 0.03
    

    power = power_mean*factor
    x0 = 0
    axe = np.arange(0, size_image/2, 2)
    I = irradiance_theoric(power, w0, axe, x0)
    
    list_of_folders = os.listdir(parent_folder)
    list_of_folders_slow = [f for f in list_of_folders if re.search('Slow_Confocal_Spectrum',f)]
    list_of_folders_slow.sort()
    
    plt.figure()
    
    for f in list_of_folders_slow[:]:
    
        NP = f.split('Spectrum_')[-1]
        NP = NP.split('.txt')[0]
        
        number_NP = NP.split('NP_')[-1]
        number_NP = int(number_NP)
        
        matrix_stokes = image_confocal_stokes(parent_folder, f, plot = False)
        
        matrix_irradiance, matrix_std_irradiance = calc_irradiance(factor, w0, err_w0, matrix_stokes, power_mean, power_std)
       
        binned_power = binning(save_folder, NP, matrix_stokes, pixel, totalbins, plot = False)
        mean_irradiance, std_irradiance = bins_irradiance(save_folder, NP, matrix_irradiance, matrix_std_irradiance, totalbins, binned_power)
        mean_distance, std_distance = binning_distance(save_folder, NP, totalbins, binned_power, size_image, pixel)
            
        
        plt.errorbar(mean_distance-mean_distance[-1], mean_irradiance, xerr = std_distance, yerr = std_irradiance, fmt = 'o')
    plt.plot(axe, I, 'k--')
    plt.xlabel(u'Distance (nm)')
    plt.ylabel(u'Irradiance (mW/nm$^{2}$)')
    figure_name = os.path.join(save_folder,'irradiance_vs_distance_of_bins.png')
    plt.savefig(figure_name, dpi = 300)
    plt.show()