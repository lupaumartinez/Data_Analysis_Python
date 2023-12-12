#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:02:07 2019

@author: luciana
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:03:37 2019

@author: Luciana
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def compare_spectrum(common_path_stokes, common_path_scattering, save_folder, name_col, number_NP, plot_each_one):
        
    number_col = int(name_col.split('_')[-1])
     
    list_of_folders = os.listdir(common_path_stokes)
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Files from PL', L, list_of_folders)
    
    folder_sca = '%s/normalized_%s'%(name_col, name_col)
    common_path_scattering = os.path.join(common_path_scattering, folder_sca)
    list_of_folders_scattering = os.listdir(common_path_scattering)
    list_of_folders_scattering = [f for f in list_of_folders_scattering if re.search('NP',f)]
    list_of_folders_scattering.sort()
    M = len(list_of_folders_scattering)  
    
    print('Files from Scattering', M, list_of_folders_scattering)
    
    if L == M:
        print('Plot PL, compare with Scattering')
    else:
        print('The files do not have the same NP. Check files')
        
    name_file = os.path.join(common_path_stokes, list_of_folders[0])
    a = np.loadtxt(name_file, skiprows=1)
    wavelength = a[:, 0]
    mean_PL = np.zeros(len(wavelength))
    
    NP_PL_list = []
    lspr_PL = []
    first_wave = 545  #538,5 #545
    end_wave = 690  #641.5  #641.5 #620
    #para PL growth
    desired_range = np.where((wavelength >= first_wave) & (wavelength <=end_wave))    
    desired_wave = wavelength[desired_range]
    x = np.linspace(desired_wave[0], desired_wave[-1], 1000)
    npol = 5
        
    name_file_scattering = os.path.join(common_path_scattering,  list_of_folders_scattering[0])
    b = np.loadtxt(name_file_scattering, skiprows=1)
    wavelength_sca = b[:, 0]
    mean_sca = np.zeros(len(wavelength_sca))
        
    plt.figure()
    plt.title(name_col)
            
    for i in range(M):
        
        NP = list_of_folders_scattering[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        NP_PL = (number_col- 1)*number_NP + int(NP)
        NP_PL_list.append(int(NP_PL))
        NP_PL = '%03d'%NP_PL
        
        print('Ploteo %s_NP_%s'%(name_col, NP_PL))
        
        file_PL = 'Luminescence_Steps_Spectrum_%s_NP_%s.txt'%(name_col, NP_PL)
        name_file = os.path.join(common_path_stokes,  file_PL)
        a = np.loadtxt(name_file, skiprows=1)
        wavelength = a[:, 0]
        luminiscence = a[:, 2]
        
    #   luminiscence_NP = a[:, 1]
    #   luminiscence_NP_substrate = a[:, 2]
    #   luminiscence_susbtrate = a[:, 3]
        
        desired_PL_stokes = luminiscence[desired_range]
        p_PL = np.polyfit(desired_wave, desired_PL_stokes, npol)
        poly_PL = np.polyval(p_PL, x)
        max_wave_poly_PL = round(x[np.argmax(poly_PL)],3)
        londa_spr_PL = np.round(max_wave_poly_PL,2)
        lspr_PL.append(londa_spr_PL)
   
        name_file_scattering = os.path.join(common_path_scattering,  list_of_folders_scattering[i])
        b = np.loadtxt(name_file_scattering, skiprows=1)
        wavelength_sca = b[:, 0]
        scattering = b[:, 1]
        
        min_PL = np.mean(luminiscence[np.where((wavelength< 535) & (wavelength< 525))])
        PL =  (luminiscence-min_PL)/(max(luminiscence)-min_PL)
        
        sca = (scattering-min(scattering))/(max(scattering)-min(scattering))
        
        mean_PL = PL + mean_PL
        mean_sca = sca + mean_sca
        
        if plot_each_one:
        
            plt.figure()        
            plt.title('%s_NP_%s'%(name_col, NP))
            plt.plot(wavelength, PL , label = 'PL')
            plt.plot(wavelength_sca, sca, label = 'Scattering')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')
            plt.legend(loc='upper right') 
            figure_name = os.path.join(save_folder, 'compare_PL_vs_Scattering_%s_NP_%s.png'%(name_col, NP_PL)) 
            plt.savefig(figure_name , dpi = 400)
            plt.close()
        
        quotient_PL_Scattering(name_col, NP, wavelength, PL, wavelength_sca, sca, plot_bool = False)
        plt.plot(wavelength, PL, label = NP)
        plt.plot(wavelength_sca, sca) 
        
    mean_PL = mean_PL/M
    mean_sca = mean_sca/M
    
    plt.plot(wavelength, mean_PL, 'k--', label = 'mean PL')
    plt.plot(wavelength_sca, mean_sca, 'r--', label = 'mean Sca')
        
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend()
    figure_name = os.path.join(save_folder, 'compare_PL_vs_Scattering_%s.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    plot_mean(save_folder, wavelength, mean_PL, wavelength_sca, mean_sca, name_col)
    quotient_mean_PL_Scattering(name_col, wavelength, mean_PL, wavelength_sca, mean_sca)
                    
    return NP_PL_list, lspr_PL

def plot_mean(save_folder, wavelength, mean_PL, wavelength_sca, mean_sca, name_col):
    
    save_folder_mean = os.path.join(save_folder, 'fig_compare_mean')
    
    name = os.path.join(save_folder_mean, 'means_PL_%s.txt'%(name_col))
    data = np.array([wavelength, mean_PL]).T
    header_txt = 'wavelength (nm) mean PL'
    np.savetxt(name, data, header = header_txt)
    
    name = os.path.join(save_folder_mean, 'means_Sca_%s.txt'%(name_col))
    data = np.array([wavelength_sca, mean_sca]).T
    header_txt = 'wavelength (nm), mean Scattering'
    np.savetxt(name, data, header = header_txt)
    
    plt.figure()
    plt.title(name_col)
    
    plt.plot(wavelength, mean_PL, color = 'C1', label = 'mean PL')
    plt.plot(wavelength_sca, mean_sca, color = 'C2', label = 'mean Sca')
        
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend()
    figure_name = os.path.join(save_folder_mean, 'mean_PL_vs_Scattering_%s.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    return

def compare_lspr(NP_PL, lspr_PL, common_path_scattering, save_folder_max, name_col):
    
    file_sca_max = '%s/fig_normalized_%s/max_wavelength.txt'%(name_col, name_col)
    
    name_file = os.path.join(common_path_scattering,  file_sca_max)
    b = np.loadtxt(name_file, skiprows=1)
    NP_sca = b[:, 0]
    max_sca_poly = b[:, 2]
    
    print(name_col)
    print('LSPR', 'NP scattering:', NP_sca, 'NP PL:', NP_PL)
    
    plt.figure()
    plt.plot(NP_sca, lspr_PL, color = 'C1',  linestyle = None, marker = 'o', label = 'lspr PL steps analysis poly')
    plt.plot(NP_sca, max_sca_poly, color = 'C2',  linestyle = None, marker ='o', label = 'Scattering')
    plt.xlabel('NP')
    plt.ylabel('LSPR')
    plt.legend(loc='lower right') 
    
    #plt.xlim(500, 1000)
    plt.ylim(540, 600)
    
    figure_name = os.path.join(save_folder_max, 'lspr_%s.png'%(name_col)) 
    
    #plt.show()
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
   # plot_time(total_time_growth, final_lspr_growth, lspr_PL, save_folder, name_col)
   
    name = os.path.join(save_folder_max,  'lspr_%s.txt'%(name_col))
    data = np.array([NP_sca, max_sca_poly, NP_PL, lspr_PL]).T
    header_txt = 'NP, max poly Sca (nm), NP, lspr PL steps analysis poly (nm)'
    np.savetxt(name, data, header = header_txt)
    
    plot_histogram(save_folder_max, name_col, lspr_PL, max_sca_poly)
    
    return

def data_hist(x, bin_positions):
    
    mu= round(np.mean(x), 2) # media
    sigma= round(np.std(x, ddof=1), 2) #desviación estándar
    N=len(x) # número de cuentas
    std_err = round(sigma / N,2) # error estándar
    
    # muestro estos resultados
    print( 'media: ', mu)
    print( 'desviacion estandar: ', sigma)
    print( 'total de cuentas: ', N)
    print( 'error estandar: ', std_err)

   # bin_size=bin_positions[1]-bin_positions[0] # calculo el ancho de los bins del histograma
   # x_gaussiana=np.linspace(mu-5*sigma, mu+5*sigma, num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
   # gaussiana= norm.pdf(x_gaussiana, mu, sigma)#*N*bin_size # calculo la gaussiana que corresponde al histograma
    
    txt = '%s + %s'%(mu, sigma)
    
    return mu, sigma, txt

def plot_histogram(save_folder, name_col, lspr_PL, max_sca):
    
    print('Plot Histogram ')   
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    lim_range = [520, 620]
    bins = 10
    
    n,bin_positions_x,p = ax1.hist(lspr_PL, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C1')
    mu_lspr_PL, sigma_lspr_PL, x_text = data_hist(lspr_PL, bin_positions_x)
    ax1.text(550, 0.10, x_text, color= 'C1', fontsize = 'medium')
    ax1.set_xlabel('LSPR PL (nm)')
    ax1.set_ylim(0, 0.12)
    ax1.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax2.hist(max_sca, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C2')
    mu_lspr_sca, sigma_lspr_sca, x_text = data_hist(max_sca, bin_positions_x)
    ax2.text(550, 0.10, x_text, color= 'C2', fontsize = 'medium')
    ax2.set_xlabel('LSPR Scattering (nm)')
    ax2.set_ylim(0, 0.12)
    ax2.set_xlim(lim_range[0],lim_range[1])
    
    fig.set_tight_layout(True)

    figure_name = os.path.join(save_folder, 'histogram_%s.png'%name_col) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    return

def std_spectrum(common_path_stokes, common_path_scattering, save_folder, name_col, number_NP):
    
    file_mean_PL = os.path.join(save_folder, 'fig_compare_mean', 'means_PL_%s.txt'%name_col)
    data_mean_PL = np.loadtxt(file_mean_PL, skiprows = 1)
    mean_PL = data_mean_PL[:, 1]
        
    file_mean_sca = os.path.join(save_folder, 'fig_compare_mean','means_Sca_%s.txt'%name_col)
    data_mean_sca = np.loadtxt(file_mean_sca, skiprows = 1)
    mean_sca = data_mean_sca[:, 1]
        
    number_col = int(name_col.split('_')[-1])
     
    list_of_folders = os.listdir(common_path_stokes)
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Files from PL', L, list_of_folders)
    
    folder_sca = '%s/normalized_%s'%(name_col, name_col)
    common_path_scattering = os.path.join(common_path_scattering, folder_sca)
    list_of_folders_scattering = os.listdir(common_path_scattering)
    list_of_folders_scattering = [f for f in list_of_folders_scattering if re.search('NP',f)]
    list_of_folders_scattering.sort()
    M = len(list_of_folders_scattering)  
    
    print('Files from Scattering', M, list_of_folders_scattering)
    
    if L == M:
        print('Plot PL, compare with Scattering')
    else:
        print('The files do not have the same NP. Check files')
        
    std_PL = np.zeros(len(mean_PL))
    std_sca = np.zeros(len(mean_sca))
        
    plt.figure()
    plt.title(name_col)
            
    for i in range(M):
        
        NP = list_of_folders_scattering[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        NP_PL = (number_col- 1)*number_NP + int(NP)
        NP_PL = '%03d'%NP_PL
        
        print('Ploteo %s_NP_%s'%(name_col, NP_PL))
        
        file_PL = 'Luminescence_Steps_Spectrum_%s_NP_%s.txt'%(name_col, NP_PL)
        name_file = os.path.join(common_path_stokes,  file_PL)
        a = np.loadtxt(name_file, skiprows=1)
        wavelength = a[:, 0]
        luminiscence = a[:, 1]
    #   luminiscence_NP = a[:, 1]
    #   luminiscence_NP_substrate = a[:, 2]
    #   luminiscence_susbtrate = a[:, 3]
   
        name_file_scattering = os.path.join(common_path_scattering,  list_of_folders_scattering[i])
        b = np.loadtxt(name_file_scattering, skiprows=1)
        wavelength_sca = b[:, 0]
        scattering = b[:, 1]
        
        min_PL = min(luminiscence[np.where(wavelength> 546)])
        PL =  (luminiscence-min_PL)/(max(luminiscence)-min_PL)
        
        sca = (scattering-min(scattering))/(max(scattering)-min(scattering))
        
        std_PL = (PL - mean_PL)**2 + std_PL
        std_sca = (sca - mean_sca)**2 + std_sca
            
    std_PL = np.sqrt(std_PL)/M
    std_sca = np.sqrt(std_sca)/M
    
    plt.plot(wavelength, std_PL, 'k--', label = 'std PL')
    plt.plot(wavelength_sca, std_sca, 'r--', label = 'std Sca')
        
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend()
    figure_name = os.path.join(save_folder, 'std_PL_vs_Scattering_%s.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close()
                    
    return

def quotient_mean_PL_Scattering(name_col, wavelength_PL, intensity_PL, wavelength_sca, intensity_sca):

    lower_lambda = 547
    upper_lambda = 625
    step = 1000
    
    wavelength_new = np.linspace(lower_lambda, upper_lambda, step)
    
    new_PL = np.interp(wavelength_new, wavelength_PL, intensity_PL)
    new_sca = np.interp(wavelength_new, wavelength_sca, intensity_sca)
    
    quotient = new_PL/new_sca
    quotient = (quotient-min(quotient))/(max(quotient)-min(quotient))
    
    npol = 3
    p = np.polyfit(wavelength_new , quotient, npol)
    q = np.polyval(p, wavelength_new )
    
    name = os.path.join(save_folder, 'fig_quotient_PL_Scattering', 'data_mean_%s.txt'%(name_col))
    data = np.array([wavelength_new, new_PL, new_sca, quotient, q]).T
    header_txt = 'wavelength (nm), PL norm, Scattering norm, Quotient PL/Sca norm, Quotient PL/Sca Fit'
    np.savetxt(name, data, header = header_txt)

    return

def quotient_PL_Scattering(name_col, NP, wavelength_PL, intensity_PL, wavelength_sca, intensity_sca, plot_bool):

    lower_lambda = 547
    upper_lambda = 625
    step = 1000
    
    wavelength_new = np.linspace(lower_lambda, upper_lambda, step)
    
    new_PL = np.interp(wavelength_new, wavelength_PL, intensity_PL)
    new_sca = np.interp(wavelength_new, wavelength_sca, intensity_sca)
    
    quotient = new_PL/new_sca
    quotient = (quotient-min(quotient))/(max(quotient)-min(quotient))
    
    npol = 3
    p = np.polyfit(wavelength_new , quotient, npol)
    q = np.polyval(p, wavelength_new )
    
    if plot_bool:
                  
        plt.figure()
        plt.title('NP_%s'%(NP))
        plt.plot(wavelength_new, quotient)
        plt.plot(wavelength_new, q, 'k--')  
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Quotient PL/Scattering')
        plt.ylim(-0.02, 1.02)
            
        figure_name = os.path.join(save_folder, 'fig_quotient_PL_Scattering', 'quotient_PL_vs_Scattering_%s_NP_%s.png'%(name_col, NP))
        plt.savefig(figure_name , dpi = 400)
        plt.close()
    
    name = os.path.join(save_folder, 'fig_quotient_PL_Scattering', 'data_%s_%s.txt'%(name_col, NP))
    data = np.array([wavelength_new, new_PL, new_sca, quotient, q]).T
    header_txt = 'wavelength (nm), PL norm, Scattering norm, Quotient PL/Sca norm, Quotient PL/Sca Fit'
    np.savetxt(name, data, header = header_txt)

    return

def compare_quotient(folder, name_col, wave_bulk, spectrum_bulk):
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('data_%s_'%name_col,f) ]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    plt.figure()
    plt.title('%s'%(name_col))

    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        file = os.path.join(folder, list_of_folders[i])
        data = np.loadtxt(file, comments = '#')
                          
        wave = data[:, 0]
        quotient = data[:, 3]
        q = data[:,4]

        plt.plot(wave, quotient, label = 'NP_%s'%(NP))
        plt.plot(wave, q, 'k--')
        
    plt.plot(wave_bulk, spectrum_bulk, 'r--', label = 'gold bulk')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Quotient PL/Scattering')
    plt.ylim(-0.02, 1.02)
    plt.xlim(540, 650)
    plt.legend(loc='upper right', fontsize = 'xx-small')
        
    figure_name = os.path.join(folder, '%s_quotient_PL_vs_Scattering.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close() 

    return

def compare_quotient_mean(folder, name_col, wave_bulk, spectrum_bulk):
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('data_mean_%s'%name_col,f) ]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    plt.figure()
    plt.title(name_col)

    for i in range(L):
        
        file = os.path.join(folder, list_of_folders[i])
        data = np.loadtxt(file, comments = '#')
                          
        wave = data[:, 0]
        quotient = data[:, 3]
        q = data[:,4]

        plt.plot(wave, quotient)
        plt.plot(wave, q, 'k--')  
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Quotient PL/Scattering')
        plt.ylim(-0.02, 1.02)
        
    plt.plot(wave_bulk, spectrum_bulk, 'r--', label = 'gold bulk')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Quotient PL/Scattering')
    plt.ylim(-0.02, 1.02)
    plt.xlim(540, 650)
    plt.legend(loc='upper right', fontsize = 'xx-small')
        
    figure_name = os.path.join(folder, '%s_quotient_mean_PL_vs_Scattering.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close() 

    return

def compare_product(folder, name_col, wave_bulk, spectrum_bulk):
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('data_%s'%name_col,f) ]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    plt.figure()
    plt.title('%s'%(name_col))

    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        file = os.path.join(folder, list_of_folders[i])
        data = np.loadtxt(file, comments = '#')
                          
        wave = data[:, 0]
        norm_PL = data[:, 1]
        norm_sca = data[:, 2]
        
        PL_bulk = np.interp(wave, wave_bulk, spectrum_bulk)
        product = norm_sca*PL_bulk

        plt.plot(wave, product, '-', label = 'NP_%s'%(NP))
        plt.plot(wave, norm_PL , '--', label = 'NP_%s'%(NP))
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering*PL bulk')
    plt.ylim(-0.02, 1.02)
    plt.xlim(540, 650)
    plt.legend(loc='upper right', fontsize = 'xx-small')
        
    figure_name = os.path.join(folder, '%s_product_Scattering_PLbulk.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close() 

    return

def compare_difference(folder, name_col, wave_bulk, spectrum_bulk):
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('data_%s'%name_col,f) ]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    plt.figure()
    plt.title('%s'%(name_col))

    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        file = os.path.join(folder, list_of_folders[i])
        data = np.loadtxt(file, comments = '#')
                          
        wave = data[:, 0]
        norm_PL = data[:, 1]
        norm_sca = data[:, 2]

        plt.plot(wave,  norm_sca - norm_PL , '-', label = 'NP_%s'%(NP))
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering - PL')
   # plt.ylim(-0.02, 1.02)
    plt.xlim(540, 650)
    plt.legend(loc='upper right', fontsize = 'xx-small')
        
    figure_name = os.path.join(folder, '%s_difference_Scattering_PLbulk.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close() 

    return              
                                       
if __name__ == '__main__':
    
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
        
   # daily_folder_PL = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/pre_growth/20210809-150932_Luminescence_Steps_10x12'
   # daily_folder_sca = '2021-08 (Growth PL circular)/2021-08-04 (pre growth, PL circular)/Scattering_unpol/photos'
    
    daily_folder_PL = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)/PL/20210818-174530_Luminescence_Steps_10x12'
    daily_folder_sca = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)/Scattering_unpol/photos'
     
    daily_folder = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)'
    
    save_folder = os.path.join(base_folder, daily_folder, 'fig_compare_PLconsustrato_Scattering_Unpol')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    save_folder_max = os.path.join(save_folder, 'fig_compare_lspr')
    if not os.path.exists(save_folder_max):
        os.makedirs(save_folder_max)
        
    save_folder_mean = os.path.join(save_folder, 'fig_compare_mean')
    if not os.path.exists(save_folder_mean):
        os.makedirs(save_folder_mean)
        
    save_folder_quotient = os.path.join(save_folder, 'fig_quotient_PL_Scattering')    
    if not os.path.exists(save_folder_quotient):
        os.makedirs(save_folder_quotient)
        
    list_number_col = 1,2, 3, 4,5,6,7, 8, 9,10,11,12
    number_NP = 10 #por columna
    plot_each_one = False
    
    for number_col in list_number_col:
        
        name_col = 'Col_%03d'%(number_col)
        
        print('Analize ', name_col)
        
        folder_PL = 'processed_data_luminiscence_sustrate_bkg_False'
        common_path = os.path.join(base_folder, daily_folder_PL, folder_PL, name_col)
        common_path_stokes = os.path.join(common_path, 'luminescence_steps')
    
        common_path_scattering = os.path.join(base_folder, daily_folder_sca)
        
        NP_PL_list, lspr_PL = compare_spectrum(common_path_stokes, common_path_scattering, save_folder, name_col, number_NP, plot_each_one = plot_each_one)
        compare_lspr(NP_PL_list, lspr_PL, common_path_scattering, save_folder_max, name_col)
    #    std_spectrum(common_path_stokes, common_path_scattering, save_folder, name_col, number_NP)
    
    
    plt.figure()
    for number_col in list_number_col:
        name_col = 'Col_%03d'%(number_col)
        
        file = os.path.join(save_folder_max,  'lspr_%s.txt'%(name_col))
        lspr = np.loadtxt(file, skiprows = 1)
        lspr_sca = lspr[:,1]
        lspr_PL = lspr[:,3]
        
        plt.plot(lspr_PL, lspr_sca, 'go')
        
    plt.plot(np.linspace(520, 630, 10), np.linspace(520, 630, 10), '--', color = 'grey')
    plt.ylim(540, 620)
    plt.xlim(540, 620)
    plt.ylabel('LSPR Scattering (nm)')
    plt.xlabel('LSPR PL (nm)')
    figure_name = os.path.join(save_folder_max, 'lspr.png') 
    plt.savefig(figure_name, dpi = 400)
    plt.close()

        
    #%%
        
    daily_folder_bulk = '2020-01-08 (lampara IR PySpectrum y espejo de oro)/espejo_oro/barrido_z_1um/processed_data_luminiscence_steps_and_glue/Col_001/luminescence_steps/Luminescence_Steps_Spectrum_Col_001_NP_014.txt'
    file_bulk = os.path.join(base_folder, daily_folder_bulk)
    data_bulk = np.loadtxt(file_bulk, skiprows = 1)
    wave_bulk =  data_bulk[:, 0]
    spectrum_bulk = data_bulk[:, 1]
    
    lower_lambda = 547
    upper_lambda = 625
    step = 1000

    desired_range = np.where((wave_bulk>=lower_lambda) & (wave_bulk<=upper_lambda))  
    wavelength_bulk = wave_bulk[desired_range]
    intensity_bulk = spectrum_bulk[desired_range]
    
    spectrum_bulk = (intensity_bulk - min(intensity_bulk ))/(max(intensity_bulk) - min(intensity_bulk ))
        
    for number_col in list_number_col:  
        name_col = 'Col_%03d'%(number_col)  
        compare_quotient(save_folder_quotient, name_col, wavelength_bulk, spectrum_bulk)
        compare_quotient_mean(save_folder_quotient, name_col, wavelength_bulk, spectrum_bulk)
   #     compare_product(save_folder_quotient, name_col, wavelength_bulk, spectrum_bulk)
   #     compare_difference(save_folder_quotient, name_col, wavelength_bulk, spectrum_bulk)