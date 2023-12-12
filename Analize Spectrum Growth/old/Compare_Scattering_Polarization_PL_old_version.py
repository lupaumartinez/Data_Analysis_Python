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

from scipy import signal

def compare_spectrum(common_path_sca_px, common_path_sca_py, save_folder, name_col, normalized_sca):  
            
    save_folder_col = os.path.join(save_folder, name_col)
    if not os.path.exists(save_folder_col):
        os.makedirs(save_folder_col)
    
  #  list_of_folders_scattering = os.listdir(common_path_sca)
  #  list_of_folders_scattering = [f for f in list_of_folders_scattering if re.search('NP',f)]
  #  list_of_folders_scattering.sort()
  #  M = len(list_of_folders_scattering)  
    
  #  print('Files from Scattering', M, list_of_folders_scattering)
    
    list_of_folders_scattering_px = os.listdir(common_path_sca_px)
    list_of_folders_scattering_px = [f for f in list_of_folders_scattering_px if re.search('NP',f)]
    list_of_folders_scattering_px.sort()
    M_px = len(list_of_folders_scattering_px)  
    
    print('Files from Scattering PX', M_px, list_of_folders_scattering_px)
    
    list_of_folders_scattering_py = os.listdir(common_path_sca_py)
    list_of_folders_scattering_py = [f for f in list_of_folders_scattering_py if re.search('NP',f)]
    list_of_folders_scattering_py.sort()
    M_py = len(list_of_folders_scattering_py)  
    
    print('Files from Scattering PY', M_py, list_of_folders_scattering_py)
    
    if not M_py == M_px: # or not M == M_py:
        
        print('The files do not have the same NP. Check files')
        
   # L = np.min([M, M_px, M_py])
   
    L = np.min([M_px, M_py])
        
    NP_list = np.zeros(L)
  #  max_wavelength_sca = np.zeros(L)
    max_wavelength_sca_px = np.zeros(L)
    max_wavelength_sca_py = np.zeros(L)
    
    for i in range(L):
        
      #  NP = list_of_folders_scattering[i].split('NP_')[-1]
      #  NP = NP.split('.')[0]
        
       # name_file = os.path.join(common_path_sca,  list_of_folders_scattering[i])
        
       # a = np.loadtxt(name_file, skiprows=1)
       # wavelength_sca = a[:, 0]
       # scattering = a[:, 1]
   
        name_file_sca_px = os.path.join(common_path_sca_px,  list_of_folders_scattering_px[i])
        name_file_sca_py = os.path.join(common_path_sca_py,  list_of_folders_scattering_px[i])
        
        NP = list_of_folders_scattering_px[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        print('Ploteo NP_%s'%(NP))
        
        b = np.loadtxt(name_file_sca_px, skiprows=1)
        wavelength_sca_px = b[:, 0]
        scattering_px = b[:, 1]
        
        c = np.loadtxt(name_file_sca_py, skiprows=1)
        wavelength_sca_py = c[:, 0]
        scattering_py = c[:, 1]
        
    #    scattering_not_norm  = a[:, 2]  
        scattering_px_not_norm = b[:, 2]
        scattering_py_not_norm = c[:, 2]
     #   scattering_mean  = (scattering_px_not_norm + scattering_py_not_norm)/2
        
     #   sca = (scattering-min(scattering))/(max(scattering)-min(scattering))
        sca_px = (scattering_px-min(scattering_px))/(max(scattering_px)-min(scattering_px))
        sca_py = (scattering_py-min(scattering_py))/(max(scattering_py)-min(scattering_py))
     #  sca_mean = (scattering_mean-min(scattering_mean))/(max(scattering_mean)-min(scattering_mean))
        
        NP_list[i] = NP
    #    max_wavelength_sca[i] = wavelength_sca[np.argmax(sca)]
        max_wavelength_sca_px[i] = wavelength_sca_px[np.argmax(sca_px)]
        max_wavelength_sca_py[i] = wavelength_sca_py[np.argmax(sca_py)]
        
        plt.figure()
        
        plt.title('NP_%s'%(NP))
        
        if normalized_sca:
            
       #     plt.plot(wavelength_sca, sca, label = 'Scattering unpol')
            plt.plot(wavelength_sca_px, sca_px, label = 'Scattering PX')
            plt.plot(wavelength_sca_py, sca_py, label = 'Scattering PY')
         #   plt.plot(wavelength_sca_px, sca_mean , label  = 'Scattering mean PX, PY')
            
        else:
            
        #    plt.plot(wavelength_sca, scattering_not_norm, label = 'Scattering unpol')
            plt.plot(wavelength_sca_px, scattering_px_not_norm, label = 'Scattering PX')
            plt.plot(wavelength_sca_py, scattering_py_not_norm, label = 'Scattering PY')
         #   plt.plot(wavelength_sca_px, scattering_mean , label  = 'Scattering mean PX, PY')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right') 
        
        #plt.xlim(500, 1000)
       # plt.ylim(0.5, 1.05)
        
        figure_name = os.path.join(save_folder_col, 'compare_Scattering_Polarization_%s_NP_%s.png'%(name_col, NP)) 
        
        #plt.show()
        plt.savefig(figure_name , dpi = 400)
        plt.close()
       
    plt.figure()
    plt.title('%s'%(name_col))
   # plt.plot(NP_list, max_wavelength_sca, 'o', label = 'Scattering')
    plt.plot(NP_list, max_wavelength_sca_px, 'o', label = 'Scattering PX')
    plt.plot(NP_list, max_wavelength_sca_py, 'o', label = 'Scattering PY')
    plt.xlabel('NP')
    plt.ylabel('Max wavelength (nm)')
    plt.ylim(530, 650)
    plt.legend()
    figure_name = os.path.join(save_folder, 'max_wavelength_Scattering_%s.png'%(name_col)) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    name = os.path.join(save_folder_col,  'max_wavelength_PL_vs_Scattering_%s.txt'%(name_col))
    data = np.array([NP_list, max_wavelength_sca_px, max_wavelength_sca_py]).T
    header_txt = 'NP, max wavelength Sca PX (nm), max wavelength PY (nm)'
    np.savetxt(name, data, header = header_txt)
                    
    return


def compare_spectrum_with_PL(common_path_PL, common_path_sca, common_path_sca_px, common_path_sca_py, save_folder, name_col, analize_quotient):
            
    save_folder_col = os.path.join(save_folder, name_col)
    if not os.path.exists(save_folder_col):
        os.makedirs(save_folder_col)
            
    list_of_folders = os.listdir(common_path_PL)
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    
    list_of_folders.sort()
    L1 = len(list_of_folders)
    print('Files from PL', L1, list_of_folders)
    
    list_of_folders_scattering = os.listdir(common_path_sca)
    list_of_folders_scattering = [f for f in list_of_folders_scattering if re.search('NP',f)]
    list_of_folders_scattering.sort()
    M = len(list_of_folders_scattering)  
    
    print('Files from Scattering', M, list_of_folders_scattering)
    
    list_of_folders_scattering_px = os.listdir(common_path_sca_px)
    list_of_folders_scattering_px = [f for f in list_of_folders_scattering_px if re.search('NP',f)]
    list_of_folders_scattering_px.sort()
    M_px = len(list_of_folders_scattering_px)  
    
    print('Files from Scattering PX', M_px, list_of_folders_scattering_px)
    
    list_of_folders_scattering_py = os.listdir(common_path_sca_py)
    list_of_folders_scattering_py = [f for f in list_of_folders_scattering_py if re.search('NP',f)]
    list_of_folders_scattering_py.sort()
    M_py = len(list_of_folders_scattering_py)  
    
    print('Files from Scattering PY', M_py, list_of_folders_scattering_py)
    
    L = np.min([L1, M, M_px, M_py])
        
    NP_list = np.zeros(L)
    max_wavelength_PL = np.zeros(L)
    max_wavelength_sca = np.zeros(L)
    max_wavelength_sca_px = np.zeros(L)
    max_wavelength_sca_py = np.zeros(L)
    max_wavelength_sca_mean = np.zeros(L)
    
    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        number_NP = int(NP.split('_')[-1][-1])
        file_scattering = 'NP_%03d.txt'%(number_NP)
    
      #  NP_sca = list_of_folders_scattering[i].split('NP _')[-1]
      #  NP_sca = NP_sca.split('.')[0]
      
        print('Ploteo NP_%s'%(NP))
         
        name_file = os.path.join(common_path_PL,  list_of_folders[i])
        a = np.loadtxt(name_file, skiprows=1)
        wavelength = a[:, 0]
        luminiscence = a[:, 1]
        
        PL = (luminiscence-min(luminiscence))/(max(luminiscence)-min(luminiscence))
         
        name_file_sca = os.path.join(common_path_sca,  file_scattering)
        
        a = np.loadtxt(name_file_sca, skiprows=1)
        wavelength_sca = a[:, 0]
        scattering = a[:, 1]
   
        name_file_sca_px = os.path.join(common_path_sca_px,  file_scattering)
        name_file_sca_py = os.path.join(common_path_sca_py,  file_scattering)
        
        b = np.loadtxt(name_file_sca_px, skiprows=1)
        wavelength_sca_px = b[:, 0]
        scattering_px = b[:, 1]
        
        c = np.loadtxt(name_file_sca_py, skiprows=1)
        wavelength_sca_py = c[:, 0]
        scattering_py = c[:, 1]
        
        scattering_not_norm  = a[:, 2]  
        scattering_px_not_norm = b[:, 2]
        scattering_py_not_norm = c[:, 2]
        scattering_mean  = (scattering_px_not_norm + scattering_py_not_norm)/2
        
        sca = (scattering-min(scattering))/(max(scattering)-min(scattering))
        sca_px = (scattering_px-min(scattering_px))/(max(scattering_px)-min(scattering_px))
        sca_py = (scattering_py-min(scattering_py))/(max(scattering_py)-min(scattering_py))
        sca_mean = (scattering_mean-min(scattering_mean))/(max(scattering_mean)-min(scattering_mean))
        
        NP_list[i] = number_NP
        max_wavelength_PL[i] = wavelength[np.argmax(PL)]
        max_wavelength_sca[i] = wavelength_sca[np.argmax(sca)]
        max_wavelength_sca_px[i] = wavelength_sca_px[np.argmax(sca_px)]
        max_wavelength_sca_py[i] = wavelength_sca_py[np.argmax(sca_py)]
        max_wavelength_sca_mean[i] = wavelength_sca_px[np.argmax(sca_mean)]
        
        plt.figure()
        
        plt.title('NP_%s'%(NP))
        plt.plot(wavelength, PL , 'k--', label = 'PL')
        plt.plot(wavelength_sca, sca, label = 'Scattering')
        plt.plot(wavelength_sca_px, sca_px, label = 'Scattering PX')
        plt.plot(wavelength_sca_py, sca_py, label = 'Scattering PY')
        plt.plot(wavelength_sca_px, sca_mean , label  = 'Scattering mean PX, PY')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right') 
        
        #plt.xlim(500, 1000)
       # plt.ylim(0.5, 1.05)
        
        figure_name = os.path.join(save_folder_col, 'compare_PL_Scattering_Polarization_%s_NP_%s.png'%(name_col, NP)) 
        
        #plt.show()
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        if analize_quotient:
            save_folder_2 = os.path.join(save_folder, 'fig_quotient_PL_Scattering')
            quotient_PL_Scattering(save_folder_2, name_col, NP, 'Scattering_unpol', wavelength, PL,  wavelength_sca_px, sca_py)
            quotient_PL_Scattering(save_folder_2, name_col, NP, 'Scattering_mean',wavelength, PL, wavelength_sca_px, sca_mean)
           
    plt.figure()
    plt.title('%s'%(name_col))
    plt.plot(NP_list, max_wavelength_PL, 'ko', label = 'PL')
    plt.plot(NP_list, max_wavelength_sca, 'o', label = 'Scattering')
    plt.plot(NP_list, max_wavelength_sca_px, 'o', label = 'Scattering PX')
    plt.plot(NP_list, max_wavelength_sca_py, 'o', label = 'Scattering PY')
    plt.plot(NP_list, max_wavelength_sca_mean, 'o', label  = 'Scattering mean PX, PY')
    plt.xlabel('NP')
    plt.ylabel('Max wavelength (nm)')
    plt.ylim(530, 650)
    plt.legend()
    figure_name = os.path.join(save_folder, 'max_wavelength_PL_vs_Scattering_%s.png'%(name_col)) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    name = os.path.join(save_folder_col,  'max_wavelength_PL_vs_Scattering_%s.txt'%(name_col))
    data = np.array([NP_list, max_wavelength_PL, max_wavelength_sca, max_wavelength_sca_px, max_wavelength_sca_py, max_wavelength_sca_mean]).T
    header_txt = 'NP, max wavelength PL (nm), max wavelength Scattering (nm), max wavelength Sca PX (nm), max wavelength PY (nm), max wavelength Sca mean (nm)'
    np.savetxt(name, data, header = header_txt)
                    
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

def plot_more_analysis(save_folder, name_col):
    
    folder = os.path.join(save_folder, name_col)
    
    name_file = os.path.join(folder, 'max_wavelength_PL_vs_Scattering_%s.txt'%name_col)
    a = np.loadtxt(name_file, skiprows=1)
    NP = a[:, 0]
    max_wavelength_PL = a[:, 1]
    max_wavelength_sca = a[:, 2]
    max_wavelength_sca_PX = a[:, 3]
    max_wavelength_sca_PY = a[:, 4]
    max_wavelength_sca_mean = a[:, 5]
    
    means, sigmas = plot_histogram(save_folder, name_col, max_wavelength_PL, max_wavelength_sca, max_wavelength_sca_PX, max_wavelength_sca_PY, max_wavelength_sca_mean)
    
    plot_ratio(save_folder, name_col, NP, max_wavelength_PL, max_wavelength_sca, max_wavelength_sca_PX, max_wavelength_sca_PY, max_wavelength_sca_mean)
    
    return means, sigmas
    
    
def plot_histogram(save_folder, name_col, max_wavelength_PL, max_wavelength_sca, max_wavelength_sca_PX, max_wavelength_sca_PY, max_wavelength_sca_mean):
    
    print('Plot Histogram ', name_col)   
    
    lim_range = [500,700]
    bins = 20
    
    f, ((ax1, ax_empty, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10,5))
    
    n,bin_positions_x,p = ax1.hist(max_wavelength_PL, bins=bins, range=lim_range, density=True, rwidth=0.9, color='K')
    mu_PL, sigma_PL, x_text = data_hist(max_wavelength_PL, bin_positions_x)
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax1.text(510, 0.06, x_text, fontsize = 'xx-small')
    ax1.set_xlabel('PL')
    ax1.set_ylim(0, 0.08)
    ax1.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax2.hist(max_wavelength_sca, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C0')
    mu_sca, sigma_sca, x_text = data_hist(max_wavelength_sca, bin_positions_x)
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax2.text(510, 0.06, x_text, fontsize = 'xx-small')
    ax2.set_xlabel('Sca unpol')
    ax2.set_ylim(0, 0.08)
    ax2.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax3.hist(max_wavelength_sca_PX, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C1')
    mu_sca_PX, sigma_sca_PX, x_text = data_hist(max_wavelength_sca_PX, bin_positions_x)
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax3.text(510, 0.06, x_text, fontsize = 'xx-small')
    ax3.set_xlabel('Sca PX')
    ax3.set_ylim(0, 0.08)
    ax3.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax4.hist(max_wavelength_sca_PY, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C2')
    mu_sca_PY, sigma_sca_PY, x_text = data_hist(max_wavelength_sca_PY, bin_positions_x)
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax4.text(510, 0.06, x_text, fontsize = 'xx-small')
    ax4.set_xlabel('Sca PY')
    ax4.set_ylim(0, 0.08)
    ax4.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax5.hist(max_wavelength_sca_mean, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C3')
    mu_sca_mean, sigma_sca_mean, x_text = data_hist(max_wavelength_sca_mean, bin_positions_x)
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax5.text(510, 0.06, x_text, fontsize = 'xx-small')
    ax5.set_xlabel('Sca mean PX, PY')
    ax5.set_ylim(0, 0.08)
    ax5.set_xlim(lim_range[0],lim_range[1])
    
    f.set_tight_layout(True)

    figure_name = os.path.join(save_folder, 'histogram_max_wavelength_%s.png'%(name_col)) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    name = os.path.join(save_folder, name_col, 'data_hist_max_wavelength_%s.txt'%(name_col))
    data = np.array([mu_PL, sigma_PL, mu_sca, sigma_sca, mu_sca_PX, sigma_sca_PX, mu_sca_PY, sigma_sca_PY, mu_sca_mean, sigma_sca_mean])
    header_txt = 'mean PL, std PL, mean Sca, std Sca, mean Sca PX, std Sca PX, mean Sca PY, std Sca PY, mean Sca mean, std Sca mean'
    np.savetxt(name, data, header = header_txt)
    
    means = mu_PL, mu_sca, mu_sca_PX, mu_sca_PY, mu_sca_mean
    sigmas = sigma_PL, sigma_sca, sigma_sca_PX, sigma_sca_PY, sigma_sca_mean
    
    return means, sigmas

def plot_ratio(save_folder, name_col, NP, max_wavelength_PL, max_wavelength_sca, max_wavelength_PX, max_wavelength_PY, max_wavelength_sca_mean):
    
    ratio_scattering = max_wavelength_PX/max_wavelength_PY
    
    ratio_PL_scaPY = max_wavelength_PL/max_wavelength_PY
    
    shift_scattering = (max_wavelength_PX - max_wavelength_PY)
    shift_PL_sca_PY = (max_wavelength_PL - max_wavelength_PY)
    shift_PL_sca_PX = (max_wavelength_PL - max_wavelength_PX)
    shift_PL_sca = max_wavelength_PL - max_wavelength_sca
    
    plt.figure()
    plt.plot(NP, ratio_scattering, 'ro', label = 'Sca PX/PY')
    plt.plot(NP, ratio_PL_scaPY, 'ko',label = 'PL/Sca PY')
    plt.xlabel('NP')
    plt.ylabel('Ratio')
    plt.legend()
    
    figure_name = os.path.join(save_folder, 'ratio_max_wavelength_%s.png'%(name_col)) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    plt.figure()
    plt.plot(NP, shift_scattering, 'o', label = 'Sca PX - Sca PY')
    plt.plot(NP, shift_PL_sca, 'o',label = 'PL - Sca unpol')
    plt.plot(NP, shift_PL_sca_PX, 'o',label = 'PL - Sca PX')
    plt.plot(NP, shift_PL_sca_PY, 'o',label = 'PL - Sca PY')
    plt.axhline(0, linestyle = '--', color = 'grey')
    plt.xlabel('NP')
    plt.ylabel('Shift LSPR (nm)')
    plt.legend()
    
    figure_name = os.path.join(save_folder, 'shift_max_wavelength_%s.png'%(name_col)) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    return

def quotient_PL_Scattering(save_folder, name_col, NP, name_scattering, wavelength_PL, intensity_PL, wavelength_sca, intensity_sca):

    lower_lambda = 547
    upper_lambda = 750
    step = 1000
    
    wavelength_new = np.linspace(lower_lambda, upper_lambda, step)

    desired_range = np.where((wavelength_PL>=lower_lambda) & (wavelength_PL<=upper_lambda))
    
    wavelength_PL = wavelength_PL[desired_range]
    intensity_PL = intensity_PL[desired_range]

    new_PL = np.interp(wavelength_new, wavelength_PL, intensity_PL)
    new_sca = np.interp(wavelength_new, wavelength_sca, intensity_sca)
    
    quotient = new_PL/new_sca
    quotient = (quotient-min(quotient))/(max(quotient)-min(quotient))
    
    window_smooth = 51
    q = signal.savgol_filter(quotient, window_smooth, 1, mode = 'mirror')
    
    plt.figure()
    plt.title('NP_%s'%(NP))
    
  #  plt.plot(wavelength_new, new_PL)
  #  plt.plot(wavelength_new, new_sca)
    
    plt.plot(wavelength_new, quotient)
    plt.plot(wavelength_new, q, 'k--')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Quotient PL/Scattering')
    plt.ylim(-0.02, 1.02)
        
    figure_name = os.path.join(save_folder, 'fig_quotient_PL_%s'%name_scattering, 'quotient_PL_vs_Scattering_%s_NP_%s.png'%(name_col, NP))
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    name = os.path.join(save_folder, 'fig_quotient_PL_%s'%name_scattering, 'data_%s_NP_%s.txt'%(name_col, NP))
    data = np.array([wavelength_new, new_PL, new_sca, quotient, q]).T
    header_txt = 'wavelength (nm), PL norm, Scattering norm, Quotient PL/Sca norm, Quotient PL/Sca smooth'
    np.savetxt(name, data, header = header_txt)

    return

def compare_quotient(folder, name_col, wave_bulk, spectrum_bulk):
    
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
        quotient = data[:, 3]
        q = data[:,4]

        plt.plot(wave, quotient, label = 'NP_%s'%(NP))
        plt.plot(wave, q, 'k--')
        
    plt.plot(wave_bulk, spectrum_bulk, 'r--', label = 'gold bulk')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Quotient PL/Scattering')
    plt.ylim(-0.02, 1.02)
    plt.xlim(540, 840)
    plt.legend(loc='upper right', fontsize = 'xx-small')
        
    figure_name = os.path.join(folder, '%s_quotient_PL_vs_Scattering.png'%(name_col))
    plt.savefig(figure_name , dpi = 400)
    plt.close()            

#%%
         
if __name__ == '__main__':
    
    #base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PySpectrum/'
    #daily_folder = '2020-02-07 (Scattering AuNPz 60 nm growth)'
    
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
    daily_folder = '2021-03-26 (Growth)/Scattering_growth/'
    
    save_folder = os.path.join(base_folder, daily_folder, 'fig_compare_Scattering_Polarization')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    list_number_col = 1,2,3,4,5,6,7,8,9,10
    
    compare_PL = False
    daily_folder_PL = '2020-02-06 (AuNPz 60 nm growth)/20200206-165513_Luminescence_Steps_10x8'
 #   daily_folder_PL ='2020-02-06 (AuNPz 60 nm growth)/fila_destruccion_scan_min_Col_001_scan_max_Col_002'
    daily_folder_PL = os.path.join(daily_folder_PL, 'processed_data_luminiscence_sustrate_bkg_True')
    
    analize_quotient = False #quotient PL/Scattering
   
    if compare_PL and analize_quotient:
        
        daily_folder_bulk = '2020-01-08 (lampara IR PySpectrum y espejo de oro)/espejo_oro/barrido_z_1um/processed_data_luminiscence_steps_and_glue/Col_001/luminescence_steps/Luminescence_Steps_Spectrum_Col_001_NP_014.txt'
        file_bulk = os.path.join(base_folder, daily_folder_bulk)
        data_bulk = np.loadtxt(file_bulk, skiprows = 1)
        wave_bulk =  data_bulk[:, 0]
        spectrum_bulk = data_bulk[:, 1]
        spectrum_bulk = spectrum_bulk/max(spectrum_bulk)
        
        save_folder_2 = os.path.join(save_folder, 'fig_quotient_PL_Scattering')    
        if not os.path.exists(save_folder_2):
            os.makedirs(save_folder_2)
        
        save_folder_quotient = os.path.join(save_folder_2, 'fig_quotient_PL_Scattering_unpol')    
        if not os.path.exists(save_folder_quotient):
            os.makedirs(save_folder_quotient)
            
        save_folder_quotient_PX = os.path.join(save_folder_2, 'fig_quotient_PL_Scattering_PX')    
        if not os.path.exists(save_folder_quotient_PX):
            os.makedirs(save_folder_quotient_PX)
            
        save_folder_quotient_PY = os.path.join(save_folder_2, 'fig_quotient_PL_Scattering_PY')    
        if not os.path.exists(save_folder_quotient_PY):
            os.makedirs(save_folder_quotient_PY)
    
        save_folder_quotient_mean = os.path.join(save_folder_2, 'fig_quotient_PL_Scattering_mean')    
        if not os.path.exists(save_folder_quotient_mean):
            os.makedirs(save_folder_quotient_mean)
    
    for number_col in list_number_col:
    
        name_col = 'Col_%03d'%(number_col)
        
        print('Analize ', name_col)
        
      #  daily_folder_sca = 'lampara_IR_unpol_1s/photos_not_normalized/Col_%03d/normalized_Col_%03d'%(number_col,number_col)
      #  daily_folder_sca_px = 'lampara_IR_PX_4s/photos_not_normalized/Col_%03d/normalized_Col_%03d'%(number_col,number_col)
      #  daily_folder_sca_py = 'lampara_IR_PY_4s/photos_not_normalized/Col_%03d/normalized_Col_%03d'%(number_col,number_col)
        
      #  daily_folder_sca = 'lampara_IR_unpol_1s/photos_not_normalized/fila_destruccion/Col_%03d/normalized_Col_%03d'%(number_col,number_col)
      #  daily_folder_sca_px = 'lampara_IR_PX_4s/photos_not_normalized/fila_destruccion/Col_%03d/normalized_Col_%03d'%(number_col,number_col)
      #  daily_folder_sca_py = 'lampara_IR_PY_4s/photos_not_normalized/fila_destruccion/Col_%03d/normalized_Col_%03d'%(number_col,number_col)
        
        daily_folder_sca_px = 'PX/photos/Col_%03d/normalized_Col_%03d'%(number_col,number_col)
        daily_folder_sca_py = 'PY/photos/Col_%03d/normalized_Col_%03d'%(number_col,number_col)
      
      #  common_path_sca = os.path.join(base_folder, daily_folder, daily_folder_sca)
        common_path_sca_px = os.path.join(base_folder, daily_folder, daily_folder_sca_px)
        common_path_sca_py = os.path.join(base_folder, daily_folder, daily_folder_sca_py)
            
        if not compare_PL:
            
         #   compare_spectrum(common_path_sca, common_path_sca_px, common_path_sca_py, save_folder, name_col, normalized_sca = True)
         compare_spectrum(common_path_sca_px, common_path_sca_py, save_folder, name_col, normalized_sca = True)
                  
        if compare_PL:       
            
            common_path_PL = os.path.join(base_folder, daily_folder_PL, name_col, 'luminescence_steps')
            compare_spectrum_with_PL(common_path_PL, common_path_sca, common_path_sca_px, common_path_sca_py, save_folder, name_col, analize_quotient)
            
            means, sigmas = plot_more_analysis(save_folder, name_col)
            
            if analize_quotient:

                compare_quotient(save_folder_quotient, name_col, wave_bulk, spectrum_bulk)
                compare_quotient(save_folder_quotient_PX, name_col, wave_bulk, spectrum_bulk)
                compare_quotient(save_folder_quotient_PY, name_col, wave_bulk, spectrum_bulk)
                compare_quotient(save_folder_quotient_mean, name_col, wave_bulk, spectrum_bulk)
                
 #%%   
        
    list_number_col = 4,5,7
    input_wavelength_PySpectrum = 580, 570, 545 #nm
    
    input_correlation_nm = dict(zip(list_number_col, input_wavelength_PySpectrum))
    
    x = np.linspace(540, 600, 10)
    
    if compare_PL:
        
        plt.figure()
       
        for number_col in list_number_col:
            
            name_col = 'Col_%03d'%(number_col)
            means, sigmas = plot_more_analysis(save_folder, name_col)
            
            input_wavelength = input_correlation_nm[number_col]
            
            plt.errorbar(input_wavelength, means[0], yerr = sigmas[0], marker = 'o', linestyle = 'None', color = 'K', capsize = 3)
            plt.errorbar(input_wavelength, means[1], yerr = sigmas[1], marker = 'o', linestyle = 'None', color = 'C0', capsize = 3)
            plt.errorbar(input_wavelength, means[2], yerr = sigmas[2], marker = 'o', linestyle = 'None', color = 'C1', capsize = 3)
            plt.errorbar(input_wavelength, means[3], yerr = sigmas[3], marker = 'o', linestyle = 'None', color = 'C2', capsize = 3)
            plt.errorbar(input_wavelength, means[4], yerr = sigmas[4], marker = 'o', linestyle = 'None', color = 'C3', capsize = 3)
            
            plt.plot(x,x, linestyle = '--', color = 'grey')
            
        #plt.xlabel('Col')
        plt.xlabel('Input Wavelength (nm)')
        plt.ylabel('Mean Max wavelength (nm)')
            
        figure_name = os.path.join(save_folder, 'all_means_max_wavelength_control_PL.png') 
        plt.savefig(figure_name, dpi = 400)
        plt.close()
        
#%%
        
    list_number_col = 1, 3, 7, 8, 4, 5
    input_time_PySpectrum = 300, 100, 0, 0, 40, 33
    
    input_correlation_s = dict(zip(list_number_col, input_time_PySpectrum))
    
    if compare_PL:
        
        plt.figure()
       
        for number_col in list_number_col:
            
            name_col = 'Col_%03d'%(number_col)
            means, sigmas = plot_more_analysis(save_folder, name_col)
            
            input_time = input_correlation_s[number_col]
            
            plt.errorbar(input_time, means[0], yerr = sigmas[0], marker = 'o', linestyle = 'None', color = 'K', capsize = 3)
            plt.errorbar(input_time, means[1], yerr = sigmas[1], marker = 'o', linestyle = 'None', color = 'C0', capsize = 3)
            plt.errorbar(input_time, means[2], yerr = sigmas[2], marker = 'o', linestyle = 'None', color = 'C1', capsize = 3)
            plt.errorbar(input_time, means[3], yerr = sigmas[3], marker = 'o', linestyle = 'None', color = 'C2', capsize = 3)
            plt.errorbar(input_time, means[4], yerr = sigmas[4], marker = 'o', linestyle = 'None', color = 'C3', capsize = 3)
            
        #plt.xlabel('Col')
        plt.xlabel('Input time (s)')
        plt.ylabel('Mean Max wavelength (nm)')
            
        figure_name = os.path.join(save_folder, 'all_means_max_wavelength_control_time.png') 
        plt.savefig(figure_name, dpi = 400)
        plt.close()