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

from Fit_raman_water import fit_signal_raman

def compare_final_spectrum(common_path_growth, common_path_PL, save_folder, name_col):
    
    common_path_PL = os.path.join(common_path_PL, name_col)
    
    common_path_PL_growth = os.path.join(common_path_growth, 'all_data_final_Stokes')
    
    list_of_folders = os.listdir(common_path_PL_growth)
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Files from final growth PL', L, list_of_folders)
    
    common_path_PL = os.path.join(common_path_PL, 'luminescence_steps')
    
    list_of_folders_PL= os.listdir(common_path_PL)
    list_of_folders_PL = [f for f in list_of_folders_PL if re.search('NP',f)]
    list_of_folders_PL.sort()
    M = len(list_of_folders_PL)  
    
    print('Files from PL steps', M, list_of_folders_PL)
    
    if L == M:
    
        print('Plot final grwoth PL, compare with PL steps')
        
    else:
        
        print('The files do not have the same NP. Check files')
        
    L = np.min([L,M])
    
    NP_growth = []
    lspr_growth = []
    
    NP_PL_list = []
    lspr_PL = []
                
    for i in range(11): #for i in range(11,23,1):
        
        NP = list_of_folders[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        NP_PL = list_of_folders_PL[i].split('NP_')[-1] #list_of_folders_PL[i-11].split('NP_')[-1]
        NP_PL = NP_PL.split('.')[0]
        
      
        print('Ploteo: from final PL grwoth NP_%s, from final PL steps  NP_%s'%(NP, NP_PL))
         
        ## PL of Growth in live
        
        name_file = os.path.join(common_path_PL_growth,  list_of_folders[i])
        
        a = np.loadtxt(name_file, skiprows=1)
        wavelength = a[:, 0]
        final_luminiscence = a[:, 1] #con sustrato
        fit_final_luminiscence = a[:, 2] #con sustrato
        
        PL_final_growth = (final_luminiscence-min(final_luminiscence))/(max(final_luminiscence)-min(final_luminiscence))
        
        fit_wavelength_growth, fit_spectrum_NP_growth, best_parameters_growth = fit_signal_raman(wavelength, final_luminiscence, 540, 650)
        fit_PL_final_growth = (fit_spectrum_NP_growth-min(fit_spectrum_NP_growth))/(max(fit_spectrum_NP_growth)-min(fit_spectrum_NP_growth))
    
        londa_spr_growth = np.round(best_parameters_growth[2],2)
        
        ## PL of steps
    
        name_file_PL = os.path.join(common_path_PL,  list_of_folders_PL[i])  #list_of_folders_PL[i-11]
        
        b = np.loadtxt(name_file_PL, skiprows=1)
        wavelength_PL= b[:, 0]
        luminiscence_NP = b[:, 1]
        luminiscence = b[:, 2]  #con sustrato

        wavelength_PL = np.array(wavelength_PL)
        roi_stokes = np.where((wavelength_PL >= wavelength[0]) & (wavelength_PL <= wavelength[-1]))
        wavelength_PL_stokes = wavelength_PL[roi_stokes]
        luminiscence_stokes = luminiscence[roi_stokes]
        
        PL_steps_stokes = (luminiscence_stokes-min(luminiscence_stokes))/(max(luminiscence_stokes)-min(luminiscence_stokes))
        PL_steps = (luminiscence-min(luminiscence_stokes))/(max(luminiscence_stokes)-min(luminiscence_stokes))
        
        fit_wavelength_PL, fit_spectrum_NP_PL, best_parameters_PL = fit_signal_raman(wavelength_PL, luminiscence, 540, 650)
        fit_PL = (fit_spectrum_NP_PL-min(luminiscence_stokes))/(max(fit_spectrum_NP_PL)-min(luminiscence_stokes))
        
        londa_spr_PL = np.round(best_parameters_PL[2],2)
        
        #PLoteo
        
        plt.figure()
        
        plt.title('NP_%s'%(NP))
        plt.plot(wavelength, PL_final_growth , label = 'growth final PL')
        plt.plot(fit_wavelength_growth, fit_PL_final_growth , 'r--', label = 'fit growth final PL')
        
        plt.plot(wavelength_PL, PL_steps, 'k', label = 'PL steps')
        plt.plot(wavelength_PL_stokes, PL_steps_stokes)
        plt.plot(fit_wavelength_PL, fit_PL, 'g--', label = 'fit PL steps')
        
        #label = 'PL steps', )
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right') 
        
        #plt.xlim(500, 1000)
       # plt.ylim(0.5, 1.05)
        
        figure_name = os.path.join(save_folder, 'compare_final_PL_%s_NP_%s.png'%(name_col, NP_PL)) 
        
        #plt.show()
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        #data save
        NP_growth.append(int(NP))
        lspr_growth.append(londa_spr_growth)
        
        NP_PL_list.append(int(NP_PL))
        lspr_PL.append(londa_spr_PL)
                    
    return NP_growth, lspr_growth, NP_PL_list, lspr_PL
   
def fit_signal(wavelength, spectrum_NP, lower_wavelength, upper_wavelength):

    fit_wavelength, fit_spectrum_NP, best_parameters = fit_signal_raman(wavelength, spectrum_NP, lower_wavelength, upper_wavelength)
    
    londa_spr = np.round(best_parameters[2],2)
    
    return fit_wavelength, fit_spectrum_NP, londa_spr


def compare_lspr(NP_growth, lspr_growth, NP_PL_list, lspr_PL, common_path_growth, common_path_PL, save_folder, name_col):
    
    name_file = os.path.join(common_path_growth,  'all_data_growth', 'all_data_growth.txt')
    b = np.loadtxt(name_file, skiprows=1)
    NP = b[:, 0]
    total_time = b[:, 3]
    final_lspr_growth = b[:, 4]

    NP = NP[:11] #NP[11:]
    final_lspr_growth =  final_lspr_growth[:11] #final_lspr_growth[11:]
    total_time = total_time[:11] #total_time[11:]
    
    name_file_2 = os.path.join(common_path_PL, 'fig_fit_all_data', 'fit_wavelength_lspr_Col_%03d_01.txt'%(number_col))
    b2 = np.loadtxt(name_file_2, skiprows=1)
    NP_PL = b2[1:, 0] #b2[:, 0]
    lspr_mode1 = b2[1:, 1] #b2[:, 1]
    
    plt.figure()
    plt.plot(NP, final_lspr_growth, 'o', label = 'lspr growth live')
    plt.plot(NP_growth, lspr_growth, 'o', label = 'lspr growth post analysis')
    plt.plot(NP_growth, lspr_PL, 'o', label = 'lspr PL steps analysis one mode')
    plt.plot(NP_growth, lspr_mode1, 'o', label = 'lspr PL steps analysis two modes')
    
    plt.xlabel('NP')
    plt.ylabel('Final LSPR')
    plt.legend(loc='lower right') 
    
    #plt.xlim(500, 1000)
    plt.ylim(540, 600)
    
    figure_name = os.path.join(save_folder, 'final_lspr_PL_%s.png'%(name_col)) 
    
    #plt.show()
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
   # plot_time(total_time_growth, final_lspr_growth, lspr_PL, save_folder, name_col)
   
    name = os.path.join(save_folder,  'lspr_fit_PL_%s.txt'%(name_col))
    data = np.array([NP, final_lspr_growth, total_time , lspr_growth, lspr_PL, lspr_mode1]).T
    header_txt = 'NP, lspr growth live (nm), Total time (s), lspr growth post analysis (nm), lspr PL steps analysis one mode (nm), lspr PL steps analysis two modes (nm)'
    np.savetxt(name, data, header = header_txt)
    
    return total_time, final_lspr_growth, lspr_PL

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


def plot_more_analysis_all(save_folder, name_col):
    
    name_file = os.path.join(save_folder, 'lspr_fit_PL_%s.txt'%name_col)
    a = np.loadtxt(name_file, skiprows=1)
    
    NP = a[:, 0]
    
    final_lspr_growth = a[:, 1]
    total_time = a[:, 2]
    lspr_growth = a[:, 3]
    
    lspr_PL = a[:, 4]
    lspr_mode1 = a[:, 5]

    means, sigmas = plot_histogram_all(save_folder, final_lspr_growth, lspr_growth, lspr_PL, lspr_mode1, 'histogram_lspr_fit_PL_%s'%(name_col))
    
    return means, sigmas
    
    
def plot_histogram_all(save_folder, final_lspr_growth, lspr_growth, lspr_PL, lspr_mode1, name):
    
    print('Plot Histogram ')   
    
    lim_range = [520, 600]
    bins = 20
    
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,5))

    n,bin_positions_x,p = ax1.hist(final_lspr_growth, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C0')
    mu_final_lspr_growth, sigma_final_lspr_growth, x_text = data_hist(final_lspr_growth, bin_positions_x)
        
    n,bin_positions_x,p = ax1.hist(lspr_growth, bins=bins, range=lim_range, density=True, rwidth=0.9,alpha = 0.7, color ='C1')
    mu_lspr_growth, sigma_lspr_growth, lspr_growth_x_text = data_hist(lspr_growth, bin_positions_x)
    
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax1.text(530, 0.25, x_text, color= 'C0', fontsize = 'medium')
    ax1.text(530, 0.23, lspr_growth_x_text, color= 'C1', fontsize = 'medium')
        
    ax1.set_xlabel('LSPR PL final growth live')
    ax1.set_ylim(0, 0.12)
    ax1.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax2.hist(lspr_PL, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C2')
    mu_lspr_PL, sigma_lspr_PL, x_text = data_hist(lspr_PL, bin_positions_x)
    
    n,bin_positions_x,p = ax2.hist(lspr_mode1, bins=bins, range=lim_range, density=True, rwidth=0.9, alpha = 0.7, color ='C3')
    mu_lspr_mode1, sigma_lspr_mode1, lspr_mode1_x_text = data_hist(lspr_mode1, bin_positions_x)
    
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax2.text(530, 0.25, x_text, color= 'C2', fontsize = 'medium')
    ax2.text(530, 0.23, lspr_mode1_x_text, color= 'C3', fontsize = 'medium')
    
    ax2.set_xlabel('LSPR PL steps')
    ax2.set_ylim(0, 0.30)
    ax2.set_xlim(lim_range[0],lim_range[1])
    
    f.set_tight_layout(True)

    figure_name = os.path.join(save_folder, name) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    name = os.path.join(save_folder, 'data_%s'%(name))
    data = np.array([mu_final_lspr_growth,sigma_final_lspr_growth, mu_lspr_growth, sigma_lspr_growth,  mu_lspr_PL,  sigma_lspr_PL, mu_lspr_mode1, sigma_lspr_mode1])
    header_txt = 'mean PL live, std PL live, mean PL live post-analysis, std PL live post-analysis,  mean PL steps one mode, std PL steps one mode, mean PL steps two modes, std PL steps two modes'
    np.savetxt(name, data, header = header_txt)
    
    means = mu_final_lspr_growth, mu_lspr_growth, mu_lspr_PL, mu_lspr_mode1
    sigmas = sigma_final_lspr_growth, sigma_lspr_growth, sigma_lspr_PL, sigma_lspr_mode1
    
    return means, sigmas
                                       
if __name__ == '__main__':
    
   # base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PySpectrum/'
    
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
  #  daily_folder = '2021-03-10 (growth HauCl4 on AuNPz)'
        
   # list_of_growth = ['20210310-164939_Growth_12x1_560','20210310-170818_Growth_12x1_565' , '20210310-173325_Growth_12x1_570', 
                 #     '20210310-180110_Growth_12x1_575', '20210310-183913_Growth_12x1_580', '20210310-191030_Growth_12x1_585']
                 
    daily_folder = '2021-03-26 (Growth)'
    
    list_of_growth = ['20210326-132548_Growth_12x2_560','20210326-141841_Growth_12x2_570','20210326-152533_Growth_12x2_580']
    
    sub_folder_growth = 'processed_data_sustrate_bkg_True'
    
   # daily_folder_PL = '2021-03-10 (growth HauCl4 on AuNPz)/2021-03-11(PL growth)/20210311-124059_Luminescence_Steps_12x10/'
   # sub_folder_PL = 'processed_data_luminiscence_sustrate_bkg_False_2/'
   
    daily_folder_PL = '2021-03-26 (Growth)/20210326-184459_Luminescence_Steps_12x10/'
    sub_folder_PL = 'processed_data_luminiscence_sustrate_bkg_False/'
    
    common_path_PL = os.path.join(base_folder, daily_folder_PL, sub_folder_PL)
    
    save_folder = os.path.join(base_folder, daily_folder_PL, sub_folder_PL, 'fig_compare_PL_growth_final')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
#%%
        
    cols = [1, 3, 5]
        
    plt.figure()

    for i in range(3):
        
        folder_growth =   list_of_growth[i] 
        common_path_growth  = os.path.join(base_folder, daily_folder, folder_growth, sub_folder_growth)
        
        number_col = cols[i]
       # name_col = 'Col_%03d'%(number_col)
        name_col = 'Col_%03d_01'%(number_col) #tomo las primeras col  nada mas
        
        NP_growth, lspr_growth, NP_PL_list, lspr_PL = compare_final_spectrum(common_path_growth, common_path_PL, save_folder, name_col) 
        
        total_time_growth, final_lspr_growth, lspr_PL = compare_lspr(NP_growth, lspr_growth, NP_PL_list, lspr_PL, common_path_growth, common_path_PL, save_folder, name_col)
        
        plt.plot(total_time_growth, final_lspr_growth, '*') # label = 'lspr growth live')
        plt.plot(total_time_growth, lspr_PL, 'o') #, label = 'lspr PL steps analysis one mode')
  
    plt.xlabel('Total time growth (s)')
    plt.ylabel('Final LSPR')
    plt.legend(loc='lower right') 
    
    plt.xlim(40, 300)
    plt.ylim(540, 600)
    
    figure_name = os.path.join(save_folder, 'time_vs_final_lspr_PL_01.png') 
    
    #plt.show()
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    
#%%
    
    cols = [1, 3, 5]
    
    folder = os.path.join(save_folder, 'info_PL_buenos_casos')

    for i in range(3):
        
        number_col = cols[i]
        
        name_col = 'Col_%03d'%(number_col)
        
        means, sigmas = plot_more_analysis_all(folder, name_col)
        
    