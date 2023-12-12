#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 12:20:43 2021

@author: luciana
"""


import os
import re
import numpy as np
import matplotlib.pyplot as plt


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

def plot_histogram(save_folder, name_col, total_time, lspr_PL_live, lspr_PL, max_sca_unpol):
    
    print('Plot Histogram ')   
    
   # f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15,5))
   
    fig = plt.figure()
    ax0 = fig.add_subplot(221)
    ax3 = fig.add_subplot(222)
    ax1 = fig.add_subplot(223)
    ax2 = fig.add_subplot(224)

    lim_range = [540, 640]
    bins = 10
    
    n,bin_positions_x,p = ax0.hist(lspr_PL_live, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C0')
    mu_lspr_PL_live, sigma_lspr_PL_live, x_text = data_hist(lspr_PL_live, bin_positions_x)
    ax0.text(550, 0.10, x_text, color= 'C0', fontsize = 'medium')
    ax0.set_xlabel('LSPR PL growth control live (nm)')
    ax0.set_ylim(0, 0.12)
    ax0.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax1.hist(lspr_PL, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C1')
    mu_lspr_PL, sigma_lspr_PL, x_text = data_hist(lspr_PL, bin_positions_x)
    ax1.text(550, 0.10, x_text, color= 'C1', fontsize = 'medium')
    ax1.set_xlabel('LSPR PL post-growth (nm)')
    ax1.set_ylim(0, 0.12)
    ax1.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax2.hist(max_sca_unpol, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C2')
    mu_lspr_sca, sigma_lspr_sca, x_text = data_hist(max_sca_unpol, bin_positions_x)
    ax2.text(550, 0.10, x_text, color= 'C2', fontsize = 'medium')
    ax2.set_xlabel('LSPR Scattering Unpol (nm)')
    ax2.set_ylim(0, 0.12)
    ax2.set_xlim(lim_range[0],lim_range[1])
    
    lim_range_2 = [30, 300]
    bins_2 = 10
    n,bin_positions_x,p = ax3.hist(total_time, bins=bins_2, range=lim_range_2, density=True, rwidth=0.9, color ='C3')
    mu_time, sigma_time, x_text = data_hist(total_time, bin_positions_x)
    ax3.text(150, 0.04, x_text, color= 'C3', fontsize = 'medium')
    ax3.set_xlabel('Time growth (s)')
    ax3.set_ylim(0, 0.05)
    ax3.set_xlim(lim_range_2[0],lim_range_2[1])
    
    fig.set_tight_layout(True)

    figure_name = os.path.join(save_folder, 'histogram_%s.png'%name_col) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    return

def plot_histogram_all(save_folder, name_col, total_time, lspr_PL_live, lspr_PL, max_sca_unpol, max_sca_px, max_sca_py):
    
    print('Plot Histogram ')   
    
   # f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15,5))
   
    fig = plt.figure()
    
    ax3 = fig.add_subplot(231)
    ax0 = fig.add_subplot(232)
    ax1 = fig.add_subplot(233)
    
    ax2 = fig.add_subplot(234)
    ax2_x = fig.add_subplot(235)
    ax2_y = fig.add_subplot(236)
    
    lim_range = [540, 640]
    bins = 10
    
    n,bin_positions_x,p = ax0.hist(lspr_PL_live, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C0')
    mu_lspr_PL_live, sigma_lspr_PL_live, x_text = data_hist(lspr_PL_live, bin_positions_x)
    ax0.text(550, 0.10, x_text, color= 'C0', fontsize = 'medium')
    ax0.set_xlabel('LSPR PL growth (nm)')
    ax0.set_ylim(0, 0.12)
    ax0.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax1.hist(lspr_PL, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C1')
    mu_lspr_PL, sigma_lspr_PL, x_text = data_hist(lspr_PL, bin_positions_x)
    ax1.text(550, 0.10, x_text, color= 'C1', fontsize = 'medium')
    ax1.set_xlabel('LSPR PL post-growth (nm)')
    ax1.set_ylim(0, 0.12)
    ax1.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax2.hist(max_sca_unpol, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C2')
    mu_lspr_sca, sigma_lspr_sca, x_text = data_hist(max_sca_unpol, bin_positions_x)
    ax2.text(550, 0.10, x_text, color= 'C2', fontsize = 'medium')
    ax2.set_xlabel('LSPR Scattering Unpol (nm)')
    ax2.set_ylim(0, 0.12)
    ax2.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax2_x.hist(max_sca_px, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C4')
    mu_lspr_sca_px, sigma_lspr_sca_px, x_text = data_hist(max_sca_px, bin_positions_x)
    ax2_x.text(550, 0.10, x_text, color= 'C4', fontsize = 'medium')
    ax2_x.set_xlabel('LSPR Sca.axis Min (nm)')
    ax2_x.set_ylim(0, 0.12)
    ax2_x.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax2_y.hist(max_sca_py, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C5')
    mu_lspr_sca_py, sigma_lspr_sca_py, x_text = data_hist(max_sca_py, bin_positions_x)
    ax2_y.text(550, 0.10, x_text, color= 'C5', fontsize = 'medium')
    ax2_y.set_xlabel('LSPR Sca.axis Max (nm)')
    ax2_y.set_ylim(0, 0.12)
    ax2_y.set_xlim(lim_range[0],lim_range[1])
    
    lim_range_2 = [30, 300]
    bins_2 = 10
    n,bin_positions_x,p = ax3.hist(total_time, bins=bins_2, range=lim_range_2, density=True, rwidth=0.9, color ='C3')
    mu_time, sigma_time, x_text = data_hist(total_time, bin_positions_x)
    ax3.text(40, 0.04, x_text, color= 'C3', fontsize = 'medium')
    ax3.set_xlabel('Time growth (s)')
    ax3.set_ylim(0, 0.05)
    ax3.set_xlim(lim_range_2[0],lim_range_2[1])
    
    fig.set_tight_layout(True)

    figure_name = os.path.join(save_folder, 'all_histogram_%s.png'%name_col) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    return

if __name__ == '__main__':
    
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
    
    file_PL = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)/PL/20210818-174530_Luminescence_Steps_10x12/range_luminiscence_sustrate_bkg_False'
    file_sca = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)/fig_compare_Scattering_Polarization_three'
    
    file_PL = os.path.join(base_folder, file_PL, 'crude_fig_compare_PL_growth_final', 'buenos_casos_3')
    
    file_sca = os.path.join(base_folder, file_sca, 'buenos_casos_3')
    
    daily_folder = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)'
    
    save_folder = os.path.join(base_folder, daily_folder, 'Plot_Resumen_All/3')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    file_mean_kinetics = os.path.join(save_folder, 'all_poly_SPR_mean_200s.txt')
    kinetics = np.loadtxt(file_mean_kinetics, skiprows = 1)
    time_kinetics = kinetics[0, :]
    lspr_kinetics = kinetics[1, :]
    
    cols = 3,4,5 #7,9,11 # 7,9,11 #3,4,5 #
    name_file = '3,4y5' #'7,9y11' #3,4y5' #'7,9y11'
    
    plot_histogram_bool = True
    plot_time_mean = False
    plot_time = False
    plot_lspr = False
    
    plt.figure()
    
    for i in range(len(cols)):
         
        number_col = cols[i]
        name_col = 'Col_%03d'%(number_col)
        
        print(name_col)
        
        file_PL_col = os.path.join(file_PL, 'lspr_fit_PL_%s.txt'%name_col)
        # NP, lspr growth live poly (nm), Total time (s), lspr growth analysis poly (nm), lspr PL steps analysis poly (nm), lspr PL steps analysis two modes (nm)
        data_PL = np.loadtxt(file_PL_col, skiprows=1)
        NP_PL = data_PL[:, 0]
        lspr_growth_live = data_PL[:, 1]
        total_time = data_PL[:, 2]
        lspr_PL_steps =  data_PL[:, 4]
        
        file_sca_col = os.path.join(file_sca, 'max_wavelength_Scattering_%s.txt'%name_col)
        # NP, max wavelength Sca PX (nm), Max polynomial PX (nm), NP, max wavelength PY (nm), Max polynomial PY (nm), NP, max wavelength Unpol (nm), Max polynomial Unpol (nm)
        data_sca = np.loadtxt(file_sca_col, skiprows=1)
        NP_sca = data_sca[:, 0]
        max_sca_px = data_sca[:, 2]
        max_sca_py = data_sca[:, 5]
        max_sca_unpol = data_sca[:, 8]
        
        if plot_histogram_bool:
            
            plot_histogram_all(save_folder, name_col, total_time, lspr_growth_live, lspr_PL_steps, max_sca_unpol, max_sca_px, max_sca_py)
            
            plt.figure()
            plt.title('%s'%name_col)
            plt.plot(NP_PL, lspr_growth_live, 'o', label = 'PL growth live')
            plt.plot(NP_PL, lspr_PL_steps, 'o', label = 'PL post growth')
            plt.plot(NP_sca, max_sca_unpol, 'o', label = 'Scattering unpol')
            plt.plot(NP_sca, max_sca_px, 'o',  color = 'C4', label = 'Sca. axis Min')
            plt.plot(NP_sca, max_sca_py, 'o',  color = 'C5', label = 'Sca. axis Max')
            plt.ylim(540, 640)
            plt.xlim(0, 20)
            plt.legend(loc='lower right', fontsize = 'small')
            plt.xlabel('NP')
            plt.ylabel('LSRP (nm)') 
            figure_name = os.path.join(save_folder, 'NPs_%s.png'%name_col) 
            plt.savefig(figure_name, dpi = 400)
            plt.close()
            
        if plot_lspr:
        
            plt.plot(lspr_PL_steps, max_sca_unpol, 'o',  color = 'C2')
            plt.plot(lspr_PL_steps, max_sca_px, 'o',  color = 'C4')
            plt.plot(lspr_PL_steps, max_sca_py, 'o',  color = 'C5')
      
        if plot_time:
      
            plt.plot(total_time, lspr_growth_live, 'o', color = 'C0')
            plt.plot(total_time, lspr_PL_steps, '.', color = 'C1')
            plt.plot(total_time, max_sca_unpol, '*', color = 'C2')
            
            plt.plot(time_kinetics, lspr_kinetics, 'k--')
           
        if plot_time_mean:
            
            plt.errorbar(np.mean(total_time), np.mean(lspr_growth_live), yerr = np.std(lspr_growth_live), xerr = np.std(total_time), fmt= 'o', color = 'C0')
            plt.errorbar(np.mean(total_time), np.mean(lspr_PL_steps), yerr = np.std(lspr_PL_steps), xerr = np.std(total_time), fmt= 'o', color = 'C1')
            plt.errorbar(np.mean(total_time), np.mean(max_sca_unpol), yerr = np.std(max_sca_unpol), xerr = np.std(total_time), fmt= '*', color = 'C2')
          #  plt.errorbar(np.mean(total_time), np.mean(max_sca_px), yerr = np.std(max_sca_px), xerr = np.std(total_time), fmt= '*', color = 'C4')
          #  plt.errorbar(np.mean(total_time), np.mean(max_sca_py), yerr = np.std(max_sca_py), xerr = np.std(total_time), fmt= '*', color = 'C5')
        
            plt.plot(time_kinetics, lspr_kinetics, 'k--')
            
    if plot_lspr:
        
        plt.plot(np.linspace(540, 640, 10), np.linspace(540, 640, 10), '--', color = 'grey')
        plt.ylim(550, 630)
        plt.xlim(550, 630)
        plt.ylabel('LSPR Scattering (nm)')
        plt.xlabel('LSPR PL (nm)')
        figure_name = os.path.join(save_folder, 'lspr_%s.png'%name_file) 
        plt.savefig(figure_name, dpi = 400)
        plt.close()
     
    if plot_time_mean:
  
        plt.ylim(540, 620)
        plt.xlim(20, 340)
        plt.xlabel('Time growth (s)')
        plt.ylabel('LSRP (nm)') 
        figure_name = os.path.join(save_folder, 'mean_lspr_vs_time_%s.png'%name_file) 
        plt.savefig(figure_name, dpi = 400)
        plt.close()
        
    if plot_time:
        
        plt.ylim(540, 620)
        plt.xlim(20, 340)
        plt.xlabel('Time growth (s)')
        plt.ylabel('LSRP (nm)') 
        figure_name = os.path.join(save_folder, 'lspr_vs_time_%s.png'%name_file) 
        plt.savefig(figure_name, dpi = 400)
        plt.close()
        