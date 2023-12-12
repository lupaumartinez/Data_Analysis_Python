#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:07:57 2021

@author: luciana
"""


 #%%   
    others = False
        
    list_number_col = 4,5,7
    input_wavelength_PySpectrum = 580, 570, 545 #nm
    
    input_correlation_nm = dict(zip(list_number_col, input_wavelength_PySpectrum))
    
    x = np.linspace(540, 600, 10)
    
    if others :
        
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
    
    if others :
        
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