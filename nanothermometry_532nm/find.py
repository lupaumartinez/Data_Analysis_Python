# -*- coding: utf-8 -*-
"""
Analysis of single AuNPs photoluminiscence spectra for temperature calculation 
acquired with PySpectrum at CIBION

Mariano Barella

14 jan 2020

"""

import os
import re
from functions_for_photoluminiscence import save_parameters
import step0_calibrate_irrad as step0
import step1_process_raw_data as step1
import step2_fit_antistokes_ratio as step2
import step3_calculate_temp as step3
import step4_statistics as step4

def run_analysis(T_lab_guess, beta_guess, \
                 find_beta, find_Tzero, \
                 control_power, calibration_power_BFP, meas_pow_bfp, base_folder,NP_folder,exp_time_slow):
    # Parameters to load
    totalbins = 10 #number of bins
    zero_in_kelvin = 273 # in K
    Tzero_guess = zero_in_kelvin + T_lab_guess
    window, deg, repetitions = 51, 1, 2
    mode = 'interp'
    factor = 0.47 # factor de potencia en la muestra
    image_size_px = 10 # IN PIXELS
    image_size_um = 0.8 # IN um
    camera_px_length = 1002 # number of pixels (number of points per spectrum)
    calibration_image_size_px = 1 # IN PIXELS
    calibration_image_size_um = 1 # IN um
#    exp_time_fast = 10 # in seconds
    exp_time_fast = 0.1 # in seconds
    w0 = 0.300#0.342# # in um
    radius = 67 # in nm
    sigma_abs = 18498 # in nm^2 
    #radius = 76 # in nm
    #sigma_abs = 34138 # in nm^2 
#    radius = 51.5 # in nm
#    sigma_abs = 20949 # in nm^2 
#    radius = 27 # in nm
#    sigma_abs = 8599 # in nm^2 
    
    start_notch = 522 # in nm where notch starts ~525 nm (safe zone)
    end_notch = 543 # in nmwhere notch starts ~540 nm (safe zone)
    start_power = 543 # in nm from here we are going to calculate irradiance
    end_power = 560#600#560 # in nm from start_power up to this value we calculate irradiance
    start_spr = end_notch # in nm lambda from where to fit lorentz
    max_spr = 560 # wavelength of the spr maximum, initial parameter for the fitting algorithm 
  
    plot_flag = False # if True will save all spectra's plots for each pixel

    # Limites for the antistokes range
    lower_londa =   515 #nm 510
    upper_londa = start_notch - 2
        
    # Parameters to load
    # Threhsold: check if data-point is going to be included i  n the analysis
    # If both crit0eria do not apply, erase datapoint
    alpha = 0.05 # alpha level for chi-squared test (compare with p-value), coarse criteria
    R2th = 0.9#9 # correlation coefficient threhsold, fine criteria
      
    single_NP_flag = False # if True only one NP will be analyzed
    NP_int = 1 # NP to be analyzed
    
    monitor_flag = False # flag to account for power monitor, False = not measured
    
    use_calibration_flag = False # flag to determine if the calibration (cts/s) are going to be used
    
    last_bin_is_bkg_flag = True # True if last bin is considered a background correction
                                                                                          
    
    do_step0, do_step1, do_step2, do_step3, do_step4 = 0,1,1,1,1
    
#    base_folder =r'G:\Mi unidad\DATA Termometria crecimiento\MuestraAuPb'
    #    base_folder =r'G:\My Drive\DATA Termometria crecimiento\MuestraAuPb'

    
#    base_folder =r'C:\Users\Ituzaingo\LRZ Sync+Share\Proyectos actuales\Temperature measurements\CIBION2021\AuPb\PLTermometria'
#    base_folder =r'G:\Mi unidad\DATA Termometria crecimiento\data_paper_nanothermometry-20210426T164235Z-001\data_paper_nanothermometry'
#    base_folder =r'G:\My Drive\DATA Termometria crecimiento\AuPd Nuevo\Au67Pd6\Combinar'
#    base_folder ='C:\\'

    parent_folder = os.path.join(base_folder, NP_folder)
    list_of_folders = os.listdir(parent_folder)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
    list_of_folders_slow = [f for f in list_of_folders if re.search('Slow_Confocal_Spectrum',f)]
    list_of_folders_slow.sort()
    
    list_of_folders_fast = [f for f in list_of_folders if re.search('Fast_Confocal_Spectrum',f)]
    list_of_folders_fast.sort()
    
    path_to = os.path.join(parent_folder,'processed_data')
                
    save_parameters(path_to, totalbins, meas_pow_bfp, Tzero_guess, beta_guess, window, deg, repetitions ,\
                    mode, factor, image_size_px, image_size_um, camera_px_length, \
                    start_notch, end_notch, start_power, end_power, start_spr, max_spr, plot_flag, \
                    lower_londa, upper_londa, alpha, R2th, single_NP_flag, \
                    NP_int, do_step0, do_step1, do_step2, do_step3, do_step4)
    
    if find_beta and not find_Tzero:
        print('\nTzero is known. Finding beta...')
    elif not find_beta and find_Tzero:
        print('\nBeta is known. Finding Tzero...')
    
    if single_NP_flag:
        print('\nAnalyzing only NP %d.' % NP_int)
    
    #####################################################################
    #####################################################################
    #####################################################################
        
    if do_step0:
        for f in list_of_folders_fast:
            if single_NP_flag:    
                if not re.search('_NP_%03d' % NP_int, f):
                    continue
                else:
                    print('\nTarget NP found!')
                
            folder = os.path.join(parent_folder,f)
        
            print('\n === NP folder:',f)
    
            print('\n------------------------  STEP 0')
        
            step0.process_irrad_calibration(folder, path_to, calibration_image_size_px, 
                                           calibration_image_size_um, camera_px_length, window, 
                                           deg, repetitions, mode, factor, meas_pow_bfp, 
                                           start_notch, end_notch, start_power, end_power, 
                                           start_spr, lower_londa, upper_londa, exp_time_fast, w0)
    else:
        print('\nSTEP 0 was not executed.')
    
    if do_step1:   
        for f in list_of_folders_slow:
            if single_NP_flag:    
                if not re.search('_NP_%03d' % NP_int, f):
                    continue
                else:
                    print('\nTarget NP found!')
                
            folder = os.path.join(parent_folder,f)
        
            print('\n === NP folder:',f)
    
            print('\n------------------------  STEP 1')
            
            if monitor_flag:
                step1.process_power_during_confocal(folder, path_to)
          #luciana version
            step1.process_confocal_to_bins(folder, path_to, totalbins, image_size_px, 
                                           image_size_um, camera_px_length, window, 
                                           deg, repetitions, mode, factor, control_power, calibration_power_BFP, 
                                           meas_pow_bfp, start_notch, end_notch, start_power, end_power, 
                                           start_spr, max_spr, lower_londa, upper_londa, 
                                           exp_time_slow, w0, use_calibration_flag, 
                                           last_bin_is_bkg_flag,EM,
                                           w0,plot_flag=False)
          #juli version con step0 True
           # step1.process_confocal_to_bins(folder, path_to, totalbins, image_size_px, 
            #                               image_size_um, camera_px_length, window, 
             #                              deg, repetitions, mode, factor, meas_pow_bfp,
              #                             start_notch, end_notch, start_power, end_power, 
               #                            start_spr, max_spr, lower_londa, upper_londa, 
                #                           exp_time_slow, w0, use_calibration_flag, 
                 #                          last_bin_is_bkg_flag,EM,
                  #                         plot_flag=plot_flag)
    else:
        print('\nSTEP 1 was not executed.')
        
    #####################################################################
    #####################################################################
    #####################################################################
        
    if do_step2:
        for f in list_of_folders_slow:
            if single_NP_flag:    
                if not re.search('_NP_%03d' % NP_int, f):
                    continue
                else:
                    print('\nTarget NP found!')
                
            folder = os.path.join(parent_folder,f)
        
            print('\n === NP folder:',f)
        
            print('\n------------------------ STEP 2')

            step2.calculate_quotient(folder, path_to, totalbins, lower_londa, 
                                     upper_londa, Tzero_guess, beta_guess, 
                                     last_bin_is_bkg_flag, find_beta, find_Tzero)
                
    else:
        print('\nSTEP 2 was not executed.')
        
    #####################################################################
    #####################################################################
    #####################################################################
    
    if do_step3:
        for f in list_of_folders_slow:
            if single_NP_flag:    
                if not re.search('_NP_%03d' % NP_int, f):
                    continue
                else:
                    print('\nTarget NP found!')
                
            folder = os.path.join(parent_folder,f)
        
            print('\n === NP folder:',f)
            
            print('\n------------------------ STEP 3')
        
            step3.calculate_temp(folder, path_to, totalbins, alpha, R2th)
    else:
        print('\nSTEP 3 was not executed.')
    
    #####################################################################
    #####################################################################
    #####################################################################
    
    if do_step4 and not single_NP_flag:
    
        print('\n------------------------ STEP 4')
    
        step4.gather_data(path_to, R2th, totalbins, monitor_flag)
        
        step4.statistics(path_to, R2th, totalbins, radius, sigma_abs, find_beta, find_Tzero)
    else:
        print('\nSTEP 4 was not executed.')
    
    print('\nProcess done.')
    
    return

if __name__ == '__main__':
    
    beta_guess = 45 # beta bb for 80 AuNPs
#    beta_guess = 62.547 # beta bb for a single 80 AuNPs, the one scanned at room temp in the heating stage temp
#    beta_guess = 62.00 # beta bb for a single 80 AuNPs, averaged across all temp in the heating stage temp
    
#    Data analysis for different power
    T_lab_guess = 22# 20#22
    
    control_power = True
    calibration_power_BFP = 3.06, 0.006 
    meas_pow_bfp = 0 #0.67#
    
    EM=80#150# La electromultiplicacion
    exp_time_slow = 1.2#2.5 # in seconds

    find_beta = True
    find_Tzero = False
#### Aca para analizar solo 1 carpeta

   # base_folder =r'C:\Users\lupau\OneDrive\Documentos\2022-05-24 AuNP 67 impresas'
   # NP_folder = '20220524-173839_Luminescence_10x1_3.0umx3.0um'
   
    base_folder =  r'C:\Users\lupau\OneDrive\Documentos\2022-06-24 Au60 satelites Pd rendija'
    NP_folder = '20220624-155124_Luminescence_Load_grid_in_best_center_size15'
    
    print('\nProcessing %s' % NP_folder)
    print('\nPhotothermal coefficient (initial guess): %.1f K µm2/mW' % beta_guess) 
    print('\nLab. temperature (initial guess): %.1f °C' % T_lab_guess) 
   # print('\nPower at BFP: %.3f mW' % meas_pow_bfp) 
    run_analysis(T_lab_guess, beta_guess, find_beta, find_Tzero, control_power, calibration_power_BFP, meas_pow_bfp, base_folder ,NP_folder,exp_time_slow)    
    
    
    # base_folder =r'G:\My Drive\DATA Termometria crecimiento\AuPd Nuevo\Au67Pd6\Combinar'

    
    # folderlist=[]
    # meas_pow_bfp_list=[0.8,0.78,0.78,0.82,1.18,1.11,1.11,1.14,1.14] #AuPd6
    # exp_time_slow_list=[2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5]
    # # meas_pow_bfp_list=[0.78,0.8,0.72,0.73,0.73,0.73,0.73] #AuPd2
    # # exp_time_slow_list=[2.5,2.5,2.5,2.5,2.5,2.5,2.5]
    # #  meas_pow_bfp_list=[2,1,6] #AuPd2
    # #exp_time_slow_list=[4,4]
    
    # folderlist = os.listdir(base_folder)
    # folderlist = [f for f in folderlist if os.path.isdir(os.path.join(base_folder,f))]
    # analizar=[1,1,1,1,1,1,1,1,1]

    # i=0
    # for NP_folder in folderlist:
    #     if analizar[i]==1:
    #         meas_pow_bfp=meas_pow_bfp_list[i]
    #         exp_time_slow=exp_time_slow_list[i]
    #         print('\nProcessing %s' % NP_folder)
    #         print('\nPhotothermal coefficient (initial guess): %.1f K µm2/mW' % beta_guess) 
    #         print('\nLab. temperature (initial guess): %.1f °C' % T_lab_guess) 
    #         print('\nPower at BFP: %.3f mW' % meas_pow_bfp) 
    #         run_analysis(T_lab_guess, beta_guess, find_beta, find_Tzero, meas_pow_bfp,base_folder ,NP_folder,exp_time_slow)
    #     i=i+1 
        
        
        