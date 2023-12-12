# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:53:47 2019

@author: Luciana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from scipy.optimize import curve_fit
import scipy.signal as sig

from Fit_raman_water import fit_signal_raman, fit_signal_raman_test

import Plot_Spectrum_liveview_luminiscence_step2 as step2

hc = 1239.84193 # Plank's constant times speed of light in eV*nm
k = 0.000086173 # Boltzmann's constant in eV/K

def smooth_Signal(signal, window, deg, repetitions):
    
    k = 0
    while k < repetitions:
        signal = sig.savgol_filter(signal, window, deg, mode = 'mirror')
        k = k + 1
        
    return signal


def calc_r2(observed, fitted):
    # Calculate coefficient of determination
    avg_y = observed.mean()
    ssres = ((observed - fitted)**2).sum()
    sstot = ((observed - avg_y)**2).sum()
    return 1.0 - ssres/sstot

def three_lorentz(x, *p):

    pi = np.pi

    I, gamma, x0, I_2, I_3, C = p
    
    a = (1/pi) * I_2 * (15.5/2)**2 / ((x - 649)**2 + (15.2/2)**2) 
    b = (1/pi) * I_3 * (183/2)**2 / ((x - 702)**2 + (183/2)**2) 
    
    return  (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + (1/pi) * I_2 * a + (1/pi) * I_3 * b + C

def lorentz(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = np.pi
    I, gamma, x0, C = p
    return (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def fit_lorentz(p, x, y):
    
    try:
        A = curve_fit(lorentz, x, y, p0 = p)

    except RuntimeError:
        print("Error - curve_fit failed")
        A =  np.zeros(4), np.zeros(4)
    
    return A

def bose_einstein(londa, *p):
    # Bose-Einstein distribution
    # temp must be in Kelvin
    hc = 1239.84193 # Plank's constant times speed of light in eV*nm
    k = 0.000086173 # Boltzmann's constant in eV/K
    
    temp = p
    
    kT = k*temp
    londa_exc = 532
    inside = hc * (1/londa - 1/londa_exc) / kT
    out = 1 / ( np.exp( inside ) - 1 )

    return out

def fit_bose_einstein(p, x, y):
    
    try:
        A = curve_fit(bose_einstein, x, y, p0 = p)

    except RuntimeError:
        print("Error - curve_fit failed")
        A =  np.zeros(1), np.zeros(1)
    
    return A

def fit_spr(spectrum, wavelength, init_param):

    best_lorentz, err = fit_lorentz(init_param, wavelength, spectrum)

    lorentz_fitted = lorentz(wavelength, *best_lorentz)
    r2_coef_pearson = calc_r2(spectrum, lorentz_fitted)

    if best_lorentz[0] != 0:

        lorentz_fitted = lorentz(wavelength, *best_lorentz)
        r2_coef_pearson = calc_r2(spectrum, lorentz_fitted)

        full_lorentz_fitted = lorentz(wavelength, *best_lorentz)
        londa_max_pl = best_lorentz[2]
        width_pl = best_lorentz[1]
        
    else: 
        
        full_lorentz_fitted = np.zeros(len(wavelength))
        londa_max_pl = 0
        width_pl =  0
        best_lorentz = np.array([0,0 ,0, 0], dtype=np.double)
    
    return full_lorentz_fitted, best_lorentz

def find_bose_enistein(spectrum, wavelength, best_lorentz):

    AS_lorentz = lorentz(wavelength, *best_lorentz)
    
    spectrum_bose_einstein = spectrum/AS_lorentz

    return spectrum_bose_einstein


def fit_linear(x, y, weights, intercept=True):
    # fit y=f(x) as a linear function
    # intercept True: y = m*x+c (non-forcing zero)
    # intercept False: y = m*x (forcing zero)
    # weights: used for weighted fitting. If empty, non weighted

    x = np.array(x)
    y = np.array(y)
    weights = np.array(weights)
    N = len(x)
    x_original = x
    y_original = y
    # calculation of weights for weaighted least squares
    if weights.size == 0:
        print('Ordinary Least-Squares')
        x = x
        y = y
        ones = np.ones(N)
    else:
        print('Weighted Least-Squares')
        x = x * np.sqrt(weights)
        y = y * np.sqrt(weights)
        ones = np.ones(N) * np.sqrt(weights)
    # set for intercept true or false
    if intercept:
        indep = ones
    else:
        indep = np.zeros(N)

    # do the fitting
    if N > 1:
        print('More than 1 point is being fitted.')
        A = np.vstack([x, indep]).T
        p, residuals, _, _ = np.linalg.lstsq(A, y)
#        print('Irradiance', x)
#        print('Temp. increase', y)
#        print('Slope', p[0], 'Offset', p[1])
        x_fitted = np.array(x_original)
        y_fitted = np.polyval(p, x_fitted)
        # calculation of goodess of the fit
        y_mean = np.mean(y)
        SST = sum([(aux2 - y_mean)**2 for aux2 in y_original])
        SSRes = sum([(aux3 - aux2)**2 for aux3, aux2 in zip(y_original, y_fitted)])
        r_squared = 1 - SSRes/SST
        # calculation of parameters and errors
        m = p[0]
        sigma = np.sqrt(np.sum((y_original - p[0]*x_original - p[1])**2) / (N - 2))
        aux_err_lstsq = (N*np.sum(x_original**2) - np.sum(x_original)**2)
        err_m = sigma*np.sqrt(N / aux_err_lstsq)
        if intercept:
            c = p[1]
            err_c = sigma*np.sqrt(np.sum(x_original**2) / aux_err_lstsq)
        else:
            c = 0
            err_c = 0
    # elif N == 0:
    #     print('No points to fit')
    #     m = 0
    #     err_m = 0
    #     c = 0
    #     err_c = 0
    #     r_squared = 0
    #     x_fitted = x
    #     y_fitted = y
    else:
        print('One single point. No fitting performed.')
        m = 0
        err_m = 0
        c = 0
        err_c = 0
        r_squared = 0
        x_fitted = x
        y_fitted = y

    return m, c, err_m, err_c, r_squared, x_fitted, y_fitted
    

def process_spectrum(folder, common_path, first_wave, starts_notch, ends_notch, last_wave, window, deg, repetitions, spectrum_time, fit_raman_water, plot_antistokes, fit_lorentz, fit_bose_einstein, calibration_BS, sustrate_bkg_bool, trace_BS_bool):
    
    ROI_size = 1
    
    NP = folder.split('Spectrum_')[-1]
    
    if NP == folder: #esto es para poder analizar un video individual que no sea de la rutina Growth
        NP = 'one'

    save_folder = os.path.join(common_path,'%s'%NP)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    name_spectrum = []
    specs = []
    
    parameters_I_spr = []
    parameters_width_spr = []
    parameters_londa_spr = []
    parameters_I_649 = []
    parameters_I_702 = []
    parameters_C = []
    
    name_spectrum_bkg = []
    specs_bkg = []
    
    list_of_files = os.listdir(folder)
    list_of_files.sort()
    
    time_BFP = []
    power_BFP = []
        
    for file in list_of_files:
        
        if file.startswith('Calibration_Shamrock'):
            name_file = os.path.join(folder, file)
            wavelength = np.loadtxt(name_file)
            
        if file.startswith('Line_Spectrum'):
            name_file = os.path.join(folder, file)
            a = file.split('step_')[-1]
            b = int(a.split('.txt')[0])
            name_spectrum.append(b)
            specs.append(np.loadtxt(name_file))
            
        if sustrate_bkg_bool:
            
            if file.startswith('Background_Line_Spectrum'):
                name_file = os.path.join(folder, file)
                a = file.split('step_')[-1]
                b = int(a.split('.txt')[0])
                name_spectrum_bkg.append(b)
                specs_bkg.append(np.loadtxt(name_file))
                
        if trace_BS_bool:
            
            if file.startswith('Trace_BS'):
                name_file = os.path.join(folder, file)
                data = np.loadtxt(name_file)
                time_BFP = data[:, 0]
                power_BFP = calibration_BS[1] + calibration_BS[0]*data[:, 1]
            
        if fit_raman_water:    
            
            if file.startswith('Parameters_FitSpectrum'):
                name_file = os.path.join(folder, file)
                parameters = np.loadtxt(name_file, comments='#')
                
                parameters_I_spr.append(parameters[0])
                parameters_width_spr.append(parameters[1])
                parameters_londa_spr.append(parameters[2])
                parameters_I_649.append(parameters[3])
                parameters_I_702.append(parameters[4])
                parameters_C.append(parameters[5])
                
    if trace_BS_bool:
           
        plt.figure()
        plt.plot(time_BFP, power_BFP)
        plt.xlabel('Time (s)')
        plt.ylabel('Power BFP (mW)')
        figure_name = os.path.join(common_path, 'Trace_BS_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        header_plots = "Time (s), Power BFP (mW)"
        data_plots = np.array([time_BFP, power_BFP]).T
        name_data = os.path.join(save_folder, 'data_power_BFP_%s.txt'%(NP))
        np.savetxt(name_data, data_plots, header = header_plots)
        
    specs = [specs for _,specs in sorted(zip(name_spectrum,specs))]            
    name_spectrum = sorted(name_spectrum)
    
    specs_bkg = [specs_bkg for _,specs_bkg in sorted(zip(name_spectrum_bkg,specs_bkg))]      
        
    L = len(specs)
            
    print('Number:', NP, 'Spectra acquired:', L)
    
    range_stokes = np.where((wavelength >= ends_notch) & (wavelength <= last_wave))
    wavelength_stokes = wavelength[range_stokes]
    
    print('Integrate points Stokes', len(list(range_stokes[0])), 'per', ROI_size) #192 caso Stokes, 202 caso antistokes
        
    range_antistokes = np.where((wavelength >= first_wave) & (wavelength <=starts_notch))
    wavelength_antistokes = wavelength[range_antistokes]
    
    print('Integrate points Anti-Stokes', len(list(range_antistokes[0])), 'per', ROI_size) #192 caso Stokes, 202 caso antistokes
        
    initial = 0
    final = L
    
    total_frame = len(range(initial, final))
    
    frame_list = np.zeros(total_frame)
    max_wavelength = np.zeros(total_frame)
    max_intensity_stokes = np.zeros(total_frame)
    integrate_stokes = np.zeros(total_frame)
    integrate_antistokes = np.zeros(total_frame)
    
    matrix_total_original = np.zeros((total_frame, len(wavelength)))
    matrix_total = np.zeros((total_frame, len(wavelength)))
    
    color_map = plt.cm.coolwarm(np.linspace(0,1,total_frame))
    
    plt.figure()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
    plt.title('NP_%1s'%(NP))
    
    for i in range(initial, final):
        
        frame_list[i] = (i+1)*spectrum_time
        
        spectrum_original = ROI_size*specs[i]
        
    #    spectrum_original_smooth = smooth_Signal(spectrum_original, window, deg, repetitions)
        matrix_total_original[i, :] = spectrum_original#l_smooth 
        
        if sustrate_bkg_bool:
            
            spectrum = spectrum_original - ROI_size*specs_bkg[i]
            
        else:
            
            spectrum = spectrum_original
         #   spectrum_bkg =  ROI_size*specs_bkg[i]
         
        spectrum = smooth_Signal(spectrum, window, deg, repetitions)
        matrix_total[i, :] = spectrum

        spectrum_stokes = spectrum[range_stokes]
        spectrum_antistokes = spectrum[range_antistokes]
    
        max_wavelength[i] = wavelength_stokes[np.argmax(spectrum_stokes)]
        
        max_intensity_stokes[i] = np.max(spectrum_stokes)
        integrate_stokes[i] = (np.sum(spectrum_stokes))
        integrate_antistokes[i] = (np.sum(spectrum_antistokes))
        
        if np.isnan(np.sum(spectrum_antistokes)):
            integrate_antistokes[i] = 0
            
        plt.plot(wavelength, spectrum)
        
    plt.axvline(532, color = 'g', linestyle = '--')
    plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
   # plt.axvspan(ends_notch, last_wave, facecolor='lightblue', alpha=0.2)
   # plt.axvspan(ends_notch, last_wave, facecolor='grey', alpha=0.1)
    #plt.axvspan(starts_notch, ends_notch, facecolor='b', alpha=0.1)
   # plt.xlim(500, 605)
  #  plt.ylim(0, 8000)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Counts')
        
    figure_name = os.path.join(common_path, 'Liveview_spectrum_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()

    if fit_raman_water:
        
        new_Ispr = []
        new_I649 = []
        new_I702 = []
        r2_list = []
            
        plt.figure()
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
        plt.title('NP_%1s'%(NP))
    
        fit_wavelength = np.linspace(530, 830, 100)
        for i in range(initial, final):
            
            Io = float(parameters_I_spr[i])
            width = float(parameters_width_spr[i])
            londa = float(parameters_londa_spr[i])
            I_649 = float(parameters_I_649[i])
            I_702 = float(parameters_I_702[i])
            C = float(parameters_C[i])
            p = np.array([Io, width, londa, I_649, I_702, C], dtype=np.double)
            fit_spectrum = three_lorentz(fit_wavelength, *p)
            
            spectrum_stokes = matrix_total_original[i, :][range_stokes] #sin restar el bkg
            fitted =  three_lorentz(wavelength_stokes, *p)
            r2 = calc_r2(spectrum_stokes, fitted)
            
            Ispr = fit_spectrum[ np.where((fit_wavelength >= londa - 2) & (fit_wavelength <= londa + 2))[0][0]]
            I702 = fit_spectrum[ np.where((fit_wavelength >= 702 - 1.5) & (fit_wavelength <= 702 + 1.5))[0][0]]
            I649 = fit_spectrum[ np.where((fit_wavelength >= 649 - 1.5) & (fit_wavelength <= 702 + 1.5))[0][0]]
            
            new_Ispr.append(Ispr)
            new_I702.append(I702)
            new_I649.append(I649)
            r2_list.append(r2)
            
            plt.plot(wavelength_stokes, spectrum_stokes)
            plt.plot(fit_wavelength, fit_spectrum, 'k--')
            
        plt.axvline(532, color = 'g', linestyle = '--')
        plt.axvspan(starts_notch, ends_notch, facecolor='grey', alpha=0.2)
       # plt.axvspan(ends_notch, last_wave, facecolor='lightblue', alpha=0.2)
       # plt.axvspan(ends_notch, last_wave, facecolor='grey', alpha=0.1)
        #plt.axvspan(starts_notch, ends_notch, facecolor='b', alpha=0.1)
       # plt.xlim(500, 605)
      #  plt.ylim(0, 8000)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Counts')
            
        figure_name = os.path.join(common_path, 'Fit_Liveview_spectrum_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
    plt.style.use('default')
   
    plt.figure()
    plt.plot(frame_list, max_wavelength, '-o')
    plt.xlabel('Time (s)')
    plt.ylabel('$\u03BB_{max}$ (nm)')
  #  plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'Max_wavelength_Stokes_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    plt.figure()
    plt.plot(frame_list, integrate_stokes, '-o')
    plt.xlabel('Time (s)')
    plt.ylabel('Integrate Stokes')
 #   plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'Integrate_Stokes_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400) 
    plt.close()
    
    plt.figure()
    plt.plot(frame_list, max_intensity_stokes, '-o')
    plt.xlabel('Time (s)')
    plt.ylabel('Max intensity Stokes')
   # plt.xlim(-2, 242)
    figure_name = os.path.join(common_path, 'max_intensity_Stokes_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    header_plots = "Time (s), Max wavelength (nm), Max Intensity Stokes, Integrate Stokes, Integrate Anti-Stokes"
    data_plots = np.array([frame_list, max_wavelength, max_intensity_stokes, integrate_stokes, integrate_antistokes]).T
    name_data = os.path.join(save_folder, 'data_plots_%s.txt'%(NP))
    np.savetxt(name_data, data_plots, header = header_plots)
    
    if plot_antistokes:
    
        plt.figure()
        plt.plot(frame_list, integrate_antistokes, '-o')
        plt.xlabel('Time (s)')
        plt.ylabel('Integrate Anti-Stokes')
       # plt.xlim(-2, 242)
        figure_name = os.path.join(common_path, 'Integrate_Anti-Stokes_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)      
        plt.close()

    if fit_lorentz:

        plt.figure()
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
        
        print('Fitting three Lorentz post liveview')
        
        header_text = 'Wavelength (nm), Intensity, Fit three Lorentz Intensity'
        
        londa_max_pl = np.zeros(total_frame)
        width_pl = np.zeros(total_frame)
        r2_lorentz = np.zeros(total_frame)
        
        #preparo ajuste
        I = 2500#20000#6000
        init_londa = 550
        init_width = 50
        I2 = 100#2500#1000
        I3 = 100#0#0
        C = 450
        
        init_parameters_NP = np.array([I, init_width, init_londa, I2, I3, C], dtype=np.double)
    
        for i in range(initial, final):
            
            spectrum_not_smooth = matrix_total_original[i, :]
            spectrum = smooth_Signal(spectrum_not_smooth, window, deg, repetitions)
            
          #  spectrum_stokes_not_smooth = spectrum_not_smooth[range_stokes] 
            spectrum_stokes = spectrum[range_stokes] #sin restar el bkg
    
            wavelength_fitted, lorentz_fitted, best_lorentz = fit_signal_raman_test(wavelength, spectrum, ends_notch, last_wave, init_parameters_NP)
            
            full_lorentz_fitted = three_lorentz(wavelength_fitted, *best_lorentz)
            fitted =  three_lorentz(wavelength_stokes, *best_lorentz)
            r2_lorentz[i] = calc_r2(spectrum_stokes, fitted)
    
            #I, gamma, x0, I_2, I_3, C = p
            londa_max_pl[i]  = best_lorentz[2]
            width_pl[i] = best_lorentz[1]
            
            #actualizo el init_parameters_NP
            init_parameters_NP = best_lorentz
            
            plt.title('NP_%1s'%(NP))
            plt.plot(wavelength_stokes, spectrum_stokes)
            plt.plot(wavelength_fitted, full_lorentz_fitted, '--k')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Normalizated')
            
            if i == initial:
                name = os.path.join(save_folder, 'data_initial_Stokes_%s.txt'%(NP))
                data = np.array([wavelength_stokes, spectrum_stokes, fitted]).T
                np.savetxt(name, data, header = header_text)
                
            if i == final - 1:
                name = os.path.join(save_folder, 'data_final_Stokes_%s.txt'%(NP))
                data = np.array([wavelength_stokes, spectrum_stokes, fitted]).T
                np.savetxt(name, data, header = header_text)
            
        figure_name = os.path.join(common_path, 'Stokes_spectrum_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)
        
        time_londa = frame_list[np.where(londa_max_pl>0)[0]]
        londa_max_pl = londa_max_pl[np.where(londa_max_pl>0)[0]]
       # time_width = frame_list[np.where(width_pl>0)[0]]
        width_pl = width_pl[np.where(londa_max_pl>0)[0]]
        r2_lorentz = r2_lorentz[np.where(londa_max_pl>0)[0]]
        
        plt.style.use('default')
        
        plt.figure()
        plt.plot(frame_list, max_wavelength, '-o', color = 'C0', label = '$\u03BB_{max}$')
        plt.plot(time_londa, londa_max_pl, '-*',  color = 'C1', label = '$\u03BB_{SPR}$ 3 Lorentz fitting')
        
        if fit_raman_water: 
            plt.plot(frame_list, parameters_londa_spr, '--*',  color = 'C2', label = '$\u03BB_{SPR}$ 3 Lorentz fitting liveview' )
        
        plt.xlabel('Time (s)')
        plt.ylabel('$\u03BB_{max}$ (nm)')
        #plt.xlim(-2, 242)
       # plt.ylim(540,700)
        plt.legend(loc = 'upper right')
        figure_name = os.path.join(common_path, 'Wavelength_SPR_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        plt.figure()
        plt.plot(time_londa, width_pl, '-*',  color = 'C1', label = '3 Lorentz fitting')
        
        if fit_raman_water: 
            plt.plot(frame_list, parameters_width_spr, '--*',  color = 'C2', label = '3 Lorentz fitting liveview' )
        plt.xlabel('Time (s)')
        plt.ylabel('Width SPR (nm)')
        plt.legend(loc = 'upper right')
       # plt.xlim(-2, 242)
        figure_name = os.path.join(common_path, 'Width_SPR_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)    
        plt.close()
        
        plt.figure()
        plt.plot(time_londa, r2_lorentz, '-*', color = 'C1', label = 'r2 3 Lorentz')
        if fit_raman_water: 
            plt.plot(frame_list, r2_list, '--*', color = 'C2', label = 'r2 3 Lorentz liveview')
       # plt.plot(frame_list, new_C, '-*', label = 'Cte: fit raman water liveview')
        plt.xlabel('Time (s)')
        plt.ylabel('r2 fitting')
        plt.legend(loc = 'upper right')
        figure_name = os.path.join(common_path, 'r2_fit_raman_water_%s.png' % (NP)) 
        plt.savefig(figure_name , dpi = 400)    
        plt.close()
        
        if fit_raman_water: 
            
            #parameters_I_spr = np.array(parameters_I_spr)/parameters_I_spr[0]
            #parameters_I_649 = np.array(parameters_I_649)/parameters_I_649[0]
            #parameters_I_702 = np.array(parameters_I_702)/parameters_I_702[0]
            #parameters_C = np.array(parameters_C)/parameters_C[0]
            
            plt.figure()
          #  plt.plot(time_londa, parameters_I_spr, '-*', label = 'I spr: fit raman water liveview')
          #  plt.plot(time_londa, parameters_I_649, '-*', label = 'I 649 nm: fit raman water liveview')
          #  plt.plot(time_londa, parameters_I_702, '-*', label = 'I 702 nm: spr fit raman water liveview')
          #  plt.plot(time_londa, parameters_C, '-*', label = 'Cte: fit raman water liveview')
            plt.plot(frame_list, new_Ispr, '-*', label = 'I spr')
            plt.plot(frame_list, new_I649, '-*', label = 'I 649 nm')
            plt.plot(frame_list, new_I702, '-*', label = 'I 702 nm')
           # plt.plot(frame_list, new_C, '-*', label = 'Cte: fit raman water liveview')
            plt.xlabel('Time (s)')
            plt.ylabel('Intensity')
            plt.legend(loc = 'upper right')
            figure_name = os.path.join(common_path, 'Intensity_fit_raman_water_%s.png' % (NP)) 
            plt.savefig(figure_name , dpi = 400)    
            plt.close()
            

        header_fitting = "Time (s), Londa SPR (nm), Width SPR (nm)"
        data_fitting = np.array([time_londa, londa_max_pl, width_pl]).T
        name_data = os.path.join(save_folder, 'data_fitting_%s.txt'%(NP))
        np.savetxt(name_data, data_fitting, header = header_fitting)
        
        if fit_raman_water: 
            
            header_fitting = "Time (s), Londa SPR (nm)"
            data_fitting = np.array([frame_list, parameters_londa_spr]).T
            name_data = os.path.join(save_folder, 'data_live_fitting_%s.txt'%(NP))
            np.savetxt(name_data, data_fitting, header = header_fitting)
            
            print('Tiempo total de creecimiento:', 'por trace BS', time_BFP[-1], 'por spectrum live', frame_list[-1])
    
            header_fitting = "Total Time BS (s), Mean Power BS (mW), Total Time Spectrum (s), Londa SPR live (nm)"
            data_fitting = np.array([time_BFP[-1], np.mean(power_BFP), frame_list[-1], parameters_londa_spr[-1]]).T
            name_data = os.path.join(save_folder, 'data_growth_final_%s.txt'%(NP))
            np.savetxt(name_data, data_fitting, header = header_fitting)
        
        if fit_bose_einstein:
        
            Temp_BE_1 = np.zeros(total_frame)
            err_Temp_BE = np.zeros(total_frame)
            
            plt.figure()
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_map)
            
            for i in range(initial, final):
                
                if sustrate_bkg_bool:
                
                    spectrum = ROI_size*(specs[i]  - specs_bkg[i])
                
                else:
                
                    spectrum = ROI_size*specs[i]
            
                spectrum = smooth_Signal(spectrum, window, deg, repetitions)
                spectrum_stokes = spectrum[range_stokes]
              #  spectrum_stokes_norm = spectrum_stokes#/max(spectrum_stokes)
                
                spectrum_antistokes = spectrum[range_antistokes]
               # spectrum_antistokes_norm = spectrum_antistokes#/max(spectrum_stokes)
                
              #  init_londa = ( wavelength_stokes[-1] + wavelength_stokes[0])/2  
              #  init_param = np.array([5000, 70, init_londa, 0], dtype=np.double)
              #  full_lorentz_fitted, best_lorentz = fit_spr(spectrum_stokes, wavelength_stokes, init_param)
            
                wavelength_fitted, full_lorentz_fitted, best_lorentz = fit_signal_raman(wavelength, spectrum, ends_notch, last_wave)
                #I, gamma, x0, I_2, I_3, C = p
                londa_max_pl[i]  = best_lorentz[2]
                width_pl[i] = best_lorentz[1]
                
                if best_lorentz[2] > 0: 
                    
                 #   AS_lorentz = lorentz(wavelength_antistokes, *best_lorentz)      
                    bose_einstein = find_bose_enistein(spectrum_antistokes, wavelength_antistokes, best_lorentz)
                    
                   # plt.plot(wavelength_antistokes, spectrum_antistokes, '-')
                   # plt.plot(wavelength_antistokes, AS_lorentz , 'k--')
                   # plt.plot(wavelength_antistokes, bose_einstein, 'm--')
                   
                  #   best_be, err = fit_bose_einstein( np.array([2000], dtype=np.double), wavelength_antistokes, spectrum_antistokes)
                  #   print(*best_be)
                   #  bose_einstein_fitted = bose_einstein(wavelength_antistokes, *best_be)
                    
                   #  plt.plot(wavelength_antistokes, bose_einstein)
                     #plt.plot(wavelength_antistokes, bose_einstein_fitted, 'k-')
        
                    inside = np.log(1/bose_einstein+1)
                    plt.plot(1/wavelength_antistokes, inside, '-')
                    
                    m, c, err_m, err_c, r_squared, x_fitted, y_fitted = fit_linear(1/wavelength_antistokes, inside, weights = [])
        
                    plt.plot(x_fitted, y_fitted, 'k--')
                  
                    
                    if r_squared > 0.1:
                        
                        print('Linear Fit bose-einstein', 'Pendiente:', m, 'Ordenada:', c, 'r2:', r_squared)
                        print('temperatura pendiente:', hc/(k*round(m)), 'de ordenada:', hc/(532*k*round(c)))
                        
                        Temp_BE_1[i] = hc/(round(m)*k)
                        err_Temp_BE[i] = hc*err_m/(k*round(m)**2)
                  
           # plt.ylabel('Bose-einstein')
           # plt.xlabel('Wavelength (nm)')
                    
            plt.ylabel('Inside bose-einstein')
            plt.xlabel('1/Wavelength (1/nm)')
            figure_name = os.path.join(common_path, 'Anti-Stokes_spectrum_%s.png' % (NP)) 
            plt.savefig(figure_name , dpi = 400)
            plt.close()
              
            Temp_BE = Temp_BE_1[np.where(Temp_BE_1>0)[0]]
            err_Temp_BE = err_Temp_BE[np.where(Temp_BE_1>0)[0]]
            time_BE = frame_list[np.where(Temp_BE_1>0)[0]]
            londa_max_pl =  londa_max_pl[np.where(Temp_BE_1>0)[0]]
            
            plt.style.use('default')
            
            plt.figure()
           # plt.errorbar(time_BE , Temp_BE, yerr = err_Temp_BE, fmt = '-*', label = 'fitting bose-einstein')
            plt.plot(time_BE , Temp_BE,'-*', label = 'fitting')
            plt.xlabel('Time (s)')
           # plt.xlim(-2, 242)
            plt.ylabel('Temperature (K)')
            plt.legend(loc = 'upper right')
            figure_name = os.path.join(common_path, 'Temperature_Time_%s.png' % (NP)) 
            plt.savefig(figure_name , dpi = 400)
            plt.close()
            
            plt.figure()
            plt.plot(londa_max_pl , Temp_BE,'-*', label = 'fitting')
            plt.xlabel('Wavelength SPR (nm)')
            plt.ylabel('Temperature (K)')
           # plt.xlim(-2, 242)
            plt.legend(loc = 'upper right')
            figure_name = os.path.join(common_path, 'Temperature_Wavelength_%s.png' % (NP)) 
            plt.savefig(figure_name , dpi = 400)
            plt.close()
        
    return frame_list, wavelength, matrix_total, NP

def plot_3D(path_to, frame_list, wavelength, matrix, NP):
    
    print('Ploteo 3D', NP)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
 
    for k in range(len(frame_list)):
        # Generate the random data for the y=k 'layer'.
        xs = wavelength
        ys = matrix[k, :]
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax.bar(xs, ys, zs=k, zdir='y', alpha=0.2)
        
   # ax.plot_wireframe(wavelength, frame_list, matrix[k, :])
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Intensity')
    # On the y axis let's only label the discrete values that we have data for.
    ax.set_yticks(frame_list)
    
    figure_name = os.path.join(path_to, '3D_spectrums_%s.png' % (NP)) 
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    return


if __name__ == '__main__':

  #  base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PySpectrum/'
  
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'

  #  daily_folder = '2021-03-26 (Growth)/20210326-132548_Growth_12x2_560'
   # daily_folder = '2021-03-26 (Growth)/20210326-141841_Growth_12x2_570'
    daily_folder = '2021-03-26 (Growth)/20210326-152533_Growth_12x2_580'
    
    parent_folder = os.path.join(base_folder, daily_folder)
    list_of_folders = os.listdir(parent_folder)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
   # list_of_folders = [f for f in list_of_folders if re.search('Liveview_Spectrum',f) and not re.search('Kinetics',f)]
    list_of_folders = [f for f in list_of_folders if re.search('Liveview_Spectrum',f)]
    
    list_of_folders.sort()
    
    L_folder = len(list_of_folders)
    
    #INPUTS
    
    sustrate_bkg_bool = True #True Resta la señal de Background (region de mismo size pero fuera de la fibra optica, en pixel 200)
    
    path_to = os.path.join(parent_folder, 'processed_data_sustrate_bkg_%s'%str(sustrate_bkg_bool))
    
    if not os.path.exists(path_to):
        os.makedirs(path_to)
        
    first_wave = 501 #region integrate_antistokes
    starts_notch = 521 #region integrate_antistokes
    
    ends_notch = 546 #  #550 gold  #sustrate 700    #water  625 #antistokes 500
    last_wave = 640 # #570 gold  #sustrate 725   #water 670  #antistokes 521
    
    window, deg, repetitions = 21, 0, 1
    
    exposure_time = 1 #s
    spectrum_time = exposure_time #1.5*exposure_time en caso de Run till Abort
    
    trace_BS_bool = True #si no se grabó la traza del fotodiodo
    calibration_BS = np.array([4.12, -0.025]) #slope, intercept
    
    plot_antistokes = False # #True: cuando se mide la parte antistokes
    
    bool_fit_lorentz = True
    bool_fit_bose_einstein = False
    
    fit_raman_water = True
    
    plot_all = False
    for f in list_of_folders:
        folder = os.path.join(parent_folder,f)
        frame_list, wavelength, matrix_total, NP = process_spectrum(folder, path_to, first_wave, starts_notch, ends_notch, last_wave, window, deg, repetitions, spectrum_time,  fit_raman_water, plot_antistokes, bool_fit_lorentz, bool_fit_bose_einstein, calibration_BS, sustrate_bkg_bool, trace_BS_bool)
        plot_all = True
     
    #plot_3D(path_to, frame_list, wavelength, matrix_total, NP)
    
    if plot_all:
        common_path = path_to
        step2.post_process_spectrum(common_path)
        step2.post2_process_spectrum(common_path)
        step2.post3_process_spectrum(common_path)
        step2.post4_process_spectrum(common_path, 'initial')
        step2.post4_process_spectrum(common_path, 'final')
        step2.post5_process_spectrum(common_path)
    