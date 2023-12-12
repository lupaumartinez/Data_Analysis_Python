#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:49:18 2020

@author: luciana
"""

import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

import scipy.signal as sig

def smooth_Signal(signal, window, deg, repetitions):
    
    k = 0
    while k < repetitions:
        signal = sig.savgol_filter(signal, window, deg, mode = 'mirror')
        k = k + 1
        
    return signal

def glue_steps(wave_PySpectrum, spectrum_py, number_pixel, grade, plot_all_step):

    L = int(len(spectrum_py)/number_pixel) #cantidad de steps
    
    n_skip_points = 30
    n = int(n_skip_points/2)
    
    spec_steps = np.zeros((number_pixel-n_skip_points, L))
    wave_steps = np.zeros((number_pixel-n_skip_points, L))
    
    spec_steps_glue = np.zeros((number_pixel-n_skip_points, L))
    wave_steps_glue = np.zeros((number_pixel-n_skip_points, L))
    
    list_of_inf = np.zeros(L)
    list_of_sup = np.zeros(L)
    
    for i in range(L):
        
        spec = spectrum_py[i*number_pixel:number_pixel*(1+i)]
        wave = wave_PySpectrum[i*number_pixel:number_pixel*(i+1)]
        
        spec_steps[:, i] = spec[n:-n]
        wave_steps[:, i] = wave[n:-n]
        
        spec_steps_glue[:, i] = spec[n:-n]
        wave_steps_glue[:, i] = wave[n:-n]
        
        list_of_inf[i] = wave_steps[0, i] # wave[0]
        list_of_sup[i] = wave_steps[-1, i]
    
    for j in range(L-1):
        
        inf = list_of_inf[j+1]
       # sup = list_of_sup[i]
        
        wave_tail = wave_steps[:, j]   
        desired_range_tail = np.where(wave_tail >= inf)[0]
        
        m = int( len(desired_range_tail))
        
        weigth_h = np.linspace(0, 1, m)**grade
        weigth_t = np.flip(weigth_h)
        
        coef = weigth_h + weigth_t
        
        weigth_h =  weigth_h/coef
        weigth_t =  weigth_t/coef
        
        desired_range_tail = range(number_pixel-n_skip_points - m,  number_pixel-n_skip_points)
        spec_tail = spec_steps[desired_range_tail , j]
        wave_tail = wave_steps[desired_range_tail , j]
        
        desired_range_head = range(0,  m)
        spec_head = spec_steps[desired_range_head, j+1]
        wave_head = wave_steps[desired_range_head, j+1]
        
        spec_weigth = weigth_h*spec_head + weigth_t*spec_tail
        
       # spec_weigth = smooth_Signal(spec_weigth, window = 21, deg = 0, repetitions = 1)

        spec_steps_glue[desired_range_tail, j] = spec_weigth
        wave_steps_glue[desired_range_tail, j] = wave_tail
        
        spec_steps_glue[desired_range_head, j+1] = spec_weigth
        wave_steps_glue[desired_range_head, j+1] = wave_head
        
    wave_final = np.reshape(wave_steps_glue, [1,wave_steps_glue.size])[0]
    spectrum_final = np.reshape(spec_steps_glue, [1,wave_steps_glue.size])[0]
    
    spectrum_final = [spectrum_all for _,spectrum_all in sorted(zip(wave_final,spectrum_final))]
    wave_final = np.sort(wave_final)
    spectrum_final = np.array(spectrum_final)
    
    wave_final, spectrum_final = interpole_spectrum(wave_final, spectrum_final, number_pixel)
    
    if plot_all_step:
    
        plt.figure()
        plt.plot(wave_final, spectrum_final, 'ko-')
        
        for i in range(L):
            plt.plot(wave_steps[:,i], spec_steps[:,i], 'o')
            
        plt.show()

    return wave_final, spectrum_final


def select_ROI(image, center_row, spot_size):
    
    down_row = center_row - int((spot_size-1)/2)
    up_row = center_row + int((spot_size-1)/2) + 1  
    roi_rows = range(down_row, up_row)
    
    spectrum = np.round(np.mean(image[:,roi_rows], axis=1),2)
    
    return spectrum

def glue_photos(wavelength, image, number_pixel, grade, plot_all_step):
    
    large = image.shape[1]
    
    #image_glue = np.zeros((image.shape[0], image.shape[1]))
    
    # factor = number_pixel/window_wavelength #103
    
  #  desired_points = round(factor*(wavelength[-1] - wavelength[0]))
  
    desired_points = number_pixel
    
    image_glue = np.zeros((desired_points, desired_points))
    
  #  plt.figure()
    
    for i in range(large):
        spectrum_row = image[:, i]
        wave_final, spectrum_final = glue_steps(wavelength, spectrum_row, number_pixel, grade, plot_all_step)
  # plt.plot(wave_final, spectrum_final)
        image_glue[:, i] = spectrum_final
  #  plt.show()
    
    return wave_final, image_glue

def interpole_spectrum(wavelength, spectrum, number_pixel):
    
   # factor = number_pixel/window_wavelength
    
  #  desired_points = round(factor*(wavelength[-1] - wavelength[0]))
    
    desired_points = number_pixel
    
    lower_lambda = wavelength[0]
    upper_lambda = wavelength[-1]
    
    wavelength_new = np.linspace(lower_lambda, upper_lambda, desired_points)

    spectrum_new = np.interp(wavelength_new, wavelength, spectrum)
    
    return wavelength_new, spectrum_new

if __name__ == '__main__':

    lampara_analize = False#False#True# False
    luminisence_analize = False#True#False# True
    
    plt.close('all')
    
    base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PySpectrum/'
   
    #PySpectrum
    
    if luminisence_analize:
    
        daily_folder = '2020-01-08 (lampara IR PySpectrum y espejo de oro)/espejo_oro/'
        subfolder = 'barrido_z_1um/Spectrum_luminescence_Col_001_NP_012'
        file_tiff_dark =  os.path.join(base_folder, daily_folder, '20200108-171911_Spectrum_Measurment_step_05_dark/20200108_171911_Picture_Andor_Spectrum_step_0005.tiff')
        
        daily_folder = '2020-01-06 (PL PX 532nm of growth)'
        subfolder = '20200106-151410_Luminescence_Steps_10x10/Spectrum_luminescence_Col_001_NP_006'
        file_tiff = os.path.join(base_folder, daily_folder, subfolder, '20200106_151656_Picture_Andor_Spectrum_step_0005.tiff')
        file_txt = os.path.join(base_folder, daily_folder, subfolder, '20200106_151656_Calibration_Shamrock_Spectrum_step_0005.txt') 
        file_tiff_sustrate = os.path.join(base_folder, daily_folder, '20200106-151317_Spectrum_Measurment_step_05_sustrate/20200106_151317_Picture_Andor_Spectrum_step_0005.tiff')
        
        image_PySpectrum_sustrate = np.transpose(io.imread(file_tiff_sustrate))
        
        center_row = 439
        spot_size = 39

    
    if lampara_analize:
        
        daily_folder = '2020-01-08 (lampara IR PySpectrum y espejo de oro)/espejo_oro/'
        subfolder = 'barrido_z_1um/Spectrum_luminescence_Col_001_NP_012'
        file_tiff = os.path.join(base_folder, daily_folder, subfolder, '20200108_162702_Picture_Andor_Spectrum_step_0005.tiff')
        file_txt = os.path.join(base_folder, daily_folder, subfolder, '20200108_162702_Calibration_Shamrock_Spectrum_step_0005.txt') 
        file_tiff_dark =  os.path.join(base_folder, daily_folder, '20200108-171911_Spectrum_Measurment_step_05_dark/20200108_171911_Picture_Andor_Spectrum_step_0005.tiff')
        
        #Lampara IR
        
        daily_folder = '2020-01-08 (lampara IR PySpectrum y espejo de oro)/lamparaIR'
        
        #AndorSolis
        
        file_asc = os.path.join(base_folder, daily_folder, 'Andor_Solis_Slit_agosto_2019', 'lamp_IR_unpol.asc')
        
        andorSolis = np.loadtxt(file_asc, dtype='float')
        wave_andorSolis = np.array(andorSolis [:, 0])
        image_andorSolis = np.array(andorSolis[:, 1:])
        
        center_row = 890#600
        spot_size = 50#201
        
        spectrum_andor = select_ROI(image_andorSolis, center_row, spot_size)
        norm_Solis = spectrum_andor/max(spectrum_andor)
        
         #Lampara IR unpol
        save_folder = os.path.join(base_folder, daily_folder, 'Slit150um/IR_unpol/1s/')
        file_tiff = os.path.join(save_folder, '20200108-143246_Spectrum_Measurment_step_08_IR_unpol', '20200108_143246_Picture_Andor_Spectrum_step_0008.tiff')
        file_txt = os.path.join(save_folder, '20200108-143246_Spectrum_Measurment_step_08_IR_unpol', '20200108_143246_Calibration_Shamrock_Spectrum_step_0008.txt')
        
        file_tiff_dark = os.path.join(save_folder, '20200108-143336_Spectrum_Measurment_step_08_dark', '20200108_143336_Picture_Andor_Spectrum_step_0008.tiff')
        
        #save_folder = os.path.join(base_folder, daily_folder, 'Slit150um/IR_unpol/0.5s/')
        #file_tiff = os.path.join(save_folder, '20200108-143110_Spectrum_Measurment_step_08_IR_unpol', '20200108_143110_Picture_Andor_Spectrum_step_0008.tiff')
        #file_txt = os.path.join(save_folder, '20200108-143110_Spectrum_Measurment_step_08_IR_unpol', '20200108_143110_Calibration_Shamrock_Spectrum_step_0008.txt')
        
        #file_tiff_dark = os.path.join(save_folder, '20200108-143411_Spectrum_Measurment_step_08_dark', '20200108_143411_Picture_Andor_Spectrum_step_0008.tiff')
        
        #PX
        #save_folder = os.path.join(base_folder, daily_folder, 'Slit150um/IR_pol_PX/2s/')
        #file_tiff = os.path.join(save_folder, '20200108-150242_Spectrum_Measurment_step_08', '20200108_150242_Picture_Andor_Spectrum_step_0008.tiff')
        #file_txt = os.path.join(save_folder, '20200108-150242_Spectrum_Measurment_step_08', '20200108_150242_Calibration_Shamrock_Spectrum_step_0008.txt')
        
        #file_tiff_dark = os.path.join(save_folder, '20200108-150414_Spectrum_Measurment_step_08_dark', '20200108_150414_Picture_Andor_Spectrum_step_0008.tiff')
        
        #PY
       # save_folder = os.path.join(base_folder, daily_folder, 'Slit150um/IR_pol_PY/2s/')
       # file_tiff = os.path.join(save_folder, '20200108-145205_Spectrum_Measurment_step_08', '20200108_145205_Picture_Andor_Spectrum_step_0008.tiff')
       # file_txt = os.path.join(save_folder, '20200108-145205_Spectrum_Measurment_step_08', '20200108_145205_Calibration_Shamrock_Spectrum_step_0008.txt')
        
        #file_tiff_dark = os.path.join(save_folder, '20200108-145417_Spectrum_Measurment_step_08_dark', '20200108_145417_Picture_Andor_Spectrum_step_0008.tiff')
        
        plt.figure()
        plt.title('AndorSolis')
        plt.imshow(image_andorSolis)
        
        plt.figure(100)
        plt.plot(wave_andorSolis, norm_Solis , 'ro-', label = 'Andor Solis')
           # fig_name = os.path.join(save_folder, 'fig_lamps_weigth_grade_%02e.png'%grade)
          #  plt.savefig(fig_name)
        
    image_PySpectrum = np.transpose(io.imread(file_tiff))
    wave_PySpectrum  = np.array(np.loadtxt(file_txt))
    
    image_PySpectrum_dark = np.transpose(io.imread(file_tiff_dark))
    
    image_PySpectrum = image_PySpectrum - image_PySpectrum_dark
        
    grade = 3
    number_pixel = 1002
    
    #First ROI, then Glue
    spectrum_py = select_ROI(image_PySpectrum, center_row, spot_size)
    wave_final, spectrum_final = glue_steps(wave_PySpectrum, spectrum_py, number_pixel, grade, plot_all_step = True)
    norm_PySpectrum = spectrum_final/max(spectrum_final)
    
    #First Glue, then ROI
    wave_final_glue, image_glue = glue_photos(wave_PySpectrum, image_PySpectrum, number_pixel, grade, plot_all_step = False)
    spectrum_photo = select_ROI(image_glue, center_row, spot_size)
    norm_spectrum_photo = spectrum_photo/max(spectrum_photo)
 
    plt.figure()
    plt.title('PySpectrum stack')
    plt.imshow(image_PySpectrum)
    
    plt.figure()
    plt.title('PySpectrum Glue')
   # plt.yticks(wave_final_glue)
    plt.imshow(image_glue)
    
    plt.figure(100)
    #plt.plot(wave_final, norm_PySpectrum, 'b*-', label = 'First ROI, then glue')
    plt.plot(wave_final, norm_PySpectrum, 'k*-', label = 'First ROI, then glue, then interpole')
    plt.plot(wave_final_glue, norm_spectrum_photo, 'go--', label = 'First Glue, then interpole, then ROI')
   # fig_name = os.path.join(save_folder, 'fig_lamps_weigth_grade_%02e.png'%grade)
  #  plt.savefig(fig_name)
    plt.legend()
    plt.show()
    
#%%

    header_txt = 'Wavelenth_PySpectrum (nm), Spectrum'
    name_data = os.path.join(save_folder,'lamps_PySpectrum_interpole_weigth_grade_%02e.txt'%(grade))
    data = np.array([list(wave_final), list(norm_PySpectrum)]).T
    np.savetxt(name_data, data, fmt='%s', header = header_txt)
    
    header_txt = 'Wavelength_Solis (nm), Spectrum_Andor_Solis'
    name_data = os.path.join(save_folder,'lamps_AndorSolis.txt')
    data = np.array([list(wave_andorSolis), list(norm_Solis)]).T
    np.savetxt(name_data, data, fmt='%s', header = header_txt)
    
    plt.figure()
    plt.plot(wave_andorSolis, norm_Solis , 'ro-', label = 'Andor Solis')
    plt.plot(wave_final, norm_PySpectrum, 'bo-', label = 'PySpectrum')
    plt.legend()
    fig_name = os.path.join(save_folder, 'fig_lamps_interpole_weigth_grade_%02e.png'%grade)
    plt.savefig(fig_name)
    plt.show()

#%%
    name = os.path.join(save_folder, 'lamps_AndorSolis.txt')
    data = np.loadtxt(name, comments = '#', delimiter = None)
                      
    wave_Andor = data[:, 0]
    s_Andor = data[:, 1]
                     
    import re
    
    list_of_folders = os.listdir(save_folder)
    list_of_folders = [f for f in list_of_folders if re.search('lamps_PySpectrum_interpole_weigth_grade',f)]
    list_of_folders.sort()
    
    l = len(list_of_folders)
        
    plt.figure()
    plt.plot(wave_Andor, s_Andor, 'k--', label = 'Andor Solis')
    
    for i in range(l):
        
        grade = list_of_folders[i].split('grade_')[-1].split('.txt')[0]
    
        name = os.path.join(save_folder, list_of_folders[i])
        data = np.loadtxt(name, comments = '#')
        
        wave = data[:, 0]
        s = data[:, 1]
        
        plt.plot(wave, s, label = 'grade_%s'%grade)
        plt.show()
        
    plt.legend()
