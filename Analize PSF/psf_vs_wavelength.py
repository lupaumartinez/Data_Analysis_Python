# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:14:24 2019

@author: Luciana Martinez

CIBION, Bs As, Argentina

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.optimize import curve_fit
from skimage import io

def gaussian(x, Io, xo, wo, C):

    return Io*np.exp(-2*(x-xo)**2/(wo**2)) + C

def fit_gaussian(gaussian, x, y, rango):
    
    #mean = sum(x * y)
    	#sigma = sum(y * (x - mean)**2)
    #sigma = np.sqrt(sum(y*(x - mean)**2))
    
    mean = 0
    sigma = 300
    
    bounds = ([0, -rango/2, 200, 0], [1, rango/2, 380, 1])

    popt, pcov = curve_fit(gaussian, x, y, p0 = [1, mean, sigma, 0], bounds = bounds)
        
    #x_fit = np.linspace(x[0], x[-1], 100)
    pixel_size = (x[-1]-x[0])/100 
    x_fit = np.arange(x[0], x[-1], pixel_size)
    
    y_fit = gaussian(x_fit, popt[0], popt[1], popt[2], popt[3])
    
    return [x_fit, y_fit], popt

def curve_gauss(image, rango):

	profile_x = np.mean(image, axis = 1)
	profile_y = np.mean(image, axis = 0)

	#axe = np.linspace(-rango/2, rango/2, image.shape[0])
	pixel_size = rango/image.shape[0] # in nm
	axe = np.arange(-rango/2 , rango/2, pixel_size)

	return [axe, profile_x, profile_y]

def select_confocal_range(wavelength, confocal_spectrum, start_wave, end_wave):
     
    desired_range = np.where((wavelength > start_wave) & (wavelength < end_wave))

    image = np.zeros((confocal_spectrum.shape[0], confocal_spectrum.shape[1]))
    
    for i in range(confocal_spectrum.shape[0]):
    	for j in range(confocal_spectrum.shape[1]):

    		spectrum_ij = confocal_spectrum[i, j, :]
    		intensity = np.mean(spectrum_ij[desired_range])

    		image[i, j] = intensity

    return image

def open_spectrum(folder, image_size_px):

    list_of_files = os.listdir(folder)
    wavelength_filename = [f for f in list_of_files if re.search('wavelength',f)]
    list_of_files.sort()
    list_of_files = [f for f in list_of_files if not os.path.isdir(folder+f)]
    list_of_files = [f for f in list_of_files if ((not os.path.isdir(folder+f)) \
                                                  and (re.search('Spectrum_',f)))]
    L = len(list_of_files)            
    
    data_spectrum = []
    name_spectrum = []
    specs = []
        
    print(L, 'spectra were acquired.')
    
    for k in range(L):
        name = os.path.join(folder,list_of_files[k])
        data_spectrum = np.loadtxt(name)
        name_spectrum.append(list_of_files[k])
        specs.append(data_spectrum)
    
    wavelength_filepath = os.path.join(folder,wavelength_filename[0])
    londa = np.loadtxt(wavelength_filepath)

    # ALLOCATING on CONFOCAL
    camera_px_length = 1002

    matrix_spec_raw = np.zeros((image_size_px,image_size_px,camera_px_length))
   
    for i in range(image_size_px):
        for j in range(image_size_px):
            matrix_spec_raw[i,j,:] = np.array(specs[i*image_size_px+j])
    del specs

    return  londa, matrix_spec_raw

def open_confocal_ph(file):
    
    image = io.imread(file)

    return image    
    

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

def do_histogram(direction):
    
    list_of_files = os.listdir(direction)
    
 #   list_of_files_ph = [f for f in list_of_all_files if re.search('NPscan',f) and not re.search('gone',f) and not re.search('back',f) ]
    list_of_files = [f for f in list_of_files if re.search('.txt',f)]
    list_of_files.sort()
    
    L = len(list_of_files)
    
    wx = np.zeros(L)
    wy = np.zeros(L)
    
    for i in range(L):
        name = os.path.join(direction, list_of_files[i])
        data = np.loadtxt(name, comments = '#', delimiter = ' ')
        wx[i] = np.mean(data[:, 1])
        wy[i] = np.mean(data[:, 2])          
        
    print('mean w fast', np.mean(wx), 'mean w slow', np.mean(wy))
        
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    ax1.plot(wy, wx, '.')
    ax1.set_xlabel('W slow [nm]')
    ax1.set_ylabel('W fast [nm]')
    ax1.set_xlim(200,400)
    ax1.set_ylim(200,400)
    ax1.set_aspect('equal')
    
    ax2.hist(wx, bins=10, range=[200,400], density = True, rwidth=0.9, color='C1')
    ax2.set_xlabel('W fast [nm]')
    ax2.set_ylim(0, 0.02)
    ax2.set_xlim(200,400)
    
    ax3.hist(wy, bins=10, range=[200,400], density = True, rwidth=0.9, color='C2')
    ax3.set_xlabel('W slow [nm]')
    
    ax3.set_ylim(0, 0.02)
    ax3.set_xlim(200,400)
    
    f.set_tight_layout(True)
    plt.savefig(direction + '/hist_spectrum_Gauss_omega.jpg', dpi = 500)
    plt.close()
    
    return
    

if __name__ == '__main__':
    
    #direction = 'C:/Users/Alumno/Dropbox/Simule Drift with Confocal Spectrum/number_pixel_12_pixel_time_0.8/Confocal_Spectrum_Col_001_NP_001'
    
    prefix_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2019/Mediciones_PySpectrum/'
 
    local_folder  = '2019-12-20 (Confocal Spectrum, veo psf de SS 50 nm)/20191219-182356_Luminescence_10x10'
   
    direction = os.path.join(prefix_folder, local_folder)
    
    save_folder = os.path.join(direction, 'processed_data_psf')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    image_size_px = 12
    ##Select waves to measurment psf
    starts_notch = 525
    ends_notch = 545
    plot_psf = False #to se all plot, adjusment of gauss en each step_wave
    plot_range = 'stokes' #'stokes', anti-stokes', 'laser', 'all'
    
    #File
    list_of_all_files = os.listdir(direction)
    
 #   list_of_files_ph = [f for f in list_of_all_files if re.search('NPscan',f) and not re.search('gone',f) and not re.search('back',f) ]
    list_of_files_ph = [f for f in list_of_all_files if re.search('back_NPscan',f)]
    list_of_files_ph.sort()
    
    L_confocal = len(list_of_files_ph)
    
    list_of_files = [f for f in list_of_all_files if re.search('Slow_Confocal',f)]
    list_of_files.sort()
    
    folder = os.path.join(direction, list_of_files[0])
    
    wavelength, matrix_spectrum = open_spectrum(folder, image_size_px)

    #Mode select wavelength
    
    if plot_range == 'stokes':
        bin_wave = 2 #nm
        range_wave = np.arange(ends_notch, wavelength[-1], bin_wave)
    
    elif plot_range == 'anti-stokes':
        bin_wave = 2 #nm
        range_wave = np.arange(wavelength[0], starts_notch, bin_wave)
        
    elif plot_range == 'laser':
        bin_wave = 2 #nm
        range_wave = np.arange(starts_notch, ends_notch, bin_wave)
        
    elif plot_range == 'all':
        bin_wave = 2 #nm
        range_wave = np.arange(wavelength[0], wavelength[-1], bin_wave)
        
    L = len(range_wave)
        
    #automatic all files:
        
    for j in range(L_confocal):
        
        file_confocal =  os.path.join(direction, list_of_files_ph[j])
        matrix_confocal_ph = open_confocal_ph(file_confocal)
        
        w_x_ph = 0
        w_y_ph = 0
                    
        image_curve_gauss = curve_gauss(matrix_confocal_ph , rango = 2000)
        
        try:
            
            image_fitx, ajuste_x = fit_gaussian(gaussian, image_curve_gauss[0], (image_curve_gauss[1]) / (max(image_curve_gauss[1])) , rango = 2000)
            image_fity, ajuste_y = fit_gaussian(gaussian, image_curve_gauss[0], (image_curve_gauss[2]) / (max(image_curve_gauss[2])) , rango = 2000)
         
            w_x_ph = round(ajuste_x[2],3)
            w_y_ph = round(ajuste_y[2],3)
        
        except: pass
        
        print('Gauss ajuste fotodiodo:', 'w fast', w_x_ph, 'w slow',  w_y_ph)
        
        if plot_psf:
       
            fig, (ax3_1, ax3_2) = plt.subplots(1, 2)
        
            ax3_1.imshow(matrix_confocal_ph)
            ax3_1.set_title('Image acquisition')
            ax3_1.grid(False)
        
            ax3_2.plot(image_curve_gauss[0], image_curve_gauss[1]/max(image_curve_gauss[1]), 'ro',label = 'Fast axis')
            ax3_2.plot(image_fitx[0], image_fitx[1], 'r-', alpha=0.5)
            ax3_2.plot(image_curve_gauss[0], image_curve_gauss[2]/max(image_curve_gauss[2]), 'bo', label = 'Slow axis')
            ax3_2.plot(image_fity[0], image_fity[1], 'b-', alpha=0.5)
            ax3_2.set_ylim(0, 1.2)
            ax3_2.legend(loc = 'upper right')
        
            fig.set_tight_layout(True)
        
            plt.show()
        
        name_NP = list_of_files[j].split('Spectrum_')[1]
        folder = os.path.join(direction, list_of_files[j])
        wavelength, matrix_spectrum = open_spectrum(folder, image_size_px)
        
        w_x = np.zeros(L)
        w_y = np.zeros(L)
        
        for i in range(L):
    
            start_wave = range_wave[i]
            end_wave = range_wave[i] + bin_wave
    
            image = select_confocal_range(wavelength, matrix_spectrum, start_wave, end_wave)
    
            image_curve_gauss = curve_gauss(image, rango = 800)
          
            try:
                image_fitx, ajuste_x = fit_gaussian(gaussian, image_curve_gauss[0], (image_curve_gauss[1]) / (max(image_curve_gauss[1])) , rango = 800)
                image_fity, ajuste_y = fit_gaussian(gaussian, image_curve_gauss[0], (image_curve_gauss[2]) / (max(image_curve_gauss[2])) , rango = 800)
             
                w_x[i] = round(ajuste_x[2],3)
                w_y[i] = round(ajuste_y[2],3)
            
            except: pass
            
            print('Gauss ajuste:', 'w fast', w_x[i], 'w slow',  w_y[i])


            if plot_psf:
               
                fig, (ax3_1, ax3_2) = plt.subplots(1, 2)
    
                ax3_1.imshow(image)
                ax3_1.set_title('Image acquisition')
                ax3_1.grid(False)
    
                ax3_2.plot(image_curve_gauss[0], image_curve_gauss[1]/max(image_curve_gauss[1]), 'ro',label = 'Fast axis')
                ax3_2.plot(image_fitx[0], image_fitx[1], 'r-', alpha=0.5)
                ax3_2.plot(image_curve_gauss[0], image_curve_gauss[2]/max(image_curve_gauss[2]), 'bo', label = 'Slow axis')
                ax3_2.plot(image_fity[0], image_fity[1], 'b-', alpha=0.5)
                ax3_2.set_ylim(0, 1.2)
                ax3_2.legend(loc = 'upper right')
    
                fig.set_tight_layout(True)
    
                plt.show()
                
        psf_w = np.array([range_wave, w_x, w_y]).T
        name = os.path.join(save_folder, 'psf_%s.txt'%name_NP)
        header_txt = 'wavelength, w_fast, w_slow'
        np.savetxt(name, psf_w, header = header_txt)
                
        plt.figure()
        plt.plot(range_wave, w_x, 'ro',  alpha=0.5,label = 'Fast axis')
        plt.plot(range_wave, w_y, 'bo', alpha=0.5, label = 'Slow axis')
        plt.scatter(532, w_x_ph, s = 20, c = 'r', marker = 's')
        plt.scatter(532, w_y_ph, s = 20, c = 'b', marker = 's')
        plt.axhline(w_x_ph, color = 'r', linestyle = '--')
        plt.axhline(w_y_ph, color = 'b', linestyle = '--')
        plt.plot(range_wave, range_wave/2, 'g-', label = 'Abbe diffaction limit')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('PSF: Gauss w (nm)')
        plt.ylim(260, 400)
        plt.xlim(530, 620)
        plt.legend(loc = 'upper right')
        name_fig = os.path.join(save_folder, 'fig_psf_%s.png'%name_NP)
        plt.savefig(name_fig, dpi = 400)
        plt.show()        
        
        
#%%
        
do_histogram(save_folder)

