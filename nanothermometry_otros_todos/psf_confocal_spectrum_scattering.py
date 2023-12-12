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
    sigma = 6000
    
    bounds = ([0, -rango/2, 1000, 0], [1, rango/2, 20000, 1])

    popt, pcov = curve_fit(gaussian, x, y, p0 = [1, mean, sigma, 0], bounds = bounds)
        
    #x_fit = np.linspace(x[0], x[-1], 100)
    pixel_size = (x[-1]-x[0])/100 
    x_fit = np.arange(x[0], x[-1], pixel_size)
    
    y_fit = gaussian(x_fit, popt[0], popt[1], popt[2], popt[3])
    
    return [x_fit, y_fit], popt

    
def gaussian2D(grid, amplitude, x0, y0, σ_x, σ_y, offset, theta=0):
    
    # TO DO (optional): change parametrization to this one
    # http://mathworld.wolfram.com/BivariateNormalDistribution.html  
    # supposed to be more robust for the numerical fit
    
    x, y = grid
    x0 = float(x0)
    y0 = float(y0)   
    a = (np.cos(theta)**2)/(2*σ_x**2) + (np.sin(theta)**2)/(2*σ_y**2)
    b = -(np.sin(2*theta))/(4*σ_x**2) + (np.sin(2*theta))/(4*σ_y**2)
    c = (np.sin(theta)**2)/(2*σ_x**2) + (np.cos(theta)**2)/(2*σ_y**2)
    G = offset + amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                            + c*((y-y0)**2)))
    return G.ravel()

def curve_gauss(image, rango):

	profile_x = np.mean(image, axis = 1)
	profile_y = np.mean(image, axis = 0)

	#axe = np.linspace(-rango/2, rango/2, image.shape[0])
	pixel_size = rango/image.shape[0] # in nm
	axe = np.arange(-rango/2 , rango/2, pixel_size) #+ pixel_size/2

	return [axe, profile_x, profile_y]

def fit_gauss2D(image, rango):
    
    pixel_size = rango/image.shape[0] # in nm
    x = np.arange(-rango/2 , rango/2, pixel_size) #+ pixel_size/2 #nm
    y = x
    
    [Mx, My] = np.meshgrid(x, y)
    
    dataG_2d = (image- np.min(image) ) / (np.max(image) -  np.min(image))
    dataG_ravel = dataG_2d.ravel()
    
    initial_sigma = 3000 #nm
    
    #primero va lo de y, luego lo de x
    initial_guess_G = [1, 0, 0, initial_sigma, initial_sigma, 0]
    bounds = ([0, x[0], y[0], 0, 0, 0], [1, x[-1], y[-1], 4*initial_sigma, 4*initial_sigma, 1])
    
    poptG, pcovG = curve_fit(gaussian2D, (Mx, My), dataG_ravel, p0= initial_guess_G, bounds = bounds)
    
    poptG = np.around(poptG, 2)
    
    #cog =  poptG[2] - x[0], poptG[1] - y[0] 
    
    w_nm = 2*poptG[4], 2*poptG[3]
    
    w_nm = np.around(w_nm, 1)
    
    wx = w_nm[0]
    wy= w_nm[1]
    
    print('w fast', wx, 'w slow', wy)
    
    return wx, wy

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
    list_of_files = [f for f in list_of_files if re.search('txt',f)]
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

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

def do_histogram_psf_PL(direction):
    
    list_of_files = os.listdir(direction)
    list_of_files = [f for f in list_of_files if re.search('.txt',f) and re.search('Col',f)]
    list_of_files.sort()
    
    L = len(list_of_files)
    
    wx = np.zeros(L)
    wy = np.zeros(L)
    
    for i in range(L):
        name = os.path.join(direction, list_of_files[i])
        data = np.loadtxt(name, comments = '#', delimiter = ' ')

        if data.shape[0] > 3:
            wx[i] = np.mean(data[:, 1])
            wy[i] = np.mean(data[:, 2])
        else:
            wx[i] = data[1]
            wy[i] = data[2]
        
    plot_histogram(direction, wx, wy, 'all_Gauss_omega')
    
    return
    
def plot_histogram(direction, wx, wy, name):
    
    mean_wx = round(np.mean(wx))
    std_wx = round(np.std(wx))
    
    mean_wy = round(np.mean(wy))
    std_wy = round(np.std(wy))
    
    print('mean w fast', round(np.mean(wx)), round(np.std(wx)), 
          'mean w slow', round(np.mean(wy)), round(np.std(wy)))
    
    text_x = u'%d $\pm$ %d'%(round(np.mean(wx)),  round(np.std(wx)))
    text_y = u'%d $\pm$ %d'%(round(np.mean(wy)),  round(np.std(wy)))
    
    header = 'mean w fast (nm), std w fast (nm), mean w slow (nm), std w slow (nm)'
    data = np.array([mean_wx, std_wx, mean_wy, std_wy])
    namefile = direction + '/hist_' + '%s.txt'%name
    np.savetxt(namefile, data, header = header, fmt = '%.f')
       
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    ax1.plot(wy, wx, 'k.')
    ax1.set_xlabel('W slow [nm]')
    ax1.set_ylabel('W fast [nm]')
    ax1.set_xlim(200,400)
    ax1.set_ylim(200,400)
    ax1.set_aspect('equal')
    
    ax2.hist(wx, bins=10, range=[200,400], density = True, rwidth=0.9, color='r', alpha = 0.5)
    ax2.text(240, 0.045, text_x, fontsize = 14, color = 'r', alpha = 0.5)
    ax2.set_xlabel('W fast [nm]')
    ax2.set_ylim(0, 0.05)
    ax2.set_xlim(200,400)
    
    ax3.hist(wy, bins=10, range=[200,400], density = True, rwidth=0.9, color='b', alpha = 0.5)
    ax3.text(240, 0.045, text_y, fontsize = 14, color = 'b', alpha = 0.5)
    ax3.set_xlabel('W slow [nm]')
    
    ax3.set_ylim(0, 0.05)
    ax3.set_xlim(200,400)
    
    f.set_tight_layout(True)
    plt.savefig(direction + '/hist_'+'%s.jpg'%name, dpi = 500)
    plt.close()
    
    plt.figure()
    plt.plot(wx, 'ro', alpha = 0.5, label = 'Fast axis')
    plt.plot(wy, 'bo', alpha = 0.5, label = 'Slow axis')
    plt.ylabel('W [nm]')
    plt.ylim(200, 400)
    plt.legend()
    plt.savefig(direction + '/' + '%s.jpg'%name, dpi = 500)
    plt.close()
    
    
    return

#%%    

if __name__ == '__main__':
    
    plt.close('all')
    
    base_folder =  r'C:\Ubuntu_archivos\Printing'
    
    daily_folder = r'2022-06-21 Scattering Au60 fibra optica\PSF lampara'
    NP_folder = '9umx9um_27x27'
    
    direction = os.path.join(base_folder, daily_folder, NP_folder)
    
    exceptions_NP = [] #Notacion: ['001'] o ['001', '009']
    
    save_folder = os.path.join(direction, 'processed_data_psf')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    confocal_2DGauss = True #opcional, guarda aparte una carpeta con mismo analisis pero ajuste 2D de la imagen
   
    size = 9*1000#800 #nm
    image_size_px = 27# 10
    
    ##Select waves to measurment psf
    starts_anti = 515
    starts_notch = 520
    ends_notch = 543
    ends_power = 560
    plot_range = 'all' #'stokes', anti-stokes', 'laser', 'all'
    
    plot_psf = ['000']  #['001'] #te muestra el ajuste para esa NP, para cada bin de wavelength
    
    #File
    list_of_all_files = os.listdir(direction)


    list_of_files = [f for f in list_of_all_files if re.search('Slow_Confocal',f)]
    
    print(list_of_files)
    list_of_files.sort()
    list_of_files = list_of_files
    L_confocal = len(list_of_files)
    
    folder = os.path.join(direction, list_of_files[0])
    wavelength, matrix_spectrum = open_spectrum(folder, image_size_px)

    #Mode select wavelength
    
    if plot_range == 'stokes':
        bin_wave = 5 #nm
        range_wave = np.arange(ends_notch, ends_power, bin_wave)
    
    elif plot_range == 'anti-stokes':
        bin_wave = 5 #nm
        range_wave = np.arange(starts_anti, starts_notch, bin_wave)
        
    elif plot_range == 'laser':
        bin_wave = 2 #nm
        range_wave = np.arange(starts_notch, ends_notch, bin_wave)
        
    elif plot_range == 'all':
        bin_wave = 100 #nm
        range_wave = np.arange(starts_anti, ends_power, bin_wave)
        
    L = len(range_wave)
    wave = range_wave + bin_wave/2
        
    #automatic all files:
        
    
    for j in range(L_confocal):
            
        name_NP = list_of_files[j].split('.')[0]
        name_NP = name_NP.split('Spectrum_')[1]
        
        NP = name_NP.split('NP_')[1]
        
        if NP in exceptions_NP:
           continue
        
        folder = os.path.join(direction, list_of_files[j])
       
        wavelength, matrix_spectrum = open_spectrum(folder, image_size_px)
        
        w_x = np.zeros(L)
        w_y = np.zeros(L)
        
        for i in range(L):
    
            start_wave = range_wave[i]
            end_wave = range_wave[i] + bin_wave
    
            image = select_confocal_range(wavelength, matrix_spectrum, start_wave, end_wave)
    
            image_curve_gauss = curve_gauss(image, rango = size)
          
            try:
                image_fitx, ajuste_x = fit_gaussian(gaussian, image_curve_gauss[0], (image_curve_gauss[1]) / (max(image_curve_gauss[1])) , rango = size)
                image_fity, ajuste_y = fit_gaussian(gaussian, image_curve_gauss[0], (image_curve_gauss[2]) / (max(image_curve_gauss[2])) , rango = size)
             
                w_x[i] = round(ajuste_x[2], 1)
                w_y[i] = round(ajuste_y[2], 1)
            
            except: pass
            
            print('Gauss ajuste:', 'w fast', w_x[i], 'w slow',  w_y[i])


            if NP in plot_psf:
               
                fig, (ax3_1, ax3_2) = plt.subplots(1, 2)
                plt.title('%s'%name_NP)
    
                ax3_1.imshow(image)
                ax3_1.set_title('Image acquisition')
                ax3_1.grid(False)
    
                ax3_2.plot(image_curve_gauss[0], image_curve_gauss[1]/max(image_curve_gauss[1]), 'ro',label = 'Fast axis')
                ax3_2.plot(image_fitx[0], image_fitx[1], 'r-', alpha=0.5)
                ax3_2.plot(image_curve_gauss[0], image_curve_gauss[2]/max(image_curve_gauss[2]), 'bo', label = 'Slow axis')
                ax3_2.plot(image_fity[0], image_fity[1], 'b-', alpha=0.5)
              #  ax3_2.set_ylim(0, 1.2)
                ax3_2.legend(loc = 'upper right')
    
                fig.set_tight_layout(True)
    
                plt.show()
                
        psf_w = np.array([wave, w_x, w_y]).T
        name = os.path.join(save_folder, 'psf_%s.txt'%name_NP)
        header_txt = 'wavelength, w_fast, w_slow'
        np.savetxt(name, psf_w, header = header_txt, fmt = '%.1f')
        
        save_each_plot = False
        
        if save_each_plot:
                
            plt.figure()
            plt.title('%s'%name_NP)
            plt.plot(wave, w_x, 'ro',  alpha=0.5,label = 'Fast axis')
            plt.plot(wave, w_y, 'bo', alpha=0.5, label = 'Slow axis')
            plt.xlim(wave[0]-10, wave[-1]+10)
        
            plt.plot(wave, wave/2, 'g-', label = 'Abbe diffaction limit')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('PSF: Gauss w (nm)')
            plt.ylim(260, 400)
            plt.legend(loc = 'upper right')
            name_fig = os.path.join(save_folder, 'fig_psf_%s.png'%name_NP)
            plt.savefig(name_fig, dpi = 400)
            plt.close()  


print('histogram psf from PL stokes')
        
do_histogram_psf_PL(save_folder)

    
#%%

if confocal_2DGauss:
    
    save_folder_2D = os.path.join(save_folder, 'gauss_2D')
    
    if not os.path.exists(save_folder_2D):
        os.makedirs(save_folder_2D)

    for j in range(L_confocal):
            
        name_NP = list_of_files[j].split('.')[0]
        name_NP = name_NP.split('Spectrum_')[1]
        
        NP = name_NP.split('NP_')[1]
        
        print(NP)
        
        if NP in exceptions_NP:
           print('salteo', NP)
           continue
        
        folder = os.path.join(direction, list_of_files[j])
       
        wavelength, matrix_spectrum = open_spectrum(folder, image_size_px)
        
        w_x = np.zeros(L)
        w_y = np.zeros(L)
        
        for i in range(L):
    
            start_wave = range_wave[i]
            end_wave = range_wave[i] + bin_wave
    
            image = select_confocal_range(wavelength, matrix_spectrum, start_wave, end_wave)
            
            try:
                
                wx, wy = fit_gauss2D(image, rango = size)
            
                w_x[i] = wx
                w_y[i] = wy
                
                print('Gauss 2D ajuste:', 'w fast', w_x[i], 'w slow',  w_y[i])
        
               
            except: pass
        
        psf_w = np.array([wave, w_x, w_y]).T
        name = os.path.join(save_folder_2D, '2D_psf_%s.txt'%name_NP)
        header_txt = 'wavelength, w_fast, w_slow'
        np.savetxt(name, psf_w, header = header_txt, fmt = '%.1f')
                
        save_each_plot = False
        
        if save_each_plot:
                
            plt.figure()
            plt.title('%s'%name_NP)
            plt.plot(wave, w_x, 'ro',  alpha=0.5,label = 'Fast axis')
            plt.plot(wave, w_y, 'bo', alpha=0.5, label = 'Slow axis')
            plt.xlim(wave[0]-10, wave[-1]+10)
            
            plt.plot(wave, wave/2, 'g-', label = 'Abbe diffaction limit')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('PSF: Gauss w (nm)')
            plt.ylim(260, 400)
            plt.legend(loc = 'upper right')
            name_fig = os.path.join(save_folder_2D, '2D_fig_psf_%s.png'%name_NP)
            plt.savefig(name_fig, dpi = 400)
            plt.close()
            

    print('histogram 2D psf from PL stokes')
            
    do_histogram_psf_PL(save_folder_2D)