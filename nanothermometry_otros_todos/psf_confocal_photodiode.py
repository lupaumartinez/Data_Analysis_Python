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
	axe = np.arange(-rango/2 , rango/2, pixel_size) + pixel_size/2

	return [axe, profile_x, profile_y]

def fit_gauss2D(image, rango):
    
    pixel_size = rango/image.shape[0] # in nm
    x = np.arange(-rango/2 , rango/2, pixel_size) + pixel_size/2 #nm
    y = x
    
    [Mx, My] = np.meshgrid(x, y)
    
    dataG_2d = (image- np.min(image) ) / (np.max(image) -  np.min(image))
    dataG_ravel = dataG_2d.ravel()
    
    initial_sigma = 150 #nm
    
    #primero va lo de y, luego lo de x
    initial_guess_G = [1, 0, 0, initial_sigma, initial_sigma, 0]
    bounds = ([0, x[0], y[0], 0, 0, 0], [1, x[-1], y[-1], 4*initial_sigma, 4*initial_sigma, 1])
    
    poptG, pcovG = curve_fit(gaussian2D, (Mx, My), dataG_ravel, p0= initial_guess_G, bounds = bounds)
    
    poptG = np.around(poptG, 2)
    
    #cog =  poptG[2] - x[0], poptG[1] - y[0] 
    
    w_nm = 2*poptG[4], 2*poptG[3]
    
    w_nm = np.around(w_nm, 3)
    
    wx = w_nm[0]
    wy= w_nm[1]
    
    print('w fast', wx, 'w slow', wy)
    
    return wx, wy

def open_confocal_ph(file):
    
    image = io.imread(file)

    return image    
    

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path
    
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
    
    base_folder = r'C:\Ubuntu_archivos\Printing'
    
    daily_folder = r'2022-06-22 Au60 satelites Pd\20220622-140802_Luminescence_Load_grid'
    
    direction = os.path.join(base_folder, daily_folder)
    
    save_folder = os.path.join(direction, 'processed_data_psf_ph_mean')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    #File
    list_of_all_files = os.listdir(direction)
    
    list_of_files = [f for f in list_of_all_files if re.search('NPscan',f) and not re.search('back',f)  and not re.search('gone',f) ]
    list_of_files.sort()
    wx_ph = []
    wy_ph = []
   
    L_confocal = len(list_of_files)
    
    confocal_ph_2DGauss = True
    
    exceptions_NP = []
    plot_psf = []
    
    for j in range(L_confocal):
            
        NP = list_of_files[j].split('.')[0]
        name_NP = NP.split('NPscan_')[1]
        
        if NP in exceptions_NP:
           continue
        
        file_confocal =  os.path.join(direction, list_of_files[j])
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
        
        wx_ph.append(w_x_ph)
        wy_ph.append(w_y_ph)
        
        if plot_psf:
       
            fig, (ax3_1, ax3_2) = plt.subplots(1, 2)
            
            plt.title('%s'%name_NP)
        
            ax3_1.imshow(matrix_confocal_ph)
            ax3_1.set_title('Image acquisition photodiode')
            ax3_1.grid(False)
        
            ax3_2.plot(image_curve_gauss[0], image_curve_gauss[1]/max(image_curve_gauss[1]), 'ro',label = 'Fast axis')
            ax3_2.plot(image_fitx[0], image_fitx[1], 'r-', alpha=0.5)
            ax3_2.plot(image_curve_gauss[0], image_curve_gauss[2]/max(image_curve_gauss[2]), 'bo', label = 'Slow axis')
            ax3_2.plot(image_fity[0], image_fity[1], 'b-', alpha=0.5)
            ax3_2.set_ylim(0, 1.2)
            ax3_2.legend(loc = 'upper right')
        
            fig.set_tight_layout(True)
        
            plt.show()  
            
    print('histogram psf from photodiode 532 nm')
    
    plot_histogram(save_folder, wx_ph, wy_ph, 'all_Gauss_omega_PH')
    
#%%

    if confocal_ph_2DGauss:
    
        wx_ph = []
        wy_ph = []
        
        for j in range(L_confocal):
                    
                NP = list_of_files[j].split('.')[0]
                NP = NP.split('NPscan_')[1]
                
                print(NP)
                
                if NP in exceptions_NP:
                   print('salteo', NP)
                   continue
                
                file_confocal =  os.path.join(direction, list_of_files[j])
                matrix_confocal_ph = open_confocal_ph(file_confocal)
                
                w_x_ph = 0
                w_y_ph = 0
                
                try:
                    
                    w_x_ph, w_y_ph = fit_gauss2D(matrix_confocal_ph, rango = 2000)
                    
                    print('Gauss ajuste fotodiodo:', 'w fast', w_x_ph, 'w slow',  w_y_ph)
                
                except: pass
                wx_ph.append(w_x_ph)
                wy_ph.append(w_y_ph)
            
        print('histogram psf 2D Gauss from photodiode 532 nm')
        plot_histogram(save_folder, wx_ph, wy_ph, 'all_Gauss_omega_PH_2DGauss')