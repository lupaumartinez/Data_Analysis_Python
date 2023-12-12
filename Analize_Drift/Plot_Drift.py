#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:47:42 2019

@author: luciana
"""

import numpy as np
import matplotlib.pyplot as plt

import time
import os
import re

from skimage import io
from scipy.optimize import curve_fit
from scipy import ndimage

def load_txt(base_folder):
    
    list_of_all_files = os.listdir(base_folder)
    file =  [f for f in list_of_all_files if re.search('txt',f)]
    
    file = os.path.join(base_folder, file[0])
    
    drift_txt = np.loadtxt(file)
    
    return drift_txt
    

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

def center_mass(image, n_filter, rango_x, rango_y):
    
    
    new_image = (image- np.min(image) ) / (np.max(image) -  np.min(image))

    for i in range(len(new_image)):
        for j in range(len(new_image)):
            if new_image[i, j] <= n_filter:
                    new_image[i, j] = 0
                    com = ndimage.measurements.center_of_mass(new_image)
                    com = np.round(com, 2)

    Nx = image.shape[0]
    px_nm = rango_x/Nx
    
    Ny = image.shape[1]
    py_nm = rango_y/Ny
    
    com_nm = np.array(com[0])*px_nm - rango_x/2 + px_nm/2 , np.array(com[1])*py_nm  - rango_y/2 + py_nm/2
    com_nm = np.around(com_nm, 2)
    
    print('px (x0, y0) = ({}, {})'.format(*com))
    print('nm (x0, y0) = ({}, {})'.format(*com_nm))
    
    return com, com_nm

def center_gauss(image, initial_position, range_x, range_y):
    
    Nx = image.shape[0]
    Ny = image.shape[1]
    
    px_nm = range_x/Nx
    py_nm = range_y/Ny
    
    x = np.arange(-Nx/2 +1/2, Nx/2)
    y = np.arange(-Ny/2 +1/2, Ny/2) 
    
    	#x = np.arange(0, Nx)
    	#y = np.arange(0, Ny)
    [Mx, My] = np.meshgrid(x, y)
    
    dataG_2d = (image- np.min(image) ) / (np.max(image) -  np.min(image))
    dataG_ravel = dataG_2d.ravel()
    
    initial_sigma = [150/px_nm, 150/py_nm]
    
    #primero va lo de y, luego lo de x
    initial_guess_G = [1, initial_position[1] + y[0],  initial_position[0] + x[0], initial_sigma[1], initial_sigma[0], 0]
    bounds = ([0, y[0], x[0], 0, 0, 0], [1,-y[0], -x[0], 4*initial_sigma[1], 4*initial_sigma[0], 1])
    
    	#initial_guess_G = [1, initial_position[0],  initial_position[1], initial_sigma[0], initial_sigma[1], 0]
    	#bounds = ([0, 0, 0, 0, 0, 0], [1, Nx, Ny, 3*initial_sigma[0], 3*initial_sigma[0], 1])
    
    poptG, pcovG = curve_fit(gaussian2D, (Mx, My), dataG_ravel, p0= initial_guess_G, bounds = bounds)
    
    poptG = np.around(poptG, 2)
    #print('A = {}, (y0, x0) = ({}, {}), σ_y = {}, σ_x = {}, bkg = {}'.format(*poptG))
    
    cog =  poptG[2] - x[0], poptG[1] - y[0]
    
    cog_nm =  (np.array(cog[0]) + x[0])*px_nm, (np.array(cog[1]) + y[0])*py_nm
    cog_nm = np.around(cog_nm, 2)
        
    print('px (x0, y0) = ({}, {})'.format(*cog))
    print('nm (x0, y0) = ({}, {})'.format(*cog_nm))
    
    sigma_nm = poptG[4]*px_nm, poptG[3]*py_nm
    w_nm = 2*np.around(sigma_nm, 2)
    print('nm (w_x, w_y) = ({}, {})'.format(*w_nm))
    
    return cog, cog_nm, w_nm

    
def analize_drift(base_folder, rango_x, rango_y, n_filter, type_scan, total_time):
    
    drift_txt = load_txt(base_folder)
        
    save_folder = os.path.join(base_folder, 'figures_drift')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    folder = os.path.join(base_folder, type_scan)
    
    list_com_x = []
    list_cog_x = []
    list_w_x = []
        
    list_com_y = []
    list_cog_y = []
    list_w_y = []
    
    list_of_files = os.listdir(folder)
   # list_of_files.sort()
    list_of_files.sort(key= lambda x: float(''.join(x.split('minute_')[1].split('.tiff')[0])))

    print(list_of_files)
    L = len(list_of_files)
    
    for file in list_of_files:
        
        name_file = os.path.join(folder, file)
        image = io.imread(name_file)
        
       #center of mass
        
        t0 = time.time()
        com, com_nm = center_mass(image, n_filter, rango_x, rango_y)
        t1 = time.time()
        t_m = (t1-t0)*1e3 
        print('center of mass fit took {} ms'.format(np.around(t_m, 2)))
        
        	#center of gauss
        
        initial_pos = com
        
        t0 = time.time()
        cog, cog_nm, w_nm = center_gauss(image, initial_pos, rango_x, rango_y)
        t1 = time.time()
        t_G = (t1-t0)*1e3 
        
        print('gaussian fit took {} ms'.format(np.around(t_G, 2)))
    
        	#plotear
        
        plt.figure('gaussian data')
        plt.imshow(image)
        plt.scatter(*cog[::-1], color = 'b')
        plt.scatter(*com[::-1], color = 'r')
        plt.show()
        
        list_com_x.append(com_nm[0])
        list_cog_x.append(cog_nm[0])
        list_w_x.append(w_nm[0])
        
        list_com_y.append(com_nm[1])
        list_cog_y.append(cog_nm[1])
        list_w_y.append(w_nm[1])
        
    
    list_time = np.linspace(0, total_time, L)
    print(list_time)

    plt.figure()
   # plt.plot(list_time, list_com_x, 'ro--', label = 'center mass axe x PI')
   # plt.plot(list_time, list_com_y, 'bo--', label = 'center mass axe y PI')
    plt.plot(list_time, list_cog_x, 'm*--', label = 'center gauss axe x PI')
    plt.plot(list_time, list_cog_y, 'g*--', label = 'center gauss axe y PI')
    
   # plt.plot(drift_txt[:,0], 1000*(drift_txt[:,1]-drift_txt[0,1]), 'ko-')
   # plt.plot(drift_txt[:,0], 1000*(drift_txt[:,2]-drift_txt[0,2]), 'yo-')
    
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Drift (nm)')
    #plt.xlim(0, total_time + time_refresh)
    plt.ylim(-800, 800)
    figure_name = os.path.join(save_folder, 'drift_xy_center_%s'%type_scan)
    plt.savefig(figure_name, dpi = 400)
    plt.show()
    
    plt.figure()
    plt.plot(list_time, list_w_x, 'm*--', label = 'gauss axe x PI')
    plt.plot(list_time, list_w_y, 'g*--', label = 'gauss axe y PI')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('W (nm)')
    plt.ylim(250, 370)
    #plt.xlim(0, total_time + time_refresh)
    figure_name = os.path.join(save_folder, 'drift_xy_psf_%s'%type_scan)
    plt.savefig(figure_name, dpi = 400)
    plt.show()
    
    header_txt = 'Time (s), Center Mass X (nm), Center Mass Y (nm), Center Gauss X (nm), Center Gauss Y (nm), PSF Gauss X (nm), PSF Gauss Y (nm)'
    data = np.array([list_time, list_com_x, list_com_y, list_cog_x, list_cog_y, list_w_x, list_w_y]).T
    name = os.path.join(save_folder,'drift_xy_%s.txt'%type_scan)    
    np.savetxt(name, data, delimiter = ',', header = header_txt)
    
    return

def compare_drift(folder, type_scan, type_data):
    
    list_of_all_files = os.listdir(folder)
    
    if type_scan == 'Scan':
    
        list_of_files = [f for f in list_of_all_files if re.search('Scan',f) and not re.search('Scan_gone',f) and not re.search('Scan_back',f) and not re.search('png',f)]
        
    elif type_scan == 'Scan_back':
        
        list_of_files = [f for f in list_of_all_files if re.search('Scan_back',f) and not re.search('png',f)]
        
    elif type_scan == 'Scan_gone':
        
        list_of_files = [f for f in list_of_all_files if re.search('Scan_gone',f) and not re.search('png',f)]
        
    list_of_files.sort()
    
    plt.figure()
    plt.title('drift_xy_compare_%s_%s'%(type_scan, type_data))
    
    print(list_of_files)
    
    for file in list_of_files:
        
        name = file.split('_')[-1].split('.txt')[0] + '°C'
        
        direction = os.path.join(folder, file)
        drift = np.loadtxt(direction, delimiter = ',', comments='#')
                                  
        list_time = drift[:, 0]
        list_com_x = drift[:, 1]
        list_com_y = drift[:, 2]
        list_cog_x = drift[:, 3]
        list_cog_y = drift[:, 4]
        list_w_x = drift[:, 5]
        list_w_y = drift[:, 6]
        
        #
        if type_data == 'X':
            
            #plt.plot(list_time, list_com_x, 'o', label = name)
            plt.plot(list_time, list_cog_x, '*', label = name)
            
            plt.ylim(-1300, 1300)    
            plt.ylabel('Drift (nm)')
            
        elif type_data == 'Y':
        
           # plt.plot(list_time, list_com_y, 'o', label = name)
            plt.plot(list_time, list_cog_y, '*', label = name)
            
            plt.ylim(-1300, 1300)
            plt.ylabel('Drift (nm)')
        
        elif type_data == 'PSF_X':
        
            plt.plot(list_time, list_w_x, '*', label = name)
            
            plt.ylim(270, 370)
            plt.ylabel('W (nm)')
        
        elif type_data == 'PSF_Y':
        
            plt.plot(list_time, list_w_y, '*', label = name)
            
            plt.ylim(270, 370)
            plt.ylabel('W (nm)')
        
    plt.legend()
    plt.xlabel('Time (s)')
    figure_name = os.path.join(folder, 'drift_xy_compare_%s_%s'%(type_scan, type_data))
    plt.savefig(figure_name, dpi = 400)
    plt.show()
    
    return
    
if __name__ == '__main__':
    
    #prefix_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2019/Mediciones_PySpectrum/'
    #local_folder  = '2019-12-03 (Confocal Spectrum, veo psf)/20191204-125150_Luminescence_11x1'
    
    prefix_folder = '/home/luciana/Printing/'
    
    local_folder = '2021-01-04 (Drift)'
    
    base_folder = os.path.join(prefix_folder, local_folder)
    
    rango_x = 2000
    rango_y = 2000
    n_filter = 0.2
    #time_refresh = 342 #s
    total_time = 60*20  #s
    type_scan = 'Scan_gone'
    
    one_drift_analize = True
    if one_drift_analize:
        analize_drift(base_folder, rango_x, rango_y, n_filter, type_scan, total_time)
    
    ##

    all_drift_analize = False
    type_scan = 'Scan_back'#'Scan_gone', 'Scan_back'
    type_data = 'PSF_Y' #'Y', 'PSF_X', 'PSF_Y'
    
    if all_drift_analize:
        local_folder  = '2019-12-03 (Confocal Spectrum, veo psf)/drift_vs_temperature'
        folder = os.path.join(prefix_folder, local_folder)
        compare_drift(folder, type_scan, type_data)
            
        
        
