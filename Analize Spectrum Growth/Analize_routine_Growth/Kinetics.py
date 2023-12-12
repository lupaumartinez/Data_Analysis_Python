#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 18:51:45 2021

@author: luciana
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')

def post3_process_spectrum(common_path, mode, color, n_points, plot_all):
    
    list_of_folders = os.listdir(common_path)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(common_path,f))]
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Plot SPR Wavelength Stokes')
    
    min_point = []
    
    total_time = []
    
    time = np.zeros(n_points)
    mean = np.zeros(n_points)
    n = 0
    
    for i in range(L):
        
        NP = list_of_folders[i].split('_')[-1]

        path_folder = os.path.join(common_path, list_of_folders[i])
        list_of_files = os.listdir(path_folder)
    
        for file in list_of_files:
            
            if file.startswith('data_live%sfitting'%mode):
                
                name_file = os.path.join(path_folder, file)
                a = np.loadtxt(name_file, skiprows=1)
     
                time_londa = a[:, 0]
                londa_max_pl = a[:, 1]
                
                total_time.append(time_londa[-1])
                
                if plot_all:
                    plt.plot(time_londa, londa_max_pl, marker = '*', linestyle = '--', color = color) # label = 'NP_%s'%(NP))
                
                min_point.append(len(time_londa))
                
                time_londa = a[: n_points, 0]
                londa_max_pl = a[: n_points, 1]
                
                if len(time_londa) >=  n_points:
                
                    mean = mean + londa_max_pl
                    time = time + time_londa
                    
                    n = n + 1
                
    min_point = min(min_point)
    print('n points optim:', min_point)
                
    mean = mean/n
    time = time/n
                    
    return time, mean, total_time


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
   # x_gaussiana= np.linspace(mu-5*sigma, mu+5*sigma, num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
   # gaussiana= norm.pdf(x_gaussiana, mu, sigma)#*N*bin_size # calculo la gaussiana que corresponde al histograma
    
    txt = '%s + %s'%(mu, sigma)
    
    return mu, sigma, txt

    
if __name__ == '__main__':
    
   
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'

    daily_folder = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/growth'
    
    paths = ['20210809-180739_Growth_10x2_col11y12_580nm', '20210809-172719_Growth_10x2_col9y10_570nm', '20210809-165810_Growth_10x2_col7y8_560nm']
   
    target = ['580', '570', '560']
    color = ['r' ,'k', 'g']
    
    n_points = [22, 15, 9] 
    
    mode = '_' #_poly_' #'_poly_' #'_poly_'#  '_'  (es lorentz) 
    
    plot_all = False
    plot_mean = True
    plot_histogram_time = True
    
    plt.figure()
    
    save_folder = 'mean_kinetics'
    save_folder = os.path.join(base_folder, daily_folder, save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for i in range(len(paths)):
    
        common_path = os.path.join(base_folder, daily_folder, paths[i], 'processed_data_sustrate_bkg_True')
    
        time, mean, total_time = post3_process_spectrum(common_path, mode, color[i], n_points[i], plot_all)
        
        if plot_mean:
        
            plt.plot(time, mean, marker = 'o', linestyle = '--', color = color[i], label = '%s'%(target[i])) 
            
            print('tiempo final', i, time[-1], 'mean total time', np.mean(total_time))
            
            name = os.path.join(save_folder,'all%sSPR_mean_%s.txt'%(mode, target[i]))
            data = np.array([time, mean])
            header_txt = 'time, mean londa'
            np.savetxt(name, data, header = header_txt)
            
        if plot_histogram_time:
            
            plt.figure()
            n,bin_positions_x,p = plt.hist(total_time, bins=5, range=[min(total_time)-10, max(total_time)+10], density=True, rwidth=0.9, color = color[i])
            mean, sigma, x_text = data_hist(total_time, bin_positions_x)
            #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
            plt.text(max(total_time)-10, 0.02, x_text, fontsize = 'small')
            plt.xlabel('Time (s)')
        
            figure_name = os.path.join(save_folder, 'histogram%sSPR_mean_%s.png'%(mode, target[i]))
            plt.savefig(figure_name, dpi = 400)
            plt.close()
        
    plt.xlim(-2, 250)
    plt.ylim(530, 590)   
        
    plt.xlabel('Time (s)')
    plt.ylabel('$\u03BB_{max}$ Stokes PL (nm)')
    plt.legend()
    figure_name = os.path.join(save_folder, 'all%sSPR_mean.png'%mode) 
    plt.savefig(figure_name , dpi = 400)
    plt.show()
    

    