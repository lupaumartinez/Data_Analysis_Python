#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:02:07 2019

@author: luciana
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:03:37 2019

@author: Luciana
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def compare_spectrum(common_path_sca_px, common_path_sca_py, common_path_sca_unpol, save_folder, name_col):
    
    common_path_fit_px = os.path.join(common_path_sca_px,'fit_polynomial_normalized_%s'%(name_col))
    common_path_fit_py = os.path.join(common_path_sca_py,'fit_polynomial_normalized_%s'%(name_col))
    common_path_fit_unpol = os.path.join(common_path_sca_unpol,'fit_polynomial_normalized_%s'%(name_col))
    
    common_path_sca_px = os.path.join(common_path_sca_px,'normalized_%s'%(name_col))
    common_path_sca_py = os.path.join(common_path_sca_py,'normalized_%s'%(name_col))
    common_path_sca_unpol = os.path.join(common_path_sca_unpol,'normalized_%s'%(name_col))
            
    save_folder_col = os.path.join(save_folder, name_col)
    if not os.path.exists(save_folder_col):
        os.makedirs(save_folder_col)
    
    list_of_folders_scattering_px = os.listdir(common_path_sca_px)
    list_of_folders_scattering_px = [f for f in list_of_folders_scattering_px if re.search('NP',f)]
    list_of_folders_scattering_px.sort()
    M_px = len(list_of_folders_scattering_px)  
    
    print('Files from Scattering PX', M_px, list_of_folders_scattering_px)
    
    list_of_folders_scattering_py = os.listdir(common_path_sca_py)
    list_of_folders_scattering_py = [f for f in list_of_folders_scattering_py if re.search('NP',f)]
    list_of_folders_scattering_py.sort()
    M_py = len(list_of_folders_scattering_py)  
    
    print('Files from Scattering PY', M_py, list_of_folders_scattering_py)
    
        
    list_of_folders_scattering = os.listdir(common_path_sca_unpol)
    list_of_folders_scattering = [f for f in list_of_folders_scattering if re.search('NP',f)]
    list_of_folders_scattering.sort()
    M = len(list_of_folders_scattering)  
    
    print('Files from Scattering Unpol', M, list_of_folders_scattering)
    
 #   list_of_folders_fit_px = os.listdir(common_path_fit_px)
 #   list_of_folders_fit_px = [f for f in list_of_folders_fit_px if re.search('NP',f)]
 #   list_of_folders_fit_px.sort()
    
 #   list_of_folders_fit_py = os.listdir(common_path_fit_py)
 #   list_of_folders_fit_py = [f for f in list_of_folders_fit_py if re.search('NP',f)]
 #   list_of_folders_fit_py.sort()
    
    if not M_py == M_px or not M_py == M or not M == M_px: 
        
        print('The files do not have the same NP. Check files')
        
   
    L = np.min([M_px, M_py, M])
        
    
    for i in range(L):
        
   
        name_file_sca_px = os.path.join(common_path_sca_px,  list_of_folders_scattering_px[i])
        name_file_sca_py = os.path.join(common_path_sca_py,  list_of_folders_scattering_px[i])
        name_file_sca = os.path.join(common_path_sca_unpol,  list_of_folders_scattering_px[i])
        
        NP = list_of_folders_scattering_px[i].split('NP_')[-1]
        NP = NP.split('.')[0]
        
        print('Ploteo NP_%s'%(NP))
        
        b = np.loadtxt(name_file_sca_px, skiprows=1)
        wavelength_sca_px = b[:, 0]
        scattering_px = b[:, 1]
        
        c = np.loadtxt(name_file_sca_py, skiprows=1)
        wavelength_sca_py = c[:, 0]
        scattering_py = c[:, 1]
        
        a = np.loadtxt(name_file_sca, skiprows=1)
        wavelength_sca = a[:, 0]
        scattering = a[:, 1]
     
        name_file_sca_px = os.path.join(common_path_fit_px,  list_of_folders_scattering_px[i])
        name_file_sca_py = os.path.join(common_path_fit_py,  list_of_folders_scattering_px[i])
        name_file_sca = os.path.join(common_path_fit_unpol,  list_of_folders_scattering_px[i])
        
        b = np.loadtxt(name_file_sca_px, skiprows=1)
        wavelength_fit_px = b[:, 0]
        fit_px = b[:, 1]
        
        c = np.loadtxt(name_file_sca_py, skiprows=1)
        wavelength_fit_py = c[:, 0]
        fit_py = c[:, 1]
        
        a = np.loadtxt(name_file_sca, skiprows=1)
        wavelength_fit = a[:, 0]
        fit = a[:, 1]
        
        plt.figure()
        
        plt.title('NP_%s'%(NP))
        
        plt.plot(wavelength_sca_px, scattering_px, label = 'Scattering PX')
        plt.plot(wavelength_sca_py, scattering_py, label = 'Scattering PY')
        plt.plot(wavelength_sca, scattering, label = 'Scattering Unpol')
        
        plt.plot(wavelength_fit_px, fit_px, '--' )
        plt.plot(wavelength_fit_py, fit_py, '--')
        plt.plot(wavelength_fit, fit, '--')   
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right') 
        
        #plt.xlim(500, 1000)
       # plt.ylim(0.5, 1.05)
        
        figure_name = os.path.join(save_folder_col, 'compare_Scattering_Polarization_%s_NP_%s.png'%(name_col, NP)) 
        
        #plt.show()
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
def compare_lspr(common_path_sca_px, common_path_sca_py, common_path_sca_unpol, save_folder, name_col):
    
    file_px = os.path.join(common_path_sca_px,'fig_normalized_%s'%(name_col), 'max_wavelength.txt')
    file_py = os.path.join(common_path_sca_py,'fig_normalized_%s'%(name_col), 'max_wavelength.txt')
    file_unpol = os.path.join(common_path_sca_unpol,'fig_normalized_%s'%(name_col), 'max_wavelength.txt')
        
    data_px = np.loadtxt(file_px, skiprows = 1)
    data_py = np.loadtxt(file_py, skiprows = 1)
    data_unpol = np.loadtxt(file_unpol, skiprows = 1)
    
    NP_list_px = np.array(data_px[:, 0])
    NP_list_py = np.array(data_py[:, 0])
    NP_list_unpol = np.array(data_unpol[:, 0])
    
    max_wavelength_sca_px = data_px[:, 1]
    max_wavelength_sca_py = data_py[:, 1]
    max_wavelength_sca_unpol = data_unpol[:, 1]
    
    fit_poly_wavelength_sca_px = data_px[:, 2]
    fit_poly_wavelength_sca_py = data_py[:, 2]
    fit_poly_wavelength_sca_unpol = data_unpol[:, 2]
    
 #   fit_wavelength_sca_px = data_px[:, 3]
 #   fit_wavelength_sca_py = data_py[:, 3]
    
    plt.figure()
    plt.title('%s'%(name_col))
    plt.plot(NP_list_px, max_wavelength_sca_px, 'o', label = 'max PX')
    plt.plot(NP_list_py, max_wavelength_sca_py, 'o', label = 'max PY')
    plt.plot(NP_list_unpol, max_wavelength_sca_unpol, 'o', label = 'max Unpol')
   # plt.plot(NP_list_px, fit_wavelength_sca_px, '*', label = 'fit PX')
   # plt.plot(NP_list_py, fit_wavelength_sca_py, '*', label = 'fit PY')
    plt.plot(NP_list_px, fit_poly_wavelength_sca_px, '*', label = 'max polynomial PX')
    plt.plot(NP_list_py, fit_poly_wavelength_sca_py, '*', label = 'max polynomial PY')
    plt.plot(NP_list_unpol, fit_poly_wavelength_sca_unpol, '*', label = 'max polynomial Unpol')
    plt.xlabel('NP')
    plt.ylabel('Max wavelength (nm)')
    plt.ylim(530, 650)
    plt.legend()
    figure_name = os.path.join(save_folder, 'max_wavelength_Scattering_%s.png'%(name_col)) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    name = os.path.join(save_folder,  'max_wavelength_Scattering_%s.txt'%(name_col))
    data = np.array([NP_list_px, max_wavelength_sca_px, fit_poly_wavelength_sca_px, NP_list_py, max_wavelength_sca_py, fit_poly_wavelength_sca_py, NP_list_unpol, max_wavelength_sca_unpol, fit_poly_wavelength_sca_unpol]).T
    header_txt = 'NP, max wavelength Sca PX (nm), Max polynomial PX (nm), NP, max wavelength PY (nm), Max polynomial PY (nm), NP, max wavelength Unpol (nm), Max polynomial Unpol (nm)'
    np.savetxt(name, data, header = header_txt)
                    
    return


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


def plot_more_analysis_all(save_folder, name_col):
    
    name_file = os.path.join(save_folder, 'max_wavelength_Scattering_%s.txt'%name_col)
    a = np.loadtxt(name_file, skiprows=1)
    
    NP_list_px = a[:, 0]
    max_wavelength_sca_PX = a[:, 1]
    fit_sca_PX = a[:, 2]
    
    NP_list_py = a[:, 3]
    max_wavelength_sca_PY = a[:, 4]
    fit_sca_PY = a[:, 5]
    
    NP_list_unpol = a[:, 6]
    max_wavelength_sca_unpol = a[:, 7]
    fit_sca_unpol = a[:, 8]
    
    means, sigmas = plot_histogram_all(save_folder, max_wavelength_sca_PX, fit_sca_PX, max_wavelength_sca_PY, fit_sca_PY, max_wavelength_sca_unpol, fit_sca_unpol, 'histogram_wavelength_%s'%(name_col))
    
    return
    
    
def plot_histogram_all(save_folder, max_wavelength_sca_PX, fit_sca_PX, max_wavelength_sca_PY, fit_sca_PY, max_wavelength_sca_unpol, fit_sca_unpol, name):
    
    print('Plot Histogram ')   
    
    lim_range = [520, 620]
    bins = 10
    
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15,5))

    n,bin_positions_x,p = ax1.hist(max_wavelength_sca_PX, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C0')
    mu_sca_PX, sigma_sca_PX, x_text = data_hist(max_wavelength_sca_PX, bin_positions_x)
        
    n,bin_positions_x,p = ax1.hist(fit_sca_PX, bins=bins, range=lim_range, density=True, rwidth=0.9,alpha = 0.7, color ='C3')
    mu_fit_sca_PX, sigma_fit_sca_PX, fit_x_text = data_hist(fit_sca_PX, bin_positions_x)
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax1.text(530, 0.12, x_text, color= 'C0', fontsize = 'medium')
    ax1.text(530, 0.10, fit_x_text, color= 'C3', fontsize = 'medium')
        
    ax1.set_xlabel('Sca PX')
    ax1.set_ylim(0, 0.12)
    ax1.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax2.hist(max_wavelength_sca_PY, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C1')
    mu_sca_PY, sigma_sca_PY, x_text = data_hist(max_wavelength_sca_PY, bin_positions_x)
    
    n,bin_positions_x,p = ax2.hist(fit_sca_PY, bins=bins, range=lim_range, density=True, rwidth=0.9, alpha = 0.7, color ='C4')
    mu_fit_sca_PY, sigma_fit_sca_PY, fit_x_text = data_hist(fit_sca_PY, bin_positions_x)
    
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax2.text(530, 0.12, x_text, color= 'C1', fontsize = 'medium')
    ax2.text(530, 0.10, fit_x_text, color= 'C4', fontsize = 'medium')
    
    ax2.set_xlabel('Sca PY')
    ax2.set_ylim(0, 0.14)
    ax2.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax3.hist(max_wavelength_sca_unpol, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C2')
    mu_sca, sigma_sca, x_text = data_hist(max_wavelength_sca_unpol, bin_positions_x)
    
    n,bin_positions_x,p = ax3.hist(fit_sca_unpol, bins=bins, range=lim_range, density=True, rwidth=0.9, alpha = 0.7, color ='C5')
    mu_fit_sca, sigma_fit_sca, fit_x_text = data_hist(fit_sca_unpol, bin_positions_x)
    
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax3.text(530, 0.12, x_text, color= 'C2', fontsize = 'medium')
    ax3.text(530, 0.10, fit_x_text, color= 'C5', fontsize = 'medium')
    
    ax3.set_xlabel('Sca Unpol')
    ax3.set_ylim(0, 0.14)
    ax3.set_xlim(lim_range[0],lim_range[1])
    
    f.set_tight_layout(True)

    figure_name = os.path.join(save_folder, name) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    name = os.path.join(save_folder, 'data_%s'%(name))
    data = np.array([mu_sca_PX, sigma_sca_PX, mu_fit_sca_PX, sigma_fit_sca_PX, 
                     mu_sca_PY, sigma_sca_PY, mu_fit_sca_PY, sigma_fit_sca_PY,
                     mu_sca, sigma_sca, mu_fit_sca, sigma_fit_sca])
    header_txt = 'mean Sca PX, std Sca PX, mean fit Sca PX, std fit Sca PX, mean Sca PY, std Sca PY, mean fit Sca PY, std fit Sca PY, mean Sca Unpol, std Sca Unpol, mean fit Sca Unpol, std fit Sca Unpol'
    np.savetxt(name, data, header = header_txt)
    
    means = mu_sca_PX, mu_fit_sca_PX, mu_sca_PY, mu_fit_sca_PY, mu_sca, mu_fit_sca
    sigmas = sigma_sca_PX, sigma_fit_sca_PX, sigma_sca_PY, sigma_fit_sca_PY, sigma_sca, sigma_fit_sca
    
    return means, sigmas

def plot_difference(save_folder, name_col):

    name_file = os.path.join(save_folder, 'max_wavelength_Scattering_%s.txt'%name_col)
    a = np.loadtxt(name_file, skiprows=1)
    
    NP_list_px = a[:, 0]
    max_wavelength_sca_PX = a[:, 1]
    fit_sca_PX = a[:, 2]
    
    NP_list_py = a[:, 3]
    max_wavelength_sca_PY = a[:, 4]
    fit_sca_PY = a[:, 5]
    
    print('Plot difference', name_col)

    plt.figure()
    plt.title('%s'%(name_col))
    plt.plot(NP_list_px, fit_sca_PX - max_wavelength_sca_PX, 'o', label = 'PX')
    plt.plot(NP_list_px, fit_sca_PY - max_wavelength_sca_PY, 'o', label = 'PY')
    plt.xlabel('NP')
    plt.ylabel('Difference max polynomial - max (nm)')
    plt.ylim(-15, 15)
    plt.legend()
    figure_name = os.path.join(save_folder, 'difference_wavelength_Scattering_%s.png'%(name_col)) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    return

def plot_difference_polarization(save_folder, name_col):

    name_file = os.path.join(save_folder, 'max_wavelength_Scattering_%s.txt'%name_col)
    a = np.loadtxt(name_file, skiprows=1)
    
    NP_list_px = a[:, 0]
    max_wavelength_sca_PX = a[:, 1]
    fit_sca_PX = a[:, 2]
    
    NP_list_py = a[:, 3]
    max_wavelength_sca_PY = a[:, 4]
    fit_sca_PY = a[:, 5]
    
    print('Plot difference', name_col)

    plt.figure()
    plt.title('%s'%(name_col))
    plt.plot(NP_list_px, fit_sca_PX - fit_sca_PY, 'k*', label = 'max polynomial')
    plt.plot(NP_list_px, max_wavelength_sca_PX - max_wavelength_sca_PY, 'ro', label = 'max')
    plt.xlabel('NP')
    plt.ylabel('Difference PX - PY (nm)')
    plt.ylim(-15, 15)
    plt.legend()
    figure_name = os.path.join(save_folder, 'difference_polarization_%s.png'%(name_col)) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    plot_histogram_all_difference(save_folder, max_wavelength_sca_PX, fit_sca_PX, max_wavelength_sca_PY, fit_sca_PY, 'histogram_difference_PX-PY_%s'%(name_col))
    
    return

def plot_histogram_all_difference(save_folder, max_wavelength_sca_PX, fit_sca_PX, max_wavelength_sca_PY, fit_sca_PY, name):
    
    print('Plot Histogram Difference Polarization')   
    
    lim_range = [-15, 15]
    bins = 10
    
    f, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5,5))
    
    dif_max = max_wavelength_sca_PX - max_wavelength_sca_PY
    dif_max_poly = fit_sca_PX - fit_sca_PY

    n,bin_positions_x,p = ax1.hist(dif_max, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C4')
    mu, sigma, x_text = data_hist(dif_max, bin_positions_x)
        
    n,bin_positions_x,p = ax1.hist(dif_max_poly , bins=bins, range=lim_range, density=True, rwidth=0.9,alpha = 0.7, color ='C5')
    mu_poly, sigma_poly, fit_x_text = data_hist(dif_max_poly , bin_positions_x)
    
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax1.text(-10, 0.10, x_text, color= 'C4', fontsize = 'medium')
    ax1.text(-10, 0.08, fit_x_text, color= 'C5', fontsize = 'medium')
        
    ax1.set_xlabel('Difference PX - PY (nm)')
    ax1.set_ylim(0, 0.12)
    ax1.set_xlim(lim_range[0],lim_range[1])

    figure_name = os.path.join(save_folder, name) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    name = os.path.join(save_folder, 'data_%s'%(name))
    data = np.array([mu, sigma, mu_poly, sigma_poly])
    header_txt = 'mean PX - PY, std PX - PY, mean poly PX - PY, std poly PX - PY'
    np.savetxt(name, data, header = header_txt)
    
    means = mu, mu_poly
    sigmas = sigma, sigma_poly
    
    return means, sigmas

#%%
         
if __name__ == '__main__':
    
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
    
   # daily_folder = '2021-03-26 (Growth)/Scattering_growth/'
   # daily_folder = '2021-06 (Growth)/2021-06-23 (growth)/Scattering_growth/'
   
    daily_folder = '2021-08 (Growth PL circular)/2021-08-18 (post growth, PL circular)/'
   
    save_folder = os.path.join(base_folder, daily_folder, 'fig_compare_Scattering_Polarization_three')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    list_number_col = [12] #1,2,3,4,5, 6,7,8,9,10,11, 12
        
    for number_col in list_number_col:
    
        name_col = 'Col_%03d'%(number_col)
        
        print('Analize ', name_col)
        
        daily_folder_sca_px = 'Scattering_pol_min_laser/photos/'
        daily_folder_sca_py = 'Scattering_pol_max_laser/photos/'
        daily_folder_sca_unpol = 'Scattering_unpol/photos/'
      
        common_path_sca_px = os.path.join(base_folder, daily_folder, daily_folder_sca_px, name_col)
        common_path_sca_py = os.path.join(base_folder, daily_folder, daily_folder_sca_py, name_col)
        common_path_sca_unpol = os.path.join(base_folder, daily_folder, daily_folder_sca_unpol, name_col)

        compare_spectrum(common_path_sca_px, common_path_sca_py, common_path_sca_unpol, save_folder, name_col)
        compare_lspr(common_path_sca_px, common_path_sca_py, common_path_sca_unpol, save_folder, name_col)
        
#%%       
    folder = os.path.join(base_folder, daily_folder, 'fig_compare_Scattering_Polarization_three', 'buenos_casos')
        
    list_number_col = 1,3,4,5,7,8,9,10,11, 12
        
    for number_col in list_number_col:

        name_col = 'Col_%03d'%(number_col)
        
        print('Analize histogram ', name_col)
        
        plot_more_analysis_all(folder, name_col)
        plot_difference_polarization(folder, name_col)
        
#%%       
    folder = os.path.join(base_folder, daily_folder, 'fig_compare_Scattering_Polarization_three', 'buenos_casos_2')
        
    list_number_col = 1,3,4,5,7,9,11
        
    for number_col in list_number_col:

        name_col = 'Col_%03d'%(number_col)
        
        print('Analize histogram ', name_col)
        
        plot_more_analysis_all(folder, name_col)
        plot_difference_polarization(folder, name_col)
        