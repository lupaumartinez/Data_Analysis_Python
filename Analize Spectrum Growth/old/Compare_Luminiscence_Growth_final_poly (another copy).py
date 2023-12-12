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

def compare_final_spectrum(common_path_growth, common_path_PL, save_folder, name_col, number_col, number_NP):
    
    common_path_PL = os.path.join(common_path_PL, name_col)
    
    common_path_PL_growth = os.path.join(common_path_growth, 'all_data_final_Stokes')
    
    list_of_folders = os.listdir(common_path_PL_growth)
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Files from final growth PL', L, list_of_folders)
    
    common_path_PL = os.path.join(common_path_PL, 'luminescence_steps')
    
    list_of_folders_PL= os.listdir(common_path_PL)
    list_of_folders_PL = [f for f in list_of_folders_PL if re.search('NP',f)]
    list_of_folders_PL.sort()
    
    M = len(list_of_folders_PL)  
    
    print('Files from PL steps', M, list_of_folders_PL)
    
    if L == M:
    
        print('Plot final grwoth PL, compare with PL steps')
        
    else:
        
        print('The files do not have the same NP. Check files')
        
    NP_growth = []
    lspr_growth = []
    
    NP_PL_list = []
    lspr_PL = []
    
    first_wave = 545  #538,5 #545
    end_wave = 690  #641.5  #641.5 #620
    
    #para PL growth
    name_file = os.path.join(common_path_PL_growth,  list_of_folders[0])
    a = np.loadtxt(name_file, skiprows=1)
    wavelength = a[:, 0]
    
    desired_range = np.where((wavelength >= first_wave) & (wavelength <=end_wave))    
    desired_wave = wavelength[desired_range]
    x = np.linspace(desired_wave[0], desired_wave[-1], 1000)
    npol = 5
    
    #para PL steps
    name_file = os.path.join(common_path_PL,  list_of_folders_PL[0])
    b = np.loadtxt(name_file, skiprows=1)
    wavelength_PL = b[:, 0]
    
    roi_stokes = np.where((wavelength_PL >= wavelength[0]) & (wavelength_PL <= wavelength[-1]))
    wavelength_PL_stokes = wavelength_PL[roi_stokes]
    
    desired_range_PL = np.where((wavelength_PL_stokes >= first_wave) & (wavelength_PL_stokes <=end_wave))    
    desired_wave_PL = wavelength_PL_stokes[desired_range_PL] 
    x_PL = np.linspace(desired_wave_PL[0], desired_wave_PL[-1], 1000)
    npol_PL = 5
                
    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        
        NP = NP.split('.')[0]
        
      #  NP_PL = list_of_folders_PL[i].split('NP_')[-1]
     #   NP_PL = NP_PL.split('.')[0]
        
        NP_PL = (number_col- 1)*number_NP + int(NP)
        
        NP_PL = '%03d'%NP_PL
        
        print('Ploteo: from final PL grwoth NP_%s, from final PL steps  NP_%s'%(NP, NP_PL))
         
        ## PL of Growth in live
        
        name_file = os.path.join(common_path_PL_growth,  list_of_folders[i])
        
        a = np.loadtxt(name_file, skiprows=1)
    #    wavelength = a[:, 0]
        final_luminiscence = a[:, 1]
        desired_PL_final_growth = final_luminiscence[desired_range]

        p = np.polyfit(desired_wave, desired_PL_final_growth, npol)
        poly = np.polyval(p, x)
        max_wave_poly = round(x[np.argmax(poly)],3)
        
        londa_spr_growth = np.round(max_wave_poly,2)
        
        norm_poly = (poly - min(poly))/(max(poly) - min(poly))
        
        PL_final_growth = (desired_PL_final_growth - min(poly))/(max(poly) - min(poly))
        
        ## PL of steps
    
        name_file_PL = os.path.join(common_path_PL,  'Luminescence_Steps_Spectrum_%s_NP_%s.txt'%(name_col, NP_PL))
        
        b = np.loadtxt(name_file_PL, skiprows=1)
     #   wavelength_PL= b[:, 0]
       # luminiscence_NP = b[:, 1]
        luminiscence = b[:, 2]  #con sustrato

        luminiscence_stokes = luminiscence[roi_stokes]
        desired_PL_steps_stokes = luminiscence_stokes[desired_range_PL]
        
        p_PL = np.polyfit(desired_wave_PL, desired_PL_steps_stokes, npol_PL)
        poly_PL = np.polyval(p_PL, x_PL)
        max_wave_poly_PL = round(x[np.argmax(poly_PL)],3)
    
        londa_spr_PL = np.round(max_wave_poly_PL,2)
        
        norm_poly_PL = (poly_PL - min(poly_PL))/(max(poly_PL) - min(poly_PL))
        PL_steps_stokes = (luminiscence_stokes- min(poly_PL))/(max(poly_PL) - min(poly_PL))
        
        #PLoteo Normalziado
        
        plt.figure()
        
        plt.title('NP_%s'%(NP))
        plt.plot(desired_wave, PL_final_growth , label = 'growth final PL')
        plt.plot(x, norm_poly , 'r--', label = 'poly growth final PL')
        
     #   plt.plot(wavelength_PL, PL_steps, 'k', label = 'PL steps')
        plt.plot(desired_wave_PL, PL_steps_stokes, label = 'PL steps stokes')
        plt.plot(x_PL, norm_poly_PL, 'g--', label = 'poly PL steps stokes')
        
        #label = 'PL steps', )
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Intensity')
        plt.legend(loc='upper right') 
        
        #plt.xlim(500, 1000)
        plt.ylim(0.5, 1.05)
        
        figure_name = os.path.join(save_folder, 'norm_compare_final_PL_%s_NP_%s.png'%(name_col, NP_PL)) 
        
        #plt.show()
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        #data save
        NP_growth.append(int(NP))
        lspr_growth.append(londa_spr_growth)
        
        NP_PL_list.append(int(NP_PL))
        lspr_PL.append(londa_spr_PL)
        
        # Crude
        
        plt.figure()
        
        plt.title('NP_%s'%(NP))
        plt.plot(wavelength, final_luminiscence , label = 'growth final PL')
        plt.plot(x, poly , 'r--', label = 'poly growth final PL')
        
     #   plt.plot(wavelength_PL, PL_steps, 'k', label = 'PL steps')
        plt.plot(wavelength_PL, luminiscence, label = 'PL steps stokes')
        plt.plot(x_PL, poly_PL, 'g--', label = 'poly PL steps stokes')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right') 
        
        figure_name = os.path.join(save_folder, 'compare_final_PL_%s_NP_%s.png'%(name_col, NP_PL)) 
        
        #plt.show()
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        
        # PLoteo Difference
        
        plt.figure()
        
        plt.title('NP_%s'%(NP))
        plt.plot(x_PL, norm_poly_PL - norm_poly, 'k-')
        
        #label = 'PL steps', )
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Difference PL - PL growth live')
       # plt.legend(loc='upper right') 
        
        plt.ylim(-1, 1)
        
        figure_name = os.path.join(save_folder, 'difference_final_PL_%s_NP_%s.png'%(name_col, NP_PL)) 
        
        #plt.show()
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
                    
    return NP_growth, lspr_growth, NP_PL_list, lspr_PL

def compare_lspr(NP_growth, lspr_growth, NP_PL_list, lspr_PL, common_path_growth, common_path_PL, save_folder, name_col):
    
    name_file = os.path.join(common_path_growth,  'all_data_growth', 'all_data_poly_growth.txt')
    b = np.loadtxt(name_file, skiprows=1)
    NP = b[:, 0]
    total_time = b[:, 3]
    final_lspr_growth = b[:, 4]
    
    #chequear que sean las mismas NP que las de growth live
    name_file_2 = os.path.join(common_path_PL, 'fig_fit_all_data', 'fit_wavelength_lspr_Col_%03d.txt'%(number_col))
    b2 = np.loadtxt(name_file_2, skiprows=1)
    
    NP_PL = b2[:, 0]
    lspr_mode1 = b2[:, 1]
    
    plt.figure()
    plt.plot(NP, final_lspr_growth, 'o', label = 'lspr growth live poly')
    plt.plot(NP_growth, lspr_growth, 'o', label = 'lspr growth analysis poly')
    plt.plot(NP_growth, lspr_PL, 'o', label = 'lspr PL steps analysis poly')
    plt.plot(NP_growth, lspr_mode1, 'o', label = 'lspr PL steps analysis two modes')
    
    plt.xlabel('NP')
    plt.ylabel('Final LSPR')
    plt.legend(loc='lower right') 
    
    #plt.xlim(500, 1000)
    plt.ylim(540, 600)
    
    figure_name = os.path.join(save_folder, 'final_lspr_PL_%s.png'%(name_col)) 
    
    #plt.show()
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
   # plot_time(total_time_growth, final_lspr_growth, lspr_PL, save_folder, name_col)
   
    name = os.path.join(save_folder,  'lspr_fit_PL_%s.txt'%(name_col))
    data = np.array([NP, final_lspr_growth, total_time , lspr_growth, lspr_PL, lspr_mode1]).T
    header_txt = 'NP, lspr growth live poly (nm), Total time (s), lspr growth analysis poly (nm), lspr PL steps analysis poly (nm), lspr PL steps analysis two modes (nm)'
    np.savetxt(name, data, header = header_txt)
    
    return total_time, final_lspr_growth, lspr_PL

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
   # x_gaussiana=np.linspace(mu-5*sigma, mu+5*sigma, num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
   # gaussiana= norm.pdf(x_gaussiana, mu, sigma)#*N*bin_size # calculo la gaussiana que corresponde al histograma
    
    txt = '%s + %s'%(mu, sigma)
    
    return mu, sigma, txt


def plot_more_analysis_all(save_folder, name_col):
    
    name_file = os.path.join(save_folder, 'lspr_fit_PL_%s.txt'%name_col)
    a = np.loadtxt(name_file, skiprows=1)
    
    NP = a[:, 0]
    
    final_lspr_growth = a[:, 1]
    total_time = a[:, 2]
    lspr_growth = a[:, 3]
    
    lspr_PL = a[:, 4]
    lspr_PL_mode1 = a[:, 5]

    means, sigmas = plot_histogram_all(save_folder, final_lspr_growth, lspr_growth, lspr_PL, lspr_PL_mode1, 'histogram_lspr_fit_PL_%s'%(name_col))
    
    return means, sigmas
    
    
def plot_histogram_all(save_folder, final_lspr_growth, lspr_growth, lspr_PL, lspr_mode1, name):
    
    print('Plot Histogram ')   
    
    lim_range = [520, 600]
    bins = 20
    
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,5))

    n,bin_positions_x,p = ax1.hist(final_lspr_growth, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C0')
    mu_final_lspr_growth, sigma_final_lspr_growth, x_text = data_hist(final_lspr_growth, bin_positions_x)
        
    n,bin_positions_x,p = ax1.hist(lspr_growth, bins=bins, range=lim_range, density=True, rwidth=0.9,alpha = 0.7, color ='C1')
    mu_lspr_growth, sigma_lspr_growth, lspr_growth_x_text = data_hist(lspr_growth, bin_positions_x)
    
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax1.text(530, 0.25, x_text, color= 'C0', fontsize = 'medium')
    ax1.text(530, 0.23, lspr_growth_x_text, color= 'C1', fontsize = 'medium')
        
    ax1.set_xlabel('LSPR PL final growth live')
    ax1.set_ylim(0, 0.12)
    ax1.set_xlim(lim_range[0],lim_range[1])
    
    n,bin_positions_x,p = ax2.hist(lspr_PL, bins=bins, range=lim_range, density=True, rwidth=0.9, color ='C2')
    mu_lspr_PL, sigma_lspr_PL, x_text = data_hist(lspr_PL, bin_positions_x)
    
    n,bin_positions_x,p = ax2.hist(lspr_mode1, bins=bins, range=lim_range, density=True, rwidth=0.9, alpha = 0.7, color ='C3')
    mu_lspr_mode1, sigma_lspr_mode1, lspr_mode1_x_text = data_hist(lspr_mode1, bin_positions_x)
    
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax2.text(530, 0.25, x_text, color= 'C2', fontsize = 'medium')
    ax2.text(530, 0.23, lspr_mode1_x_text, color= 'C3', fontsize = 'medium')
    
    ax2.set_xlabel('LSPR PL steps')
    ax2.set_ylim(0, 0.30)
    ax2.set_xlim(lim_range[0],lim_range[1])
    
    f.set_tight_layout(True)

    figure_name = os.path.join(save_folder, name) 
    plt.savefig(figure_name, dpi = 400)
    plt.close()
    
    name = os.path.join(save_folder, 'data_%s'%(name))
    data = np.array([mu_final_lspr_growth,sigma_final_lspr_growth, mu_lspr_growth, sigma_lspr_growth,  mu_lspr_PL,  sigma_lspr_PL, mu_lspr_mode1, sigma_lspr_mode1])
    header_txt = 'mean PL live, std PL live, mean PL live poly-analysis, std PL live poly-analysis,  mean PL steps poly-analysis, std PL steps poly-anaylisis, mean PL steps two modes, std PL steps two modes'
    np.savetxt(name, data, header = header_txt)
    
    means = mu_final_lspr_growth, mu_lspr_growth, mu_lspr_PL, mu_lspr_mode1
    sigmas = sigma_final_lspr_growth, sigma_lspr_growth, sigma_lspr_PL, sigma_lspr_mode1
    
    return means, sigmas
                                       
if __name__ == '__main__':
    
   # base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PySpectrum/'
    
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
        
    daily_folder = '2021-06 (Growth)/2021-06-19 (growth)'
    
  #  list_of_growth = ['20210619-152004_Growth_12x1_580_4s_em50', '20210619-155338_Growth_12x1_570_4s_em50', '20210619-163455_Growth_12x1_560_4s_em50']  
 #   list_of_growth = ['20210619-133702_Growth_12x1_580_3s_em60', '20210619-140537_Growth_12x1_570_3s_em60', '20210619-142647_Growth_12x1_560_3s_em60']
 
    list_of_growth = ['20210618-175939_Growth_12x1_580_2s', '20210618-184443_Growth_12x1_570_2s', '20210619-130726_Growth_12x1_560_2s_em100']
    
    sub_folder_growth = 'processed_data_sustrate_bkg_True'
   
    daily_folder_PL = '2021-06 (Growth)/2021-06-23 (growth)/Luminescence_Steps/20210623-144358_Luminescence_Steps_12x10'
    sub_folder_PL = 'processed_data_luminiscence_sustrate_bkg_False/'
    
    common_path_PL = os.path.join(base_folder, daily_folder_PL, sub_folder_PL)
    
    save_folder = os.path.join(base_folder, daily_folder_PL, sub_folder_PL, 'S_fig_compare_PL_growth_final_2sEM100')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
#%%
    number_NP = 12  
    cols = [2, 3, 4] #[5,6, 7] # [8, 9, 10]
        
    plt.figure()

    for i in range(3):
        
        folder_growth =   list_of_growth[i] 
        common_path_growth  = os.path.join(base_folder, daily_folder, folder_growth, sub_folder_growth)
        
        number_col = cols[i]
        name_col = 'Col_%03d'%(number_col) 
        
        NP_growth, lspr_growth, NP_PL_list, lspr_PL = compare_final_spectrum(common_path_growth, common_path_PL, save_folder, name_col, number_col, number_NP) 
        
        total_time_growth, final_lspr_growth, lspr_PL = compare_lspr(NP_growth, lspr_growth, NP_PL_list, lspr_PL, common_path_growth, common_path_PL, save_folder, name_col)
        
        plt.plot(total_time_growth, final_lspr_growth, '*') # label = 'lspr growth live poly')
        plt.plot(total_time_growth, lspr_PL, 'o') #, label = 'lspr PL steps analysis poly')
  
    plt.xlabel('Total time growth (s)')
    plt.ylabel('Final LSPR')
    plt.legend(loc='lower right') 
    
    plt.xlim(40, 360)
    plt.ylim(540, 600)
    
    figure_name = os.path.join(save_folder, 'time_vs_final_lspr_PL.png') 
    
    #plt.show()
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    
#%%
    cols = [2, 3, 4] #[5,6, 7] # [8, 9, 10]

    for i in range(3):
        
        number_col = cols[i]
        
        name_col = 'Col_%03d'%(number_col)
        
        means, sigmas = plot_more_analysis_all(save_folder, name_col)
    