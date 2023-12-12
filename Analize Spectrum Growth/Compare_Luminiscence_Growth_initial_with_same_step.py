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

def compare_initial_spectrum(common_path_growth, common_path_PL, save_folder, name_col, number_col, number_NP, bool_norm):
    
    save_folder_files = os.path.join(save_folder, 'files_differences')
    if not os.path.exists(save_folder_files):
        os.makedirs(save_folder_files)
        
    if bool_norm:
        
        save_folder_files_norm = os.path.join(save_folder, 'files_differences_norm')
        if not os.path.exists(save_folder_files_norm):
            os.makedirs(save_folder_files_norm)
    
    common_path_PL = os.path.join(common_path_PL, name_col)
    
    common_path_PL_growth = os.path.join(common_path_growth, 'all_data_initial')
    
    list_of_folders = os.listdir(common_path_PL_growth)
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Files from initial growth PL', L, list_of_folders)
    
    common_path_PL = os.path.join(common_path_PL, 'luminescence_steps')
    
    list_of_folders_PL= os.listdir(common_path_PL)
    list_of_folders_PL = [f for f in list_of_folders_PL if re.search('NP',f)]
    list_of_folders_PL.sort()
    
    M = len(list_of_folders_PL)  
    
    print('Files from PL steps', M, list_of_folders_PL)
    
    if L == M:
    
        print('Plot initial grwoth PL, compare with PL steps')
        
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
    
    desired_range_PL = np.where((wavelength_PL >= first_wave) & (wavelength_PL <=end_wave))    
    desired_wave_PL = wavelength_PL[desired_range_PL] 
    x_PL = np.linspace(desired_wave_PL[0], desired_wave_PL[-1], 1000)
    npol_PL = 5
                
    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        
        NP = NP.split('.')[0]
        
      #  NP_PL = list_of_folders_PL[i].split('NP_')[-1]
     #   NP_PL = NP_PL.split('.')[0]
        
        NP_PL = (number_col- 1)*number_NP + int(NP)
        
        NP_PL = '%03d'%NP_PL
        
        print('Ploteo: from initial PL grwoth NP_%s, from initial PL steps  NP_%s'%(NP, NP_PL))
         
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
        
     #   norm_poly = (poly - min(poly))/(max(poly) - min(poly))
     #   PL_final_growth = (desired_PL_final_growth - min(poly))/(max(poly) - min(poly))
     
        base_notch = final_luminiscence[0]
        
        norm_poly = (poly -  base_notch)/(max(poly) -  base_notch)
        PL_final_growth = (desired_PL_final_growth -  base_notch)/(max(poly) -  base_notch)
        
        ## PL of steps
    
        name_file_PL = os.path.join(common_path_PL,  'Luminescence_Steps_Spectrum_%s_NP_%s.txt'%(name_col, NP_PL))
        
        b = np.loadtxt(name_file_PL, skiprows=1)
        luminiscence = b[:, 1]  #con sustrato
        
        desired_PL_steps_stokes = luminiscence[desired_range_PL]
        
        p_PL = np.polyfit(desired_wave_PL, desired_PL_steps_stokes, npol_PL)
        poly_PL = np.polyval(p_PL, x_PL)
        max_wave_poly_PL = round(x[np.argmax(poly_PL)],3)
    
        londa_spr_PL = np.round(max_wave_poly_PL,2)
        
  #      norm_poly_PL = (poly_PL - min(poly_PL))/(max(poly_PL) - min(poly_PL))
  #      PL_steps_stokes = (desired_PL_steps_stokes- min(poly_PL))/(max(poly_PL) - min(poly_PL))
         
        base_notch_PL = luminiscence[0]
        
        norm_poly_PL = (poly_PL - base_notch_PL)/(max(poly_PL) -base_notch_PL)
        PL_steps_stokes = (desired_PL_steps_stokes- base_notch_PL)/(max(poly_PL) - base_notch_PL)
        
        # Crude
        
        plt.figure()
        
        plt.title('NP_%s'%(NP))
        plt.plot(wavelength, final_luminiscence , label = 'growth initial PL')
        plt.plot(x, poly , 'r--')
        
        plt.plot(wavelength_PL, luminiscence, label = 'PL steps')
        plt.plot(x_PL, poly_PL, 'g--')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right')
        plt.ylim(0, 5000)
        
        figure_name = os.path.join(save_folder, 'compare_initial_PL_%s_NP_%s.png'%(name_col, NP_PL)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        # PLoteo Difference
        
       #   plt.figure()
        
        #  plt.title('NP_%s'%(NP))
        #  plt.plot(wavelength, luminiscence - final_luminiscence, 'k')
        #  plt.plot(x, poly_PL - poly, 'r--')
        
        #  plt.xlabel('Wavelength (nm)')
      #    plt.ylabel('Difference PL - PL growth live')
       #   plt.ylim(-1000, 3200)
        
        #  figure_name = os.path.join(save_folder, 'difference_initial_PL_%s_NP_%s.png'%(name_col, NP_PL)) 
        
        #  plt.savefig(figure_name , dpi = 400)
       #   plt.close()
        
        header_txt = "Wavelength, Difference PL - PL growth live, PL, PL growth live"
        data = np.array([wavelength, np.round(luminiscence - final_luminiscence, 3), luminiscence, final_luminiscence]).T
        name_data = os.path.join(save_folder_files, 'difference_initial_PL_%s_NP_%s.txt'%(name_col, NP_PL))
        np.savetxt(name_data, data, header = header_txt)
        
        header_txt = "Wavelength, Difference Poly PL - PL growth live, PL, PL growth live"
        data = np.array([x, np.round(poly_PL - poly, 3), poly_PL, poly]).T
        name_data = os.path.join(save_folder_files, 'difference_poly_initial_PL_%s_NP_%s.txt'%(name_col, NP_PL))
        np.savetxt(name_data, data, header = header_txt)
        
        if bool_norm:
            
            header_txt = "Wavelength, Difference Norm PL - PL growth live, norm PL, norm PL growth live"
            data = np.array([desired_wave, np.round(PL_steps_stokes-  PL_final_growth, 3), PL_steps_stokes, PL_final_growth]).T
            name_data = os.path.join(save_folder_files_norm, 'difference_initial_PL_%s_NP_%s.txt'%(name_col, NP_PL))
            np.savetxt(name_data, data, header = header_txt)
            
            header_txt = "Wavelength, Difference Norm Poly PL - PL growth live, PL, PL growth live"
            data = np.array([x, np.round(norm_poly_PL - norm_poly, 3), norm_poly_PL, norm_poly]).T
            name_data = os.path.join(save_folder_files_norm, 'difference_poly_initial_PL_%s_NP_%s.txt'%(name_col, NP_PL))
            np.savetxt(name_data, data, header = header_txt)
        
        #PLoteo Normalziado
        
        plt.figure()
        plt.title('NP_%s'%(NP))
        plt.plot(desired_wave, PL_final_growth , label = 'growth initial PL')
        plt.plot(x, norm_poly , 'r--', label = 'poly growth initial PL')
        plt.plot(desired_wave_PL, PL_steps_stokes, label = 'PL steps stokes')
        plt.plot(x_PL, norm_poly_PL, 'g--', label = 'poly PL steps stokes')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Intensity')
        plt.legend(loc='upper right') 
        
        plt.ylim(0, 1.2)
        
        figure_name = os.path.join(save_folder, 'norm_compare_initial_PL_%s_NP_%s.png'%(name_col, NP_PL)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        #data save
        NP_growth.append(int(NP))
        lspr_growth.append(londa_spr_growth)
        
        NP_PL_list.append(int(NP_PL))
        lspr_PL.append(londa_spr_PL)
        
        
        # PLoteo Difference
        
        #  plt.figure()
        
       #   plt.title('NP_%s'%(NP))
      #    plt.plot(x_PL, norm_poly_PL - norm_poly, 'k-')
        
        #  plt.xlabel('Wavelength (nm)')
       #   plt.ylabel('Difference Normalized PL - PL growth live')
        
       #   plt.ylim(-1, 1)
        
       #   figure_name = os.path.join(save_folder, 'norm_difference_initial_PL_%s_NP_%s.png'%(name_col, NP_PL)) 

      #    plt.savefig(figure_name , dpi = 400)
      #    plt.close()
        
    return NP_growth, lspr_growth, NP_PL_list, lspr_PL


def plot_all_difference(save_folder, name_col, bool_norm):
    
    if bool_norm:
        folder =  os.path.join(save_folder, name_col, 'files_differences_norm')
       # fig_lim = -1, 1
        fig_lim = -1, 1
    else:
        folder =  os.path.join(save_folder, name_col, 'files_differences')
        fig_lim = -1500, 3500
        
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('difference',f) and not re.search('poly',f)]
    list_of_folders.sort()
    
    plt.figure()
    plt.title('%s'%(name_col))
    
    for e in list_of_folders:
        
        NP = e.split('NP_')[-1]
        
        NP = NP.split('.')[0]
    
        file = os.path.join(folder, e)
        data = np.loadtxt(file)
        wavelength = data[:, 0]
        difference = data[:, 1]
        
        file_poly = os.path.join(folder, 'difference_poly_initial_PL_%s_NP_%s.txt'%(name_col, NP))
        data = np.loadtxt(file_poly)
        x = data[:, 0]
        poly_difference = data[:, 1]
        
        plt.plot(wavelength, difference, label = '%s'%(NP))
        plt.plot(x, poly_difference, 'k--')
        
        #label = 'PL steps', )
        
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Difference PL - PL growth live')
    plt.legend(loc='upper right', fontsize = 'x-small')
    plt.ylim(fig_lim)
    plt.xlim(530, 660)
    
    figure_name = os.path.join(save_folder, 'all_difference_initial_PL_%s_norm_%s.png'%(name_col, str(bool_norm)) )
    
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    return

def plot_all(save_folder, name_col, bool_norm):
    
    if bool_norm:
        folder =  os.path.join(save_folder, name_col, 'files_differences_norm')
       # fig_lim = -0.1, 1.1
        fig_lim = 0, 1.2
    else:
        folder =  os.path.join(save_folder, name_col, 'files_differences')
        fig_lim = 500, 5000
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('difference',f) and not re.search('poly',f)]
    list_of_folders.sort()
    
    plt.figure()
    plt.title('%s'%(name_col))
    
    for e in list_of_folders:
        
        NP = e.split('NP_')[-1]
        
        NP = NP.split('.')[0]
    
        file = os.path.join(folder, e)
        data = np.loadtxt(file)
        wavelength = data[:, 0]
        pl = data[:, 2]
        pl_live = data[:, 3]
        
        file_poly = os.path.join(folder, 'difference_poly_initial_PL_%s_NP_%s.txt'%(name_col, NP))
        data = np.loadtxt(file_poly)
        x = data[:, 0]
        poly_pl = data[:, 2]
        poly_pl_live = data[:, 3]
        
      #  plt.plot(wavelength, pl, 'C1')#, label = '%s'%(NP))
      #  plt.plot(wavelength, pl_live, 'C0')#, label = '%s'%(NP))
        
        plt.plot(x, poly_pl, 'C2')#, label = '%s'%(NP))
        plt.plot(x, poly_pl_live, 'C3')#, label = '%s'%(NP))
        
        #label = 'PL steps', )
        
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('PL')
 #   plt.legend(loc='upper right', fontsize = 'x-small')
    plt.ylim(fig_lim)
    plt.xlim(540, 650)
    
    figure_name = os.path.join(save_folder, 'all_initial_PL_%s_norm_%s.png'%(name_col, str(bool_norm)))
    
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    return
        

                                       
if __name__ == '__main__':
    
   # base_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PySpectrum/'
    
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
        
  #  daily_folder = '2021-06 (Growth)/2021-06-19 (growth)'
  #  list_of_growth = ['20210619-152004_Growth_12x1_580_4s_em50', '20210619-155338_Growth_12x1_570_4s_em50', '20210619-163455_Growth_12x1_560_4s_em50']  
  #  list_of_growth = ['20210619-133702_Growth_12x1_580_3s_em60', '20210619-140537_Growth_12x1_570_3s_em60', '20210619-142647_Growth_12x1_560_3s_em60']
  #  list_of_growth = ['20210618-175939_Growth_12x1_580_2s', '20210618-184443_Growth_12x1_570_2s', '20210619-130726_Growth_12x1_560_2s_em100']
  #  daily_folder_PL = '2021-06 (Growth)/2021-06-18 (growth)/20210618-145900_Luminescence_Steps_12x10_pre_growth'
  #  number_NP = 12  
  #  cols = [8, 9, 10] #[2, 3, 4]  #[5,6, 7] 
  
 #   daily_folder = '2021-08 (Growth PL circular)/2021-08-04 (pre growth, PL circular)/laser_polarizacion_circular_MAL/'
 #   list_of_growth = ['20210804-171638_Growth_10x1']   
 #   daily_folder_PL = '2021-08 (Growth PL circular)/2021-08-04 (pre growth, PL circular)/laser_polarizacion_circular_MAL/20210804-175940_Luminescence_Steps_10x12'
 #   number_NP = 10
 #   cols = [1]
    
    daily_folder = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/growth/'
    
   # list_of_growth = ['20210809-190759_Growth_10x1_col3_80s']#, '20210809-192328_Growth_10x1_col4_140s', '20210809-194745_Growth_10x1_col5_200s']
   # number_NP = 10
   # cols = [3]#, 4, 5]  
   
    list_of_growth = ['20210809-165810_Growth_10x2_col7y8_560nm', '20210809-172719_Growth_10x2_col9y10_570nm', '20210809-180739_Growth_10x2_col11y12_580nm']
    number_NP = 10
    cols =  [7, 9, 11]  
   
    daily_folder_PL = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/pre_growth/20210809-150932_Luminescence_Steps_10x12'

    sub_folder_growth = 'range_growth_sustrate_bkg_False'
    sub_folder_PL = 'range_luminiscence_sustrate_bkg_False/'
    
    common_path_PL = os.path.join(base_folder, daily_folder_PL, sub_folder_PL)
    
    save_folder = os.path.join(base_folder, daily_folder_PL, sub_folder_PL, 'crude_fig_compare_PL_growth_initial')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    plt.figure()
    
    bool_norm = True

    for i in range(len(cols)):
        
        folder_growth =   list_of_growth[i] 
        common_path_growth  = os.path.join(base_folder, daily_folder, folder_growth, sub_folder_growth)
        
        number_col = cols[i]
        name_col = 'Col_%03d'%(number_col)
        
        save_folder_col = os.path.join(save_folder, name_col)
        if not os.path.exists(save_folder_col):
            os.makedirs(save_folder_col)
        
        NP_growth, lspr_growth, NP_PL_list, lspr_PL = compare_initial_spectrum(common_path_growth, common_path_PL, save_folder_col, name_col, number_col, number_NP, bool_norm ) 
   
    for i in range(len(cols)):
        
        number_col = cols[i]
        name_col = 'Col_%03d'%(number_col)   
        
        plot_all_difference(save_folder, name_col, False )
        plot_all(save_folder, name_col, False  )
        
        if bool_norm:
            
            plot_all_difference(save_folder, name_col, True)
            plot_all(save_folder, name_col, True )