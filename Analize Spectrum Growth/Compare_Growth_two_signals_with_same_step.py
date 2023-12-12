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

def compare_spectrum(common_path_growth_1, common_path_growth_2, save_folder, bool_norm):
    
    save_folder_files = os.path.join(save_folder, 'files_differences')
    if not os.path.exists(save_folder_files):
        os.makedirs(save_folder_files)
        
    if bool_norm:
        
        save_folder_files_norm = os.path.join(save_folder, 'files_differences_norm')
        if not os.path.exists(save_folder_files_norm):
            os.makedirs(save_folder_files_norm)
    
    common_path_PL_growth_1 = os.path.join(common_path_growth_1, 'all_data_final')
    
    common_path_PL_growth_2 = os.path.join(common_path_growth_2, 'all_data_initial')
    
    list_of_folders = os.listdir(common_path_PL_growth_1)
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
    
    print('Files from final growth PL 1', L, list_of_folders)
    
    list_of_folders_2= os.listdir(common_path_PL_growth_2)
    list_of_folders_2 = [f for f in list_of_folders_2 if re.search('NP',f)]
    list_of_folders_2.sort()
    M = len(list_of_folders_2)  
    
    print('Files from initial growth PL 2', M, list_of_folders_2)
    
    if L == M:
    
        print('OK')
        
    else:
        
        print('The files do not have the same NP. Check files')
        L = min(L, M)
        
    NP_growth = []
    lspr_growth = []
    
    NP_growth_2 = []
    lspr_growth_2 = []
    
    first_wave = 545  #538,5 #545
    end_wave = 690  #641.5  #641.5 #620
    
    #para PL growth
    name_file = os.path.join(common_path_PL_growth_1,  list_of_folders[0])
    a = np.loadtxt(name_file, skiprows=1)
    wavelength = a[:, 0]
    
    desired_range = np.where((wavelength >= first_wave) & (wavelength <=end_wave))    
    desired_wave = wavelength[desired_range]
    x = np.linspace(desired_wave[0], desired_wave[-1], 1000)
    npol = 5
    
    #para PL steps
    name_file = os.path.join(common_path_PL_growth_2,  list_of_folders_2[0])
    b = np.loadtxt(name_file, skiprows=1)
    wavelength_PL = b[:, 0]
    
    desired_range_PL = np.where((wavelength_PL >= first_wave) & (wavelength_PL <=end_wave))    
    desired_wave_PL = wavelength_PL[desired_range_PL] 
    x_PL = np.linspace(desired_wave_PL[0], desired_wave_PL[-1], 1000)
    npol_PL = 5
                
    for i in range(L):
        
        NP = list_of_folders[i].split('NP_')[-1]
        
        NP = NP.split('.')[0]
        
        NP_2 =  int(NP)
        
        NP_2 = '%03d'%NP_2
        
        print('Ploteo: from final PL grwoth 1 NP_%s, from initial PL grwoth 2 NP_%s'%(NP, NP_2))
         
        ## PL of Growth in live
        
        name_file = os.path.join(common_path_PL_growth_1,  list_of_folders[i])
        
        a = np.loadtxt(name_file, skiprows=1)
        final_luminiscence = a[:, 1]
        desired_PL_final_growth = final_luminiscence[desired_range]

        p = np.polyfit(desired_wave, desired_PL_final_growth, npol)
        poly = np.polyval(p, x)
        max_wave_poly = round(x[np.argmax(poly)],3)
        
        londa_spr_growth = np.round(max_wave_poly,2)
     
        base_notch = final_luminiscence[0]
        
        norm_poly = (poly -  base_notch)/(max(poly) -  base_notch)
        PL_final_growth = (desired_PL_final_growth -  base_notch)/(max(poly) -  base_notch)
        
        ## PL of steps
        name_file_PL = os.path.join(common_path_PL_growth_2, 'initial_NP_%s.txt'%NP_2)
        
        b = np.loadtxt(name_file_PL, skiprows=1)
        luminiscence = b[:, 1]  #con sustrato
        
        desired_PL_steps_stokes = luminiscence[desired_range_PL]
        
        p_PL = np.polyfit(desired_wave_PL, desired_PL_steps_stokes, npol_PL)
        poly_PL = np.polyval(p_PL, x_PL)
        max_wave_poly_PL = round(x[np.argmax(poly_PL)],3)
    
        londa_spr_growth_2  = np.round(max_wave_poly_PL,2)
         
        base_notch_PL = luminiscence[0]
        
        norm_poly_PL = (poly_PL - base_notch_PL)/(max(poly_PL) -base_notch_PL)
        PL_steps_stokes = (desired_PL_steps_stokes- base_notch_PL)/(max(poly_PL) - base_notch_PL)
        
        # Crude
        
        plt.figure()
        
        plt.title('NP_%s'%(NP))
        plt.plot(wavelength, final_luminiscence , label = 'growth final PL 1')
        plt.plot(x, poly , 'r--')
        
        plt.plot(wavelength_PL, luminiscence,  label = 'growth initial PL 2')
        plt.plot(x_PL, poly_PL, 'g--')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right')
        plt.ylim(0, 5000)
        
        figure_name = os.path.join(save_folder, 'compare_growth_final_initial_NP_%s.png'%(NP_2)) 
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
        
        header_txt = "Wavelength, Difference PL growth live 2 - PL growth live 1,PL growth live 2, PL growth live 1"
        data = np.array([wavelength, np.round(luminiscence - final_luminiscence, 3), luminiscence, final_luminiscence]).T
        name_data = os.path.join(save_folder_files, 'difference_growth_final_initial_NP_%s.txt'%(NP_2)) 
        np.savetxt(name_data, data, header = header_txt)
        
        header_txt =" Wavelength, Difference Poly PL growth live 2 - PL growth live 1,PL growth live 2, PL growth live 1"
        data = np.array([x, np.round(poly_PL - poly, 3), poly_PL, poly]).T
        name_data = os.path.join(save_folder_files, 'difference_poly_growth_final_initial_NP_%s.txt'%(NP_2)) 
        np.savetxt(name_data, data, header = header_txt)
        
        if bool_norm:
            
            header_txt = "Wavelength, Difference Norm PL growth live 2 - PL growth live 1, Norm PL growth live 2, Norm PL growth live 1"
            data = np.array([desired_wave, np.round(PL_steps_stokes-  PL_final_growth, 3), PL_steps_stokes, PL_final_growth]).T
            name_data = os.path.join(save_folder_files_norm, 'difference_growth_final_initial_NP_%s.txt'%(NP_2)) 
            np.savetxt(name_data, data, header = header_txt)
            
            header_txt = "Wavelength, Difference Norm Poly PL growth live 2 - PL growth live 1, Norm PL growth live 2, Norm PL growth live 1"
            data = np.array([x, np.round(norm_poly_PL - norm_poly, 3), norm_poly_PL, norm_poly]).T
            name_data = os.path.join(save_folder_files_norm, 'difference_poly_growth_final_initial_NP_%s.txt'%(NP_2))
            np.savetxt(name_data, data, header = header_txt)
        
        #PLoteo Normalziado
        
        plt.figure()
        plt.title('NP_%s'%(NP))
        plt.plot(desired_wave, PL_final_growth , label = 'growth final PL 1')
        plt.plot(x, norm_poly , 'r--')# label = 'poly growth final PL 1')
        plt.plot(desired_wave_PL, PL_steps_stokes, label = 'growth initial PL 2')
        plt.plot(x_PL, norm_poly_PL, 'g--')# label = 'poly growth final PL 2')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Intensity')
        plt.legend(loc='upper right') 
        
        plt.ylim(0, 1.2)
        
        figure_name = os.path.join(save_folder, 'norm_growth_final_initial_NP_%s.png'%(NP_2)) 
        plt.savefig(figure_name , dpi = 400)
        plt.close()
        
        #data save
        NP_growth.append(int(NP))
        lspr_growth.append(londa_spr_growth)
        
        NP_growth_2.append(int(NP_2))
        lspr_growth_2.append(londa_spr_growth_2)
        
        
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
        
    return


def plot_all_difference(save_folder, bool_norm):
    
    if bool_norm:
        folder =  os.path.join(save_folder, 'files_differences_norm')
       # fig_lim = -1, 1
        fig_lim = -1, 1
    else:
        folder =  os.path.join(save_folder, 'files_differences')
        fig_lim = -1500, 3500
        
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('difference',f) and not re.search('poly',f)]
    list_of_folders.sort()
    
    plt.figure()
    
    for e in list_of_folders:
        
        NP = e.split('NP_')[-1]
        
        NP = NP.split('.')[0]
    
        file = os.path.join(folder, e)
        data = np.loadtxt(file)
        wavelength = data[:, 0]
        difference = data[:, 1]
        
        file_poly = os.path.join(folder, 'difference_growth_final_initial_NP_%s.txt'%(NP)) 
        data = np.loadtxt(file_poly)
        x = data[:, 0]
        poly_difference = data[:, 1]
        
        plt.plot(wavelength, difference, label = '%s'%(NP))
        plt.plot(x, poly_difference, 'k--')
        
        #label = 'PL steps', )
        
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Difference PL growth live 2 - PL growth live 1')
    plt.legend(loc='upper right', fontsize = 'x-small')
    plt.ylim(fig_lim)
    plt.xlim(530, 660)
    
    figure_name = os.path.join(save_folder, 'all_difference_growth_final_initial_norm_%s.png'%( str(bool_norm)) )
    
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    return

def plot_all(save_folder, bool_norm):
    
    if bool_norm:
        folder =  os.path.join(save_folder, 'files_differences_norm')
       # fig_lim = -0.1, 1.1
        fig_lim = 0, 1.2
    else:
        folder =  os.path.join(save_folder, 'files_differences')
        fig_lim = 500, 5000
    
    list_of_folders = os.listdir(folder)
    list_of_folders = [f for f in list_of_folders if re.search('difference',f) and not re.search('poly',f)]
    list_of_folders.sort()
    
    plt.figure()
    
    for e in list_of_folders:
        
        NP = e.split('NP_')[-1]
        
        NP = NP.split('.')[0]
    
        file = os.path.join(folder, e)
        data = np.loadtxt(file)
        wavelength = data[:, 0]
        pl = data[:, 2]
        pl_live = data[:, 3]
        
        file_poly = os.path.join(folder, 'difference_poly_growth_final_initial_NP_%s.txt'%(NP)) 
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
    plt.ylabel('PL growth live')
 #   plt.legend(loc='upper right', fontsize = 'x-small')
    plt.ylim(fig_lim)
    plt.xlim(540, 650)
    
    figure_name = os.path.join(save_folder, 'all_growth_final_initial_norm_%s.png'%( str(bool_norm)) )
    
    plt.savefig(figure_name , dpi = 400)
    plt.close()
    
    return
        

                                       
if __name__ == '__main__':
    
    base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'
    
    daily_folder_1 = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/growth/'
   
    daily_folder_2 = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/post_growth/'
    
   # folder_growth = ['20210809-165810_Growth_10x2_col7y8_560nm', '20210809-203300_Growth_10x1_col7_80seg_agua']
   # folder_growth = ['20210809-172719_Growth_10x2_col9y10_570nm', '20210809-205507_Growth_10x1_col9_80seg_agua']
   # folder_growth = ['20210809-180739_Growth_10x2_col11y12_580nm', '20210809-211130_Growth_10x1_col11_80seg_agua']
   # folder_growth = ['20210809-190759_Growth_10x1_col3_80s', '20210809-212908_Growth_10x3_col3_80seg_agua']
   # folder_growth = ['20210809-192328_Growth_10x1_col4_140s', '20210809-212908_Growth_10x3_col4_80seg_agua'] #NP_2 =  int(NP) + 10
   # folder_growth = ['20210809-194745_Growth_10x1_col5_200s', '20210809-212908_Growth_10x3_col5_80seg_agua']  #NP_2 =  int(NP) + 20
   
    
   # folder_growth = ['20210809-203300_Growth_10x1_col7_80seg_agua', '20210809-203300_Growth_10x1_col7_80seg_agua']
  #  folder_growth = ['20210809-205507_Growth_10x1_col9_80seg_agua' ,'20210809-205507_Growth_10x1_col9_80seg_agua']
    #folder_growth = ['20210809-211130_Growth_10x1_col11_80seg_agua', '20210809-211130_Growth_10x1_col11_80seg_agua']
    #folder_growth = ['20210809-212908_Growth_10x3_col3_80seg_agua', '20210809-212908_Growth_10x3_col3_80seg_agua']
   # folder_growth = ['20210809-212908_Growth_10x3_col4_80seg_agua', '20210809-212908_Growth_10x3_col4_80seg_agua']
  #  folder_growth = ['20210809-212908_Growth_10x3_col5_80seg_agua', '20210809-212908_Growth_10x3_col5_80seg_agua']
    folder_growth = ['20210809-221629_Growth_6x1_col6_80seg_agua', '20210809-221629_Growth_6x1_col6_80seg_agua']
    
    folder_growth_1 = folder_growth[0]
    folder_growth_2 = folder_growth[1]
    
    sub_folder_growth = 'range_growth_sustrate_bkg_False'
    
    common_path_1 = os.path.join(base_folder, daily_folder_2, folder_growth_1, sub_folder_growth)
    common_path_2 = os.path.join(base_folder, daily_folder_2, folder_growth_2, sub_folder_growth)
    
    save_folder = os.path.join(base_folder, daily_folder_2, folder_growth_2, 'fig_compare_PL_growth_initial_final_all_water')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    bool_norm = True
    
    compare_spectrum(common_path_1, common_path_2, save_folder, bool_norm) 

    plot_all_difference(save_folder, False )
    plot_all(save_folder, False  )
    
    if bool_norm:
        
        plot_all_difference(save_folder,  True)
        plot_all(save_folder, True )