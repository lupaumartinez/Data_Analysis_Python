# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:22:25 2022

@author: Luciana
"""

import os
import re
import numpy as np
import shutil

if __name__ == '__main__':

    base_folder = r'C:\Ubuntu_archivos\Printing'
    
    daily_folder = r'2022-07-06 Nanostars R20 drop and cast'
    
    folder_NPs = '20220706-183129_Scattering_Steps_Load_grid_Final'
    
    parent_folder = os.path.join(base_folder, daily_folder, folder_NPs)

    list_of_folders = os.listdir(parent_folder)
    list_of_folders =  [f for f in list_of_folders if re.search('Spectrum_luminescence_',f)]
    
    savefolder = os.path.join(parent_folder, 'more_NPs_per_photo')
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    
    for f in list_of_folders:
    
        NP = f.split('NP_')[1]
        
        #save_folder = os.path.join(savefolder, 'Col_%s'%NP)
        #if not os.path.exists(save_folder):
        #    os.makedirs(save_folder)
     
        folder = os.path.join(parent_folder, f)
         
        list_of_files = os.listdir(folder)
        
        photo_file = [f for f in list_of_files if re.search('Picture',f)][0]
        original = os.path.join(folder, photo_file)
        target = os.path.join(savefolder, 'Col_%s.tiff'%NP)
        shutil.move(original, target)
        
    #%%
    list_of_files = os.listdir(savefolder)
    list_of_files.sort()
    print(list_of_files)
        
        