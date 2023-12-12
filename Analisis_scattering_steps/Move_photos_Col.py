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

    base_folder = r'C:\Users\lupau\OneDrive\Documentos'
    
    daily_folder = r'2023-03-17 80 nm SS Scattering post PL'
    
    parent_folder = os.path.join(base_folder, daily_folder)

    list_of_folders = os.listdir(parent_folder)
    list_of_folders =  [f for f in list_of_folders if re.search('Col',f)]
    
    savefolder = os.path.join(parent_folder, 'photos')
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    
    for f in list_of_folders:
    
        col = f.split('Col_')[1]
        
        folder = os.path.join(parent_folder, f)
         
        list_of_files = os.listdir(folder)
        
        photo_file = [f for f in list_of_files if re.search('Picture',f)][0]
        original = os.path.join(folder, photo_file)
        target = os.path.join(savefolder, 'Col_%s.tiff'%col)
        shutil.move(original, target)
        
    #%%
    list_of_files = os.listdir(savefolder)
    list_of_files.sort()
    print(list_of_files)
        
        