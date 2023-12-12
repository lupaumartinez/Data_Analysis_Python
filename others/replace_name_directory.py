# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os, re

base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/'

daily_folder = 'test/20210623-152506_Luminescence_Steps_12x6'

folder = os.path.join(base_folder, daily_folder)

list_of_files = os.listdir(folder)
list_of_files = [f for f in list_of_files if re.search('back_NPscan',f) ]
list_of_files.sort(reverse=True)

print(list_of_files)

L = len(list_of_files)

for i in range(L):
    filename = list_of_files[i]
    
    a = filename.split('NPscan_')[-1]
    b = int(a.split('.tiff')[0])
    
    NP = b + 12*4
    
    filepath = os.path.join(folder,filename)

    new_filename = 'back_NPscan_%03d.tiff'%NP
    
    new_filepath =  os.path.join(folder,new_filename)
    
    os.rename(filepath, new_filepath)