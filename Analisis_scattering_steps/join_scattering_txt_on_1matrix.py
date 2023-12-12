# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:37:04 2022

@author: Luciana
"""

import os
import re
import numpy as np


base_folder = r'C:\Ubuntu_archivos\Printing\Nanostars\NanoStarts P2\2020-03-18 (P2 R5)\Scattering_lamp_IR_unpol'

laser_folder = 'laser_verde_impresion_4mW'

folder = os.path.join(base_folder, laser_folder, 'photos')

save_folder = os.path.join(r'C:\Ubuntu_archivos\Printing\2022-07-05 Nanostars R5 Scattering del 2020-03-18', '1', 'all_txt_%s'%laser_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

list_files = os.listdir(folder)
list_files =  [f for f in list_files if re.search('Col',f)]
list_files.sort()

totalNP = 10 #viene de print(len(list_files_spec)), el caso maximo de NP, el resto queda con ceros
large_data =  514 #514  #viene de print("check large data", data.shape[0])

print(list_files)

for l in list_files:

    matrix = np.zeros((large_data, totalNP +1))  
    
    f = os.path.join(folder, l, 'normalized_%s'%l)
    
    list_files_spec = os.listdir(f)
    list_files_spec  =  [f for f in list_files_spec if re.search('NP',f)]
    list_files_spec.sort()
    
    print(l,len(list_files_spec))
    
    for k in list_files_spec:
    
        file_NP = os.path.join(f, k)
        
        name_NP = file_NP.split('.txt')[0]
        name_NP =  name_NP.split('NP_')[1]
        number_NP = int(name_NP)
        print(number_NP)
    
        data = np.loadtxt(file_NP)
        
        print("check large data", data.shape[0])
        
        wavelength = data[:, 0]
        spec =  data[:, 1]
        spec_not_norm =  data[:, 2]
        
        matrix[:, 0] = wavelength
        matrix[:, number_NP+1] = spec
        
        if len(list_files_spec)< totalNP:
            print(file_NP)
            print('menos NPs', matrix[:, 1])
                
    name = os.path.join(save_folder, 'matrix_%s.txt'%l)
    np.savetxt(name, matrix)