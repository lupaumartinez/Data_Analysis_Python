#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:12:33 2019

Replace comma by dot in every .asc file inside a folder

@author: Mariano Barella

CIBION, Buenos Aires, Argentina
"""

import os, re

base_folder = base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/Luciana Martinez/Programa_Python/Analize Spectrum Growth/Analize_Scattering/'
folder = os.path.join(base_folder, 'Resumen_lamparaIR_new')

list_of_files = os.listdir(folder)
list_of_files.sort()
list_of_files = [f for f in list_of_files if not os.path.isdir(folder+f)]
list_of_files = [f for f in list_of_files if re.search('.asc',f)]

L = len(list_of_files)

for i in range(L):
    filename = list_of_files[i]
    filepath = os.path.join(folder,filename)
    with open(filepath,'r+') as f:
        text = f.read()
        f.seek(0)
        f.truncate()
        f.write(text.replace(',','.'))
        f.close()