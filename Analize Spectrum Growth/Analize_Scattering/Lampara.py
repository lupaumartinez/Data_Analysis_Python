#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 21:44:16 2021

@author: luciana
"""

import numpy as np
import matplotlib.pyplot as plt
import os


base_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/Luciana Martinez/Programa_Python/Analize Spectrum Growth/Analize_Scattering/'

file_old = os.path.join(base_folder, 'Resumen_lamparaIR', 'lamp_IR_unpol_grade_2_PySpectrum.txt')#'lamp_IR_unpol_AndorSolis.txt')
lampara_old = np.loadtxt(file_old, comments = '#')
wave_lampara_old = np.array(lampara_old[:, 0])
espectro_lampara_old = np.array(lampara_old[:, 1:])

file = os.path.join(base_folder, 'Resumen_lamparaIR_new', 'LampIR_post alineacion_grating 39.asc' )
lampara = np.loadtxt(file)
wave_lampara = np.array(lampara[:, 0])
espectro_lampara = np.array(lampara[: ,1])

file_py = os.path.join(base_folder, 'Resumen_lamparaIR_new', 'lamparaIR_grade_2.txt')
lampara_new = np.loadtxt(file_py, comments = '#')
wave_lampara_new = np.array(lampara_new[:, 0])
espectro_lampara_new = np.array(lampara_new[:, 1])

#norm_new = (espectro_lampara_new-min(espectro_lampara_new))/(max(espectro_lampara_new)-min(espectro_lampara_new))

plt.plot(wave_lampara_new, espectro_lampara_new, 'r')
plt.plot(wave_lampara_old, espectro_lampara_old/max(espectro_lampara_old), 'b')
plt.plot(wave_lampara, espectro_lampara/max(espectro_lampara), 'k')
plt.show()

#name = os.path.join(base_folder, 'Resumen_lamparaIR_new', 'LampIR_post alineacion_grating 39.txt')
#data = np.array([wave_lampara_new , espectro_lampara_new ]).T
#header_text = 'wavelength, intensity'
#np.savetxt(name, data, header = header_text)