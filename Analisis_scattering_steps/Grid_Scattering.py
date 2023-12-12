# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:23:45 2022

@author: Luciana
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def ideal_grid(save_folder, n, N, d_n, d_N, scattering, bins):
    
    datos = np.zeros((3, n*N))
    
    plt.figure()
    plt.ylim(n*d_n, -d_n)
    plt.xlim(-d_N, N*d_N)
    
    for k in range(N):
       for i in range(n):
           
           y =  i*d_n
           x = k*d_N
           
           step =  k*n + i
           
           print(step)
           
           datos[1, step]= y
           datos[0, step]= x
           
           londa_max = scattering[step]
           
           datos[2, step] = londa_max
           alpha = 0.5
           
           if 400<=londa_max < bins[1]:
               c = 'm'
               plt.plot(x, y, 'o',markersize = '10', color = c, alpha = alpha)
               plt.text(x-0.5 , y-1 , '%d'%londa_max)
               
           if bins[1]<=londa_max < bins[2]:
               c = 'C1'
               plt.plot(x, y,  '^',markersize = '10', color = c , alpha = alpha)
               plt.text(x-0.5 , y-1 , '%d'%londa_max)
               
           if bins[2]<=londa_max < bins[3]:
              c = 'C0'
              plt.plot(x, y, 'd', markersize = '10',color = c , alpha = alpha)
              plt.text(x-0.5 , y-1 , '%d'%londa_max)
              
           if bins[3]<=londa_max <= 900:
              c = 'b'
              plt.plot(x, y, '*', markersize = '12', color = c , alpha = alpha)
              plt.text(x-0.5 , y-1 , '%d'%londa_max)
              
    name = os.path.join(save_folder, 'grid_scattering_printing.png')
    plt.savefig(name, dpi = 400)

    return datos

def plot_histogram(save_folder, list_londa_max, y_star, y_err_star, y_sph, y_err_sph):
    
    medium =  (y_sph + y_star)/2

    bins = 4
    step_bin = ((y_star + 2*y_err_star) - (y_sph - 2*y_err_sph))/bins
    rango = [y_sph - 2*y_err_sph, y_sph - 2*y_err_sph + bins*step_bin]
    ylim = len(list_londa_max) #100
    
    plt.figure()
    plt.axvspan(y_star - 2*y_err_star, y_star + 2*y_err_star, alpha=0.2, color='b')
    plt.axvspan(y_sph - 2*y_err_sph, y_sph + 2*y_err_sph, alpha=0.2, color='m')
    
    plt.vlines(y_star - 2*y_err_star, 0, ylim, color = 'k', alpha = 0.1)
    plt.vlines(y_sph + 2*y_err_sph, 0, ylim, color = 'k', alpha = 0.1)
    plt.vlines(medium, 0, ylim, color = 'k', alpha = 0.1)
    
    nbins, bins, patches = plt.hist(list_londa_max, bins = bins, range = rango, rwidth=True, align = 'mid', color = 'r', alpha = 0.5)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Frequency')
    plt.ylim(0, ylim)
    plt.show()

    name = os.path.join(save_folder, 'hist_all_scattering_printing.png')
    plt.savefig(name, dpi = 400)

    return nbins, bins

plt.close('all')

n = 10 #cantidad de particulas por columna
N = 6 #cantidad de columnas
d_n = 3 #espaciado entre particulas
d_N = 3 #espaciado entre columnas

folder = r'C:\Ubuntu_archivos\Printing\2022-07-20 Nanostars R20 taco4 SEM\G1\scattering'
file = os.path.join(folder, 'londa_max.txt')
save_folder = folder

scattering  = np.loadtxt(file, skiprows = 1)
nbins, bins = plot_histogram(save_folder, scattering, 850, 20, 585, 20)

total_NP = n*N
print(total_NP, len(scattering))

print(scattering)
grid = ideal_grid(save_folder, n, N, d_n, d_N, scattering, bins)