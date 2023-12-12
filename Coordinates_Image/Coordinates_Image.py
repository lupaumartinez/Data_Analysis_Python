#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:40:42 2021

@author: luciana
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from skimage import io
import os

def bin_image(image, sigma):
    
    blur = skimage.filters.gaussian(image, sigma=sigma)
    
    return blur

def bin_image_2(image, filter_value):
    
    bin_image = np.zeros((image.shape[0], image.shape[1]))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            if image[i, j] > filter_value:

                bin_image[i, j] = 1
                
    return bin_image

def make_sorted(coordinates):
    
    coordinate_x = coordinates[:, 1]
    coordinate_y = coordinates[:, 0]
    
    zipped_lists = zip(coordinate_y, coordinate_x)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    new_coordinate_y, new_coordinate_x = [list(tuple) for tuple in tuples]
    
    new_coordinates = np.zeros(coordinates.shape)
    new_coordinates[:, 1] = new_coordinate_x
    new_coordinates[:, 0] = new_coordinate_y
    
    N = len(new_coordinate_x)
    print(N)
    
    return new_coordinates

def make_sorted_2(coordinates, list_N):
    
    coordinate_x = coordinates[:, 1]
    coordinate_y = coordinates[:, 0]
    
    new_coordinate_x = np.zeros(len(coordinate_x))
    new_coordinate_y = np.zeros(len(coordinate_y))
    
    a = 0
    
    for i in range(len(list_N)):
        
        b = list_N[i] + a
        
        zipped_lists = zip(coordinate_x[a:b], coordinate_y[a:b])    
        sorted_pairs = sorted(zipped_lists)
        tuples = zip(*sorted_pairs)
        new_coordinate_x[a:b], new_coordinate_y[a:b] = [list(tuple) for tuple in tuples]
        
        a = b
    
    print(new_coordinate_x , new_coordinate_y)
    
    new_coordinates = np.zeros(coordinates.shape)
    new_coordinates[:, 1] = new_coordinate_x
    new_coordinates[:, 0] = new_coordinate_y
    
    return new_coordinates

def delete_coordinates(pree_coordinates, index_delete):
    
    if index_delete == []:
        
        coordinates = pree_coordinates
        
    else:
        
        x = pree_coordinates[:, 1]
        y = pree_coordinates[:, 0]
        i = 0
        for e in index_delete:
            
            x = np.delete(x, e - i)
            y = np.delete(y, e - i)
            i = i + 1
        
        N = len(x)
        
        coordinates = np.zeros((N, N))
        coordinates[:, 1] = x
        coordinates[:, 0] = y
        
        print(N)
        
    return coordinates

def make_grid(coordinates, pixel_size):
    
    N = coordinates.shape[0]
    
    grid = np.zeros((3, N))
    
    for i in range(N):
    
        x = (coordinates[i, 1] - coordinates[0, 1])*pixel_size
        y = (coordinates[i, 0]- coordinates[0, 0])*pixel_size
        
        grid[0, i] = x
        grid[1, i] = y
    
    return grid

def ideal_grid(n, N, d_n, d_N):
    
   # n = grid[0] #particulas por columna
   # N = grid[1] #cantidad de columnas
   # d_n = grid[2] #espaciado entre particulas
   # d_N = grid[3] #espaciado entre columnas

    datos = np.zeros((3, n*N))

    i = 0
    k = 0

    for i in range(n):
       for k in range(N):
           datos[1, k*n+i]= k*d_N
           datos[0, k*n+i]= i*d_n
           
    return datos

folder = 'C:/Users/lupau/OneDrive/Documentos/Luciana Martinez/Programa_Python/'
file = 'Grid_canon.jpg'#'Image_Letters.jpg'
file = os.path.join(folder, file)

image_RGB = io.imread(file)
im = skimage.color.rgb2gray(image_RGB)
im = im.T
#im1 = bin_image(im1, sigma = 0.5)
#im2 = bin_image_2(im1, 0.59)

coordinates_1 = peak_local_max(im, min_distance=10, threshold_abs = 0.53)

coordinates_2 = make_sorted(coordinates_1)

index_delete = [18,30, 39, 105] # [19, 68, 92, 94]
coordinates = delete_coordinates(coordinates_2, index_delete)

list_N_row = [10, 8, 10, 9, 10, 8, 10, 10, 9, 10, 10, 9]   # [11, 11, 11, 12, 11, 12, 11, 11, 12, 11]
coordinates = make_sorted_2(coordinates, list_N_row)

#%%
    
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)

ax = axes.ravel()
ax[0].imshow(im, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Filter Gausssian')

ax[1].imshow(im, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Bin')

ax[2].imshow(im, cmap=plt.cm.gray)
ax[2].plot(coordinates_2[:, 1], coordinates_2[:, 0], 'r.')

for i in range(coordinates_2.shape[0]):
    ax[2].text(coordinates_2[i, 1], coordinates_2[i, 0], '%d'%i, color = 'white')
    
ax[2].axis('off')
ax[2].set_title('Peak local max')

fig.tight_layout()
plt.show()

#%%
plt.figure()
plt.imshow(im, cmap=plt.cm.gray)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
for i in range(coordinates.shape[0]):
    plt.text(coordinates[i, 1], coordinates[i, 0], '%d'%i, color = 'white')
plt.axis('off')
plt.show()

#%%

#ideal = ideal_grid(10, 12, 3, 4)

grid = make_grid(coordinates, pixel_size = 0.059)
Npart = grid.shape[1]
print(Npart)

list_NP_column = [10, 8, 10, 9, 10, 8, 10, 10, 9, 10, 10, 9]

for j in range(len(list_NP_column)):

    for i in range(list_NP_column[j]-1):
        
        diff_x = grid[1, i+1] - grid[1, i]
        diff_y = grid[0, i+1] - grid[0, i]
        
        diff = round(np.sqrt(diff_x**2 + diff_y**2),4)
        
    print(j, np.mean(diff))
 
print(np.mean(diff))

list_NP_row = [11, 11, 11, 12, 11, 12, 11, 11, 12, 11]

for j in range(len(list_NP_column)):
    
    a = 0
    
    for i in range(list_NP_column[j]-1):
        
        b = list_NP_column[j] - 1
        diff_x = grid[1, 1 + b] - grid[1, a]
        diff_y = grid[0, 1 + b] - grid[0, a]
        
        a = list_NP_column[j]

        print(diff_x, diff_y)
