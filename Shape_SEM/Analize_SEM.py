#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:33:44 2020

@author: luciana
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re

from skimage import io
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.filters.rank import gradient
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon

import skimage.filters
import skimage.viewer

import radialProfile
from scipy import ndimage
from scipy import stats


def open_image(folder, file):
    
    name_file = os.path.join(folder, file)
    image = io.imread(name_file)
    
    return image

def open_image_RGB(folder, file):
    
    image_RGB = open_image(folder, file)
    
   # image = rgb2gray(image_RGB)
    
    image = skimage.color.rgb2gray(image_RGB)
    
    return image

def crop_image(image, large, size_ROI):
    
    crop_image = image[0:large, :]
    
    x = np.where(crop_image == np.max(crop_image))[0][0]
    y = np.where(crop_image == np.max(crop_image))[1][0]
    
    size_ROI = int(size_ROI/2)
    
    crop_image = crop_image[x-size_ROI:x+size_ROI, y-size_ROI:y+size_ROI]
    
    return crop_image

def crop_all_images(base_folder, size_ROI, save_folder):

    list_of_files = os.listdir(base_folder)
    list_of_files = [f for f in list_of_files if re.search('tif',f)]
    list_of_files.sort()
    
    for file in list_of_files:
     
        image = open_image_RGB(base_folder, file)
        image = crop_image(image, 700, size_ROI)
        
      #  plt.imshow(image, cmap = 'gray')
        name_image = os.path.join(save_folder, file)
        #plt.savefig(name_image)
        
        io.imsave(fname=name_image, arr=skimage.img_as_ubyte(image))
        
    plt.close('all')
        
    return None

def bin_image(image, sigma):
    
    blur = skimage.filters.gaussian(image, sigma=sigma)
    
    return blur

def bin_image_2(image, filter_value):
    
    bin_image = np.zeros((image.shape[0], image.shape[1]))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            bin_image[i, j] = image[i, j]
            
            if image[i, j] < filter_value:

                bin_image[i, j] = 0
                
    return bin_image
    

def fourier(img):
    
    f = np.fft.fft2(img)
    fshift  = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    return magnitude_spectrum

def psd_1D(psd_2D):
    
   #  azimuthally averaged 1D power spectrum 
    
    psd1D = radialProfile.azimuthalAverage(psd_2D)
    
    return psd1D

def process_image(base_folder, file, sigma, filter_value, tolerance):
        
    image1 = open_image_RGB(base_folder, file)
    
    #image2 = bin_image(image1, sigma)
    #image3 = bin_image_2(image2, filter_value)
  
    selection_element = disk(7) # matrix of n pixels with a disk shape
    image_sharpness = gradient(image1, selection_element)
    
    imageS2 = bin_image(image_sharpness, sigma = sigma)
    imageS3 = bin_image_2(imageS2, filter_value)
    
    plot_bool = True
    
    if plot_bool:

        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
        
        ax = axes.ravel()
    
        ax[0].imshow(image1, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Image')
        
        ax[1].imshow(imageS2, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Sharpness Filter')
        
        ax[2].imshow(imageS3, cmap=plt.cm.gray)
        for contour in find_contours(imageS3, 0):
            coords = approximate_polygon(contour, tolerance=tolerance)
            ax[2].plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
        ax[2].axis('off')
        ax[2].set_title('BIN, Sharpness Bin')
        
        fig.tight_layout()
        plt.show()
    
  #  psd2D = fourier(imageS3)
    
  #  plt.figure()
  #  plt.imshow(psd2D)
    
  #  angF, psd1D = GetRPSD(psd2D, dTheta = 30, rMin = 0, rMax = 100)
    
    return #angF, psd1D

#=============================================================================
# Get PSD 1D (total power spectrum by angular bin)
#=============================================================================
def GetRPSD(psd2D, dTheta, rMin, rMax):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2
    
    # note that displaying PSD as image inverts Y axis
    # create an array of integer angular slices of dTheta
    Y, X  = np.ogrid[0:h, 0:w]
    theta = np.rad2deg(np.arctan2(-(Y-hc), (X-wc)))
    theta = np.mod(theta + dTheta/2 + 360, 360)
    theta = dTheta * (theta//dTheta)
    theta = theta.astype(np.int)
    
    # mask below rMin and above rMax by setting to -100
    R     = np.hypot(-(Y-hc), (X-wc))
    mask  = np.logical_and(R > rMin, R < rMax)
    theta = theta + 100
    theta = np.multiply(mask, theta)
    theta = theta - 100
    
    # SUM all psd2D pixels with label 'theta' for 0<=theta❤60 between rMin and rMax
    angF  = np.arange(0, 360, int(dTheta))
    psd1D = ndimage.sum(psd2D, theta, index=angF)
    
    # normalize each sector to the total sector power
    pwrTotal = np.sum(psd1D)
    psd1D    = psd1D/pwrTotal
    
    return angF, psd1D
#=============================================================================
    
#%%

if __name__ == '__main__':
    
   # plt.close('all')
    
    prefix_folder = '/Ubuntu_archivos/Printing/Nanostars/Data_Iani/190919-Ianina Violi/' 
    
    local_folder  = '13-08 G3 640 3.2 mW/no_crop'
  #  local_folder = '18-09 G3 640 1.7mW/no_crop'
   # local_folder = '18-09 G2 640 1.6mW/Col 2/no_crop'
    
  #  local_folder = '18-09 G1 640 2.3mW/no_crop'
    
    pre_base_folder = os.path.join(prefix_folder, local_folder)
    
    save_folder = os.path.join(pre_base_folder,'crop_image') 
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    crop_all_images(pre_base_folder, 400, save_folder)
    
    base_folder = os.path.join(pre_base_folder, 'crop_image')
    
    list_of_files = os.listdir(base_folder)
    list_of_files = [f for f in list_of_files if re.search('tif',f)]
    list_of_files.sort()
    
    sigma = 0
    filter_value = 0.78  #7#.5
    tolerance = 1
    
   # plt.figure()
    
    for file in list_of_files:
        
        process_image(base_folder, file, sigma, filter_value, tolerance)
    
     #   angF, psf1D = process_image(base_folder, file, sigma, filter_value)
        
      #  name_file = file.split('.tif')[0]
        
      #  plt.plot(angF, psf1D, 'o', label = name_file)
        
      #  a = np.max(psf1D)
      #  b = np.min(psf1D)
        
      #  A = 2*(a - b) #anisotropy
        
      #  print(name_file, A)
        
  #  plt.legend()
        
  #  plt.xlabel("Spatial Frequency")
  #  plt.ylabel("Power Spectrum")
    
   # plt.xlim(-10, 250)
   # plt.ylim(0.02, 0.03)
 
    
#%% prueba, circulo 
   
def image_gauss(tamaño, x_NP, y_NP):

    A = 1
    b = 20 #pixel
    
    i = 0
    j = 0
    
    I_NP = np.zeros((tamaño, tamaño))
    
    for i  in range(tamaño):
        for j  in range(tamaño):
            I_NP[i, j] = A*np.exp(   -( (i-x_NP)**2 + (j-y_NP)**2 )/(2*b)**2  )
            
    I_NP = I_NP + 0.1*np.random.normal(0, 1, I_NP.shape)
            
    return I_NP

def image_circle(tamaño, x_NP, y_NP):
    
    A = 0.05

    i = 0
    j = 0

    I_NP = np.zeros((tamaño, tamaño))

    for i  in range(tamaño):
        for j  in range(tamaño):
            I_NP[i, j] = A*np.sqrt( (i-x_NP)**2 + (j-y_NP)**2 )
            
    I_NP = stats.norm.pdf(I_NP)
    
    I_NP = I_NP + 0.01*np.random.normal(0, 1, I_NP.shape)
    
    return I_NP

def image_circle_full(tamaño, x_NP, y_NP, ratio):
    
    A = 1

    i = 0
    j = 0

    I_NP = np.zeros((tamaño, tamaño))

    for i  in range(tamaño):
        for j  in range(tamaño):
            if np.sqrt( (i-x_NP)**2 + (j-y_NP)**2 ) < ratio:
                I_NP[i, j] = A
    
    I_NP = I_NP + 0.1*np.random.normal(0, 1, I_NP.shape)
    
    return I_NP

def image_square_full(tamaño, x_NP, y_NP, large):
    
    A = 1

    i = 0
    j = 0

    I_NP = np.zeros((tamaño, tamaño))

    for i  in range(tamaño):
        for j  in range(tamaño):
            if np.sqrt( (i-x_NP)**2 ) < large and np.sqrt( (j-y_NP)**2 ) < large:
                I_NP[i, j] = A
    
    I_NP = I_NP  + 0.1*np.random.normal(0, 1, I_NP.shape)
    
    return I_NP

#image = image_circle(400, 200, 200)
#image = image_gauss(400, 200, 200)
size = 400
image = image_square_full(size, size/2, size/2, 50)
image2 = image_circle_full(size, size/2, size/2, 50)

image3 = image_gauss(size, size/2, size/2)

psd2D = fourier(image)
psd2D2 = fourier(image2)
psd2D3 = fourier(image3)

plt.figure()
plt.imshow(image)
plt.figure()
plt.imshow(psd2D)

plt.figure()
plt.imshow(image2)
plt.figure()
plt.imshow(psd2D2)

plt.figure()
plt.imshow(image3)
plt.figure()
plt.imshow(psd2D3)

angF, psd1D = GetRPSD(psd2D, dTheta = 30, rMin = 10, rMax = 100)
angF2, psd1D2 = GetRPSD(psd2D2, dTheta = 30, rMin = 10, rMax = 100)
angF3, psd1D3 = GetRPSD(psd2D3, dTheta = 30, rMin = 10, rMax = 100)

a = np.max(psd1D)
b = np.min(psd1D)
A = 2*(a-b) #anisotropy
print('anisotropy square', A)

a2 = np.max(psd1D2)
b2 = np.min(psd1D2)
A2 = 2*(a2-b2) #anisotropy
print('anisotropy circle', A2)

a3 = np.max(psd1D3)
b3 = np.min(psd1D3)
A3 = 2*(a3-b3) #anisotropy
print('anisotropy square+circle', A3)

plt.figure()
plt.plot(angF, psd1D, 'o', label = 'square')
plt.plot(angF2, psd1D2, 'o', label = 'circle')
plt.plot(angF3, psd1D3, 'o', label = 'gauss')
plt.legend()

radial = radialProfile.azimuthalAverage(psd2D, center = None)

radial2 = radialProfile.azimuthalAverage(psd2D2, center = None)

radial3 = radialProfile.azimuthalAverage(psd2D3, center = None)

plt.figure()
plt.plot(radial , 'o', label = 'square')
plt.plot(radial2 , 'o', label = 'circle')
plt.plot(radial3 , 'o', label = 'gauss')
plt.legend()
