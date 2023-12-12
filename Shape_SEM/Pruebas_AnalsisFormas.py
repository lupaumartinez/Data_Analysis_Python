# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 20:33:44 2022

@author: lupau
"""


import os
import re
import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage
from scipy import stats

from skimage import io
from skimage.color import rgb2gray
from skimage import filters, measure, morphology

import skimage.filters
import skimage.viewer


def image_gauss(tamaño, x_NP, y_NP):

    A = 1
    b = 20 #pixel
    
    i = 0
    j = 0
    
    I_NP = np.zeros((tamaño, tamaño))
    
    for i  in range(tamaño):
        for j  in range(tamaño):
            I_NP[i, j] = A*np.exp(   -( (i-x_NP)**2 + (j-y_NP)**2 )/(2*b)**2  )
            
    I_NP = I_NP + 0.1*(1-np.random.normal(0, 1, I_NP.shape))
            
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
            if np.sqrt( (i-x_NP)**2 + (j-y_NP)**2 ) <= ratio:
                I_NP[i, j] = A
    
    I_NP = I_NP #+ 0.1*(1-np.random.normal(0, 1, I_NP.shape))
    
    return I_NP

def image_square_full(tamaño, x_NP, y_NP, large):
    
    A = 1

    i = 0
    j = 0

    I_NP = np.zeros((tamaño, tamaño))

    for i  in range(tamaño):
        for j  in range(tamaño):
            if np.sqrt( (i-x_NP)**2 ) <= large and np.sqrt( (j-y_NP)**2 ) <= large:
                I_NP[i, j] = A
    
    I_NP = I_NP  #+ 0.1*(1-np.random.normal(0, 1, I_NP.shape))
    
    return I_NP

def bin_image(image, filter_value):
    
    image = filters.gaussian(image, sigma=0)
    
    bin_image = np.zeros((image.shape[0], image.shape[1]))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            bin_image[i, j] = image[i, j]
            
            if image[i, j] < filter_value:

                bin_image[i, j] = 0
                
    return bin_image

size = 400
image1 = image_circle_full(size, size/2, size/2, 50)
image2 = image_circle_full(size, size/2, size/2, 100)

large = 100
image3 = image_square_full(size, size/2, size/2, large)
#image4 = image_gauss(size, size/2, size/2)

list_area_theoric = [np.pi*50**2, np.pi*100**2, large**2]
list_perimeter_theoric = [2*np.pi*50, 2*np.pi*100, 4*large]

list_of_files = [image1, image2, image3]

filter_value = 0.1 #50

list_perimeter = []
list_area = []
list_metric = []
list_eccentricity = []

fig_bool = True

for file in list_of_files:
    
    img1 = file
    
    img = bin_image(img1, filter_value)
    
  #  plt.imshow(img, cmap=plt.cm.gray)
    
    # Binary image, post-process the binary mask and compute labels
    threshold = filters.threshold_otsu(img)
    mask0 = img > threshold
    mask1 = morphology.remove_small_objects(mask0, 100)
    mask = morphology.remove_small_holes(mask1, 50)
    
    labels = measure.label(mask)
    props = measure.regionprops(labels)
  #  print(props)
    properties = ['area', 'eccentricity', 'perimeter']#, 'intensity_mean']
    
    area = getattr(props[0], properties[0])
    perimeter = getattr(props[0], properties[2])
    eccentricity = getattr(props[0], properties[1])
    
    list_area.append(area)
    list_perimeter.append(perimeter)
    list_eccentricity.append(eccentricity)
        
    metric = 4*np.pi*area/(perimeter**2)
    
    list_metric.append(metric)
    
    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    label_i = props[0].label
    contour = measure.find_contours(labels == label_i, 0.5)[0]
    y, x = contour.T
        
        
    if fig_bool:
            
        fig, axes = plt.subplots(1, 4, figsize=(8, 4), sharex=True, sharey=True)
        
        fig.suptitle(file)
        
        ax = axes.ravel()
        
        ax[0].imshow(img1, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Image')
        
        ax[1].imshow(img, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Bin Image')
            
        ax[2].imshow(mask1, cmap=plt.cm.gray)
        ax[2].axis('off')
        ax[2].set_title('Mask')
            
        ax[3].imshow(labels, cmap=plt.cm.gray)
      #  ax[3].plot(x, y, 'r--') 
        ax[3].axis('off')
        ax[3].set_title('Contour')
        
    
    for prop_name in properties:
        
        print(prop_name , getattr(props[0], prop_name))
    
#circle    
#ratio_area = np.sqrt(np.array(list_area)/(np.pi))
#ratio_perimeter = 2*(np.array(list_area)/np.array(list_perimeter))
#diameter = 2*ratio_area

plt.figure()
plt.plot(list_area, '*')
plt.plot(list_area_theoric, 'o')
plt.legend()
plt.show()

plt.figure()
plt.plot(list_perimeter, '*')
plt.plot(list_perimeter_theoric, 'o')
plt.legend()
plt.show()