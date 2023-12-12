
import os
import re
import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage

from skimage import io
from skimage.color import rgb2gray
from skimage import filters, measure, morphology

import skimage.filters
import skimage.viewer

def open_image_RGB(folder, file):
    
    name_file = os.path.join(folder, file)
    image = io.imread(name_file)
    image = rgb2gray(image)
    
    return image

def center_of_mass(image, n_filter):
    
    new_image = (image- np.min(image) ) / (np.max(image) -  np.min(image))
    
    #new_image = filters.gaussian(image, sigma=0)

    for i in range(len(new_image)):
        for j in range(len(new_image)):
            if new_image[i, j] <= n_filter:
                    new_image[i, j] = 0
                    com = ndimage.measurements.center_of_mass(new_image)
                    com = np.round(com, 2)
    
    return com

def crop_image(image, large, size_ROI):
    
    crop_image = image[200:600, 300:800]
    
  #  plt.figure()
    
  #  plt.imshow(crop_image)
    
   # x = np.where(crop_image == np.max(crop_image))[0][0]
  #  y = np.where(crop_image == np.max(crop_image))[1][0]
    
   # zoom_image = crop_image[x-size_ROI*2:x+size_ROI*2, y-size_ROI*2:y+size_ROI*2]
    
    xcm, ycm = ndimage.measurements.center_of_mass(crop_image) #center_of_mass(crop_image, 0.7)
    xcm = int(xcm)
    ycm = int(ycm)
    
    print('centro de masa', xcm, ycm)
    
    size_ROI = int(size_ROI/2)
    
    crop_image = crop_image[xcm-size_ROI:xcm+size_ROI, ycm-size_ROI:ycm+size_ROI]
    
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
        
    #plt.close('all')
        
    return None


def bin_image(image, filter_value):
    
    image = filters.gaussian(image, sigma=0)
    
    bin_image = np.zeros((image.shape[0], image.shape[1]))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            bin_image[i, j] = image[i, j]
            
            if image[i, j] < filter_value:

                bin_image[i, j] = 0
                
    return bin_image
    
prefix_folder = '/Ubuntu_archivos/Printing/Nanostars/Data_Iani/190919-Ianina Violi/' 

local_folder  = '13-08 G3 640 3.2 mW/no_crop'
#local_folder  = '18-09 G1 640 2.3mW/no_crop'

pre_base_folder = os.path.join(prefix_folder, local_folder)

save_folder = os.path.join(pre_base_folder,'crop_image3') 
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
crop_all_images(pre_base_folder, 300, save_folder)

base_folder = os.path.join(pre_base_folder, 'crop_image3')
    
list_of_files = os.listdir(base_folder)
list_of_files = [f for f in list_of_files if re.search('tif',f)]
list_of_files.sort()

filter_value = 0.4 #50

list_metric = []

for file in list_of_files:
    
    name_file = os.path.join(base_folder, file)
    img1 = io.imread(name_file)
    
    img = bin_image(img1, filter_value)
    
    # Binary image, post-process the binary mask and compute labels
    threshold = filters.threshold_otsu(img)
    mask0 = img > threshold
    mask1 = morphology.remove_small_objects(mask0, 50)
    mask = morphology.remove_small_holes(mask1, 50)
    
    labels = measure.label(mask)
    props = measure.regionprops(labels, img)
    properties = ['area', 'eccentricity', 'perimeter']#, 'intensity_mean']
    
    area = getattr(props[0], properties[0])
    perimeter = getattr(props[0], properties[2])
        
    metric = 4*np.pi*area/(perimeter**2)
    
    list_metric.append(metric)
    
    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    label_i = props[0].label
    contour = measure.find_contours(labels == label_i, 0.5)[0]
    y, x = contour.T

    for prop_name in properties:
        
        print(prop_name , getattr(props[0], prop_name))
        
    fig_bool = True
        
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
            
        ax[3].imshow(mask, cmap=plt.cm.gray)
        ax[3].plot(x, y, 'r--') 
        ax[3].axis('off')
        ax[3].set_title('Contour')
    
plt.figure()
for e in range(len(list_of_files)):
    plt.plot(e, list_metric[e], 'o', label = list_of_files[e])
plt.legend()
        