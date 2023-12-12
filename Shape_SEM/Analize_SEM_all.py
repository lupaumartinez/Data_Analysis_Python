
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
    
    crop_image = image[300:700, 600:1000]
    
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
        image = crop_image(image, 500, size_ROI)
        
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

def curve_gauss(image):
    
    profile_x = np.mean(image, axis = 1)
    profile_y = np.mean(image, axis = 0)
    plt.figure()
    plt.plot(profile_x, 'r')
    plt.plot(profile_y, 'b')
    plt.show()
    
    return profile_x, profile_y
    
prefix_folder = 'C:/Users/lupau/OneDrive/Documentos/Paper_Growth_Lenovo_SEM/Muestras 1'

pixel_size = 0.74 #pixel/nm col 1

local_folder  = 'col1' 
name_col = local_folder.split('col')[-1]
col = int(name_col)
name_col = 'Col_%03d'%col

pre_base_folder = os.path.join(prefix_folder, local_folder)

save_folder = os.path.join(prefix_folder,'analize_sem', name_col) 
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
crop_all_images(pre_base_folder, 300, save_folder)
print('all image crop')
    
list_of_files = os.listdir(save_folder)
list_of_files = [f for f in list_of_files if re.search('tif',f) and not re.search('contour',f)]
list_of_files.sort()

filter_value = 0.18 #50

list_perimeter = []
list_area = []
list_metric = []
list_eccentricity = []
list_NP = []

for file in list_of_files:
    
    name = file.split(local_folder)[-1]
    name = name.split('_')[0]
    name = name.split('np')[-1]
    NP = int(name)
    
    list_NP.append(NP)
    
    name_file = os.path.join(save_folder, file)
    
    img1 = io.imread(name_file)
    
    x, y = curve_gauss(img1)
    
    img = bin_image(img1, filter_value)
    
  #  plt.imshow(img, cmap=plt.cm.gray)
    
    # Binary image, post-process the binary mask and compute labels
    threshold = filters.threshold_otsu(img)
    mask0 = img > threshold
    mask1 = morphology.remove_small_objects(mask0, 100)
    mask = mask1 #morphology.remove_small_holes(mask1, 50)
    
    labels = measure.label(mask)
    props = measure.regionprops(labels, img)
    print(props)
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
        
    fig_bool = False
        
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
        
        name = 'contour_' + file + '.png'
        figure_name = os.path.join(save_folder, name)
        plt.savefig(figure_name)
        
    
    for prop_name in properties:
        
        print(prop_name , getattr(props[0], prop_name))
    
    
ratio_area = np.sqrt(np.array(list_area)/(np.pi))
#ratio = 2*(np.array(list_area)/np.array(list_perimeter))
diameter = 2*ratio_area/pixel_size

plt.figure()
plt.plot(list_NP, diameter, '*')
plt.legend()
plt.show()

data = np.array([list_NP, diameter, list_area, list_perimeter, list_eccentricity]).T
header = 'NP, diameter (nm), area, perimeter, eccentricity'
name = os.path.join(prefix_folder,'analize_sem', 'diameter_%s.txt'%name_col)
np.savetxt(name, data, header = header)
        