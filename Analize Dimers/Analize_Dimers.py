# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:37:37 2019

@author: Luciana
"""

import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from PSF import center_of_mass, center_of_gauss2D, find_two_centers, two_centers_of_gauss2D

ScanImage = [1, -1, 2]
MethodCenter = ['CM', 'Gauss']

def open_data(folder, type_scan):
    
    list_of_files = []
    
    for files in os.listdir(folder):
        if files.startswith(type_scan):
            list_of_files.append(files)

   # list_of_files = os.listdir(folder)
    list_of_files.sort()
    
    first_scan = io.imread(os.path.join(folder, list_of_files[0]))
    stack = np.zeros((len(list_of_files), first_scan.shape[0],first_scan.shape[1]))
            
    for n in range(0,len(list_of_files)):
        #print('archivo', list_of_files[n])
        stack[n,:,:]= io.imread(os.path.join(folder, list_of_files[n]))
        
    return stack

def plot_scan(stack_scan):
    for i in range(stack_scan.shape[0]):
        plt.figure()
        plt.imshow(stack_scan[i,:,:])
        
def plot_scan_subtract(stack_scan_dimer, stack_scan_preescan):
    
    image_type_minimum = -1
    image_type_maximum = 1
    method_center_option = 'Gauss'
    
    Npixel = stack_scan_dimer.shape[1]
    
    NP_list = []
    dx = []
    dy = []
    xdrift_pre = []
    ydrift_pre = []
    factor_scale = 2000/Npixel
    
    for i in range(stack_scan_dimer.shape[0]):
        
        scan_subtract = stack_scan_dimer[i,:,:] - stack_scan_preescan[i,:,:] 
        #image_dimer = (stack_scan_dimer[i,:,:]-np.min(stack_scan_dimer[i,:,:]))/(np.max(stack_scan_dimer[i,:,:])-np.min(stack_scan_dimer[i,:,:]))
        #image_preescan = (stack_scan_preescan[i,:,:]-np.min(stack_scan_preescan[i,:,:]))/(np.max(stack_scan_preescan[i,:,:])-np.min(stack_scan_preescan[i,:,:]))
        #scan_subtract = image_dimer - image_preescan
        
        center_mass_pree = CMmeasure(stack_scan_preescan[i,:,:], image_type_minimum, method_center_option)
        center_mass_subtract = CMmeasure(scan_subtract , image_type_maximum, method_center_option)
        
        center_mass_2, center_mass_1 = two_CMmeasure(stack_scan_dimer[i,:,:], image_type_maximum)
        cm_subtract = center_mass_1 - center_mass_2
        
        cm_drift = center_mass_2 - center_mass_pree
        
        NP_list.append(i+1)
        dx.append(cm_subtract[0]*factor_scale)
        dy.append(cm_subtract[1]*factor_scale)
        xdrift_pre.append(cm_drift[0]*factor_scale)
        ydrift_pre.append(cm_drift[1]*factor_scale)
        
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
        ax1.imshow(stack_scan_preescan[i,:,:] )
        ax1.scatter(*center_mass_pree[::-1])
        
        ax2.imshow(stack_scan_dimer[i,:,:])
        ax2.scatter(*center_mass_1[::-1])
        ax2.scatter(*center_mass_2[::-1])
        
        ax3.imshow(scan_subtract)
        ax3.scatter(*center_mass_subtract[::-1])
        
    return NP_list, dx, dy, xdrift_pre, ydrift_pre
        

def norm_image(image, image_type):
    
    Z = image
    
    Zmin = np.min(Z)
    Zmax = np.max(Z)
    Zn = (Z-Zmin)/(Zmax- Zmin)  #lleva a ceros y unos
    
    if image_type == ScanImage[0]: #caso NPs maximos
        Zn = Zn #queda NP = 1, bkg = 0
        print('go to CM de un maximo')  
                
    elif image_type == ScanImage[1]: #caso NPs minimos
        print('go to CM de un minimo')  
        Zn = Zn - 1 #queda NP = -1, bkg = 0
        Zn = np.abs(Zn)  #queda NP = 1, bkg = 0
            
    elif image_type == ScanImage[2]:  #para caso de 2 NP, una minima y la otra maxima
        bkg = np.mean(Z)
        Zn = (Z-bkg)/(Zmax- bkg)  #lleva a ceros el bkg, 1 NP maxima, <0 NP minima
        Zn = np.abs(Zn)  #lleva a ceros el bkg, 1 NP maxima, >0 NP minima            
            
    return Zn

def filter_image(image, image_bin, n_filter):
    
    for i in range(len(image[:,1])):
        for j in range (len(image[1,:])):
            if image_bin[i,j] < n_filter:
                image_bin[i,j] = 0
                
    return image_bin

def CMmeasure(image, image_type, method_center_option):

    Z = image #ida y vuelta
    
    Zn = norm_image(Z, image_type)  #normalizo la imagen para que quede con 1
    
    n_filter = 0.6  #numero para filtrar la imagem
    Zfilter = filter_image(Z, Zn, n_filter)
    
    if method_center_option == MethodCenter[0]:
        xo, yo = center_of_mass(Zfilter)  #ycm, xcm = ndimage.measurements.center_of_mass(Zn)

    elif method_center_option == MethodCenter[1]:
    
        x_cm, y_cm = center_of_mass(Zfilter)
        xo, yo = center_of_gauss2D(Zn, x_cm, y_cm)
        
    return np.array([yo, xo])
        
def two_CMmeasure(image, image_type):
    
    Z = image #ida y vuelta
    
    Zn = norm_image(Z, image_type)  #normalizo la imagen para que quede con 1
    
    n_filter = 0.3 #numero para filtrar la imagem
    Zfilter = filter_image(Z, Zn, n_filter)
        
    xo_1, yo_1, xo_2, yo_2 = find_two_centers(Zfilter)
    xo, yo, xo2, yo2 = two_centers_of_gauss2D(Zn, xo_1, yo_1, xo_2, yo_2)
    
    cgauss = [np.array([yo, xo]), np.array([yo2, xo2])]
    
    d = [(yo**2 + xo**2), (yo2**2 + xo2**2)]
    i = np.argmin(d)
    imax = np.argmax(d)
    
    cgauss_1 = cgauss[i]
    cgauss_2 = cgauss[imax]
    
    return cgauss_1, cgauss_2
 
    
def find_center_of_mass(stack_scan_2, stack_scan_1):
    
    image_type_maximum = 1
    image_type_minimum = -1
    method_center_option = 'Gauss'
    
    Npixel = stack_scan_2.shape[1]
    
    NP_list = []
    dx = []
    dy = []
    factor_scale = 2000/Npixel #nm/pixel scan
    
    for i in range(stack_scan_2.shape[0]):
        
        image1 = stack_scan_1[i,:,:]
        image2 = stack_scan_2[i,:,:] - image1
        
        center_mass_2 =  CMmeasure(image2, image_type_maximum, method_center_option)
        center_mass_1 = CMmeasure(image1, image_type_minimum, method_center_option)
        cm_subtract = center_mass_2 - center_mass_1
        
        NP_list.append(i+1)
        dx.append(cm_subtract[0]*factor_scale)
        dy.append(cm_subtract[1]*factor_scale)
        
    return NP_list, dx, dy

def plot_scan_target(stack_scan):
    
    image_type_maximum = 1
    image_type_minimum = -1
    method_center_option = 'Gauss'
    
    Npixel = stack_scan.shape[1]
    center_image = np.array([int(stack_scan.shape[1]/2), int(stack_scan.shape[2]/2)])
    
    target_x = []
    target_y = []
    factor_scale = 2000/Npixel #nm/pixel scan
    
    for i in range(stack_scan.shape[0]):
        
        image = stack_scan[i,:,:]
        
        center_mass = CMmeasure(image, image_type_minimum, method_center_option)
        
        #el 08/11:
        target_x.append(-factor_scale*(center_mass[0] - center_image[0] + 1/2)) #agregue 1/2 por el tema de los bordes px
        target_y.append(-factor_scale*(center_mass[1] - center_image[1] + 1/2)) #agregue 1/2 por el tema de los bordes px
        
       # target_x.append(-factor_scale*(center_mass[0] - center_image[0] )) 
       # target_y.append(-factor_scale*(center_mass[1] - center_image[1] )) 
        
        f, ax1 = plt.subplots(1, 1)
    
        ax1.imshow(image)
        ax1.scatter(*center_mass[::-1])
        
    return target_x, target_y


def plot_scan_cm(stack_scan):

    Npixel = stack_scan.shape[1]
    
    image_type_maximum = 1
    image_type_minimum = -1
    method_center_option = 'Gauss'
    
    target_x = []
    target_y = []
    factor_scale = 2000/Npixel #um/pixel scan
    
    for i in range(stack_scan.shape[0]):
        
        image = stack_scan[i,:,:]
        
        center_mass = CMmeasure(image, image_type_minimum, method_center_option)
        
        target_x.append(-factor_scale*(center_mass[0]))
        target_y.append(-factor_scale*(center_mass[1]))
        
       # f, ax1 = plt.subplots(1, 1)
    
       # ax1.imshow(image)
       # ax1.scatter(*center_mass[::-1])
        
    return target_x, target_y


def data_hist(x):
    
    mu= round(np.mean(x), 2) # media
    sigma= round(np.std(x, ddof=1), 2) #desviación estándar
    N=len(x) # número de cuentas
    std_err = round(sigma / N,2) # error estándar
    
    # muestro estos resultados
    print( 'media: ', mu)
    print( 'desviacion estandar: ', sigma)
    print( 'total de cuentas: ', N)
    print( 'error estandar: ', std_err)

    txt = '%s + %s'%(mu, sigma)
    
    return mu, sigma, txt
        
#%%
        
plt.close('all')

prefix_folder = r'C:\Users\lupau\OneDrive\Documentos\2022-11-17 Dimeros AuNPz80nm y Nanostars P2R20'
local_folder  = '20221117-172020_Dimers_4x1_3.0umx3.0um_80nmR20_105nm'

dimer_DX = 500 #nm
dimer_DY = 0

direction = os.path.join(prefix_folder, local_folder)
print(direction)

save_folder = os.path.join(direction, 'figures_dimers')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#%%

direction_centerscan = direction #+ "/" + "Center_Scan"
direction_preescan = os.path.join(direction, "Pree_Scan")
direction_dimerscan = os.path.join(direction, "Dimer_Scan")

list_type_scan = ['gone_NPscan', 'back_NPscan', 'NPscan']

#type_scan = 'gone_NPscan'
#type_scan = 'back_NPscan'
#type_scan = 'NPscan'

for type_scan in list_type_scan:
    
    print('Analizando el scan:',type_scan)
    
    stack_centerscan = open_data(direction_centerscan, type_scan)
    stack_preescan = open_data(direction_preescan, type_scan)
    stack_dimerscan = open_data(direction_dimerscan, type_scan)
    
    #print('Analizando el scan:', type_scan, 'Figuras center scan')
    #plot_scan(stack_centerscan)
    
    print('Analizando el scan:',type_scan, 'Centro de masa de confocal preescan')
    
    target_x, target_y = plot_scan_target(stack_preescan)
    mu_x, sigma_x, txt_x = data_hist(target_x)
    mu_y, sigma_y, txt_y = data_hist(target_y)
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(target_x , bins=23, density = True, rwidth=0.9,color='C1')
    ax1.set_xlabel('DX [nm]')
    ax1.set_xlim(-50, 800)
    ax2.hist(target_y  , bins=23, density = True, rwidth=0.9,color='C2')
    ax2.set_xlabel('DY [nm]')
    ax2.set_xlim(-50, 800)
    plt.savefig(os.path.join(save_folder, 'hist_target_pree_scan_%s.png'%(type_scan)), dpi = 400)
    plt.close()
    
    print('Analizando el scan:',type_scan, 'Centro de masa de la resta entre la confocal de dimero y preescan')
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(target_x , 'o')
    ax1.set_ylabel('DX [nm]')
    ax2.plot(target_y , 'o')
    ax2.set_ylabel('DY [nm]')
    ax1.set_ylim(-50, 800)
    ax1.set_xlim(-1, 11)
    ax2.set_ylim(-50, 800)
    ax2.set_xlim(-1, 11)
    ax1.text(8, mu_x, txt_x, fontsize = 'xx-small')
    ax2.text(8, mu_y, txt_y, fontsize = 'xx-small')
    plt.savefig(os.path.join(save_folder, 'target_pree_scan_%s.png'%(type_scan)), dpi = 400)
    plt.close()
    
    NP_list2, dx2, dy2, xdrift_pre, ydrift_pre = plot_scan_subtract(stack_dimerscan, stack_preescan)
    
    NP_list, dx, dy = find_center_of_mass(stack_dimerscan, stack_preescan)
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(dx , bins=23, density = True, rwidth=0.9,color='C1')
    ax1.set_xlabel('DX [nm]')
    ax1.set_xlim(-50, 800)
    ax1.axvline(dimer_DX, color = 'r', linestyle = '--')
    ax2.hist(dy , bins=23,  density = True, rwidth=0.9,color='C2')
    ax2.set_xlabel('DY [nm]')
    ax2.set_xlim(-50, 800)
    ax2.axvline(dimer_DY, color = 'r', linestyle = '--')
    plt.savefig(os.path.join(save_folder, 'hist_dimer_scan_%s.png'%(type_scan)), dpi = 400)
    plt.close()
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(dx , 'o')
    ax1.plot(dx2 , '*')
    ax1.set_xlabel('DX [nm]')
   # ax1.set_ylim(-50, 800)
   # ax1.set_xlim(-1, 11)
    ax1.axhline(dimer_DX, color = 'r', linestyle = '--')
    
    ax2.plot(dy , 'o')
    ax2.plot(dy2 , '*')
    ax2.set_xlabel('DY [nm]')
   # ax2.set_ylim(-50, 800)
   # ax2.set_xlim(-1, 11)
    ax2.axhline(dimer_DY, color = 'r', linestyle = '--')
    plt.savefig(os.path.join(save_folder, 'dimer_scan_%s.png'%(type_scan)), dpi = 400)
    plt.close()
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(xdrift_pre , 'o')
    ax1.set_xlabel('Drift X Pree [nm]')
   # ax1.set_ylim(-50, 800)
   # ax1.set_xlim(-1, 11)
    
    ax2.plot(ydrift_pre, 'o')
    ax2.set_xlabel('Drift Y Pree [nm]')
   # ax2.set_ylim(-50, 800)
   # ax2.set_xlim(-1, 11)
    plt.savefig(os.path.join(save_folder, 'drift_pree_scan_%s.png'%(type_scan)), dpi = 400)
    plt.close()
    
#%%

    
f, (ax1, ax2) = plt.subplots(1, 2)

for type_scan in list_type_scan:
    
    print('Analizando el scan:',type_scan)    
    
    print('Analizando el scan:',type_scan, 'Centro de masa de confocales')
    
    stack_centerscan = open_data(direction_centerscan, type_scan)
    stack_preescan = open_data(direction_preescan, type_scan)
    stack_dimerscan = open_data(direction_dimerscan, type_scan)
    
    cm_x, cm_y = plot_scan_cm(stack_centerscan)
    
    if type_scan == 'back_NPscan':
        cm_x_back = cm_x
        cm_y_back = cm_y
               
    if type_scan == 'gone_NPscan': 
        cm_x_gone = cm_x
        cm_y_gone = cm_y
    
    ax1.plot(cm_x, '-o', label = '%s'%(type_scan))
    ax1.set_ylabel('CM X [nm]')
    ax1.legend()
    ax2.plot(cm_y, '-o', label = '%s'%(type_scan))
    ax2.set_ylabel('CM Y [nm]')
    ax2.legend()
    
cm_x_mean = (np.array(cm_x_back) + np.array(cm_x_gone))/2
cm_y_mean = (np.array(cm_y_back) + np.array(cm_y_gone))/2

ax1.plot(cm_x_mean, '--k*')
ax2.plot(cm_y_mean, '--k*')    
f.set_tight_layout(True)
plt.savefig(os.path.join(save_folder, 'cm_center_scan.png'), dpi = 400)
plt.close()

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(cm_x_mean - cm_x, '-ko')
ax2.plot(cm_y_mean - cm_y, '-ko') 
ax1.set_ylabel('CM X mean - CM X image mean [nm]')
ax2.set_ylabel('CM Y mean - CM Y image mean [nm]')
f.set_tight_layout(True)
plt.savefig(os.path.join(save_folder, 'diff_cm_center_scan.png'), dpi = 400)
plt.close()
