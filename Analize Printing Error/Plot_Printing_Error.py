# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:53:47 2019

@author: Luciana
"""

from tkinter import Tk, filedialog
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy import stats

def plot_printing_error_txt(direction, save_folder):

    for files in os.listdir(direction):
        if files.startswith("printing_error-"):
            name = direction + "/" + files
            data = np.loadtxt(name)
                
    error_x = -data[:,0]
    error_y = -data[:,1]
    
  #  print('Cantidad printing error txt:', len(error_x), len(error_y))
    
   # desired_range_x = np.where((error_x > -200) & (error_x < 200))
   # error_x = error_x[desired_range_x]
    
   # desired_range_y = np.where((error_y > -200) & (error_y < 200))
   # error_y = error_y[desired_range_y]
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    ax1.plot(error_y, error_x, '.')
    ax1.set_xlabel('Error Y [nm]')
    ax1.set_ylabel('Error X [nm]')
    ax1.set_xlim(-100,100)
    ax1.set_ylim(-100,100)
    ax1.set_aspect('equal')
    
    n,bin_positions_x,p = ax2.hist(error_x, bins=15, range=[-150,150], density=True, rwidth=0.9, color='C1')
    mu_x, sigma_x, x_text = data_hist(error_x, bin_positions_x)
    #ax2.plot(x_gaussiana, x_fit,'r--') #grafico la gaussiana
    ax2.text(0, 0.008, x_text, fontsize = 'xx-small')
    ax2.set_xlabel('Error X [nm]')
    ax2.set_ylim(0, 0.01)
    ax2.set_xlim(-150, 150)
    
    n,bin_positions_y,p = ax3.hist(error_y, bins=15, range=[-150,150], density = True, rwidth=0.9, color='C2')
    mu_y, sigma_y, y_text = data_hist(error_y, bin_positions_y)
   # ax3.plot(y_gaussiana, y_fit,'r--') #grafico la gaussiana
    ax3.text(0, 0.008, y_text,fontsize = 'xx-small')
    ax3.set_xlabel('Error Y [nm]')
    ax3.set_ylim(0, 0.01)
    ax3.set_xlim(-150, 150)
    
    sigma = np.sqrt(sigma_x**2 + sigma_y**2)
    circle = plt.Circle((mu_y, mu_x), sigma, color='g', linestyle = '--', fill=False)
    ax1.add_artist(circle)
    print('media promedio x e y:', sigma)
    
    f.set_tight_layout(True)
    
    plt.savefig(save_folder + '/Printing_error_txt.jpg', dpi = 500)
    plt.close()
    
    percentile(error_x, error_y, save_folder, 'txt')
    
    return error_x, error_y

def data_hist(x, bin_positions):
    
    mu= round(np.mean(x), 2) # media
    sigma= round(np.std(x, ddof=1), 2) #desviación estándar
    N=len(x) # número de cuentas
    std_err = round(sigma / N,2) # error estándar
    
    # muestro estos resultados
    print( 'media: ', mu)
    print( 'desviacion estandar: ', sigma)
    print( 'total de cuentas: ', N)
    print( 'error estandar: ', std_err)

   # bin_size=bin_positions[1]-bin_positions[0] # calculo el ancho de los bins del histograma
   # x_gaussiana=np.linspace(mu-5*sigma, mu+5*sigma, num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
   # gaussiana= norm.pdf(x_gaussiana, mu, sigma)#*N*bin_size # calculo la gaussiana que corresponde al histograma
    
    txt = '%s + %s'%(mu, sigma)
    
    return mu, sigma, txt

def percentile(target_x, target_y, save_folder, method):
    
    x = abs(target_x)
    y = abs(target_y)
    r = np.sqrt(x**2 + y**2)
    
    x = np.sort(x)
    y = np.sort(y)
    r = np.sort(r)
    
  #  n_bins = len(r)
  
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    
    mu_y = np.mean(y)
    sigma_y = np.std(y)
  
    mu= np.mean(r)
    sigma = np.std(r)

    plt.figure()
  #  n, bins, patches = plt.hist(x, bins= n_bins, histtype='step', cumulative= True, rwidth=0.9, color='C1', label = 'Error X')
  #  n, bins, patches = plt.hist(y, bins= n_bins, histtype='step', cumulative= True, rwidth=0.9, color='C2', label = 'Error Y')
   # n, bins, patches = plt.hist(r, bins= n_bins, histtype='step', cumulative= True, rwidth=0.9, color='C0', label = 'Total error')
    
    # Add a line showing the expected distribution.
    
    xplot = np.linspace(0, 200, 1000)
    
    cdf_x = stats.norm.cdf(xplot, loc=mu_x, scale=sigma_x)
    cdf_y = stats.norm.cdf(xplot, loc=mu_y, scale=sigma_y)
    cdf_r = stats.norm.cdf(xplot, loc=mu, scale=sigma)
    
   # r_teo = ((1 / (np.sqrt(2 * np.pi) * sigma)) *np.exp(-0.5 * (1 / sigma * (n- mu))**2))
   # r_teo = r_teo.cumsum()
   # r_teo = 100*r_teo/r_teo[-1]
   
    plt.plot(xplot, cdf_x*100, '--', linewidth = 1.5,  color = 'C1', label = 'Error X')
    plt.plot(xplot, cdf_y*100, '--', linewidth = 1.5,  color = 'C2', label = 'Error Y')
    plt.plot(xplot, cdf_r*100, '--', linewidth = 1.5,  color = 'C3', label = 'Error r')
    
    #plt.axhline(34.1, color = 'grey', linestyle = '--')
    #plt.axhline(68.27, color = 'grey', linestyle = '--')
    #plt.axhline(95.45, color = 'grey', linestyle = '--')
    
    plt.axhline(33, color = 'grey', linestyle = '--')
    plt.axhline(66, color = 'grey', linestyle = '--')
    plt.axhline(90, color = 'grey', linestyle = '--')
    
    plt.ylim(0, 100)
    plt.xlim(0, 200)
    plt.legend(loc='lower right')
    plt.ylabel('Percentile')
    plt.xlabel('Error [nm]')
    plt.axes().set_aspect('equal')
    
    plt.savefig(save_folder + '/Percentile_printing_error_%1s.jpg'%method, dpi = 500)
    plt.close()
    
    R_90 = printing_radius(xplot, cdf_r, 90)
    R_66 = printing_radius(xplot, cdf_r, 66)
    R_33 = printing_radius(xplot, cdf_r, 33)
    
    plot_2d_scatter(save_folder, method, target_x, target_y, R_33, R_66, R_90)
    
def printing_radius(r, cdf_r, p):
    
    index = np.argmin(abs(cdf_r*100 - p))

    R = r[index]
    
    print('printing radius:', cdf_r[index]*100, R)
    
    return R
    
def plot_2d_scatter(save_folder, method, target_x, target_y, R_33, R_66, R_90):
     
    theta = np.linspace(0,2*np.pi, 100)
    
    plt.plot(target_y, target_x, '.')
    plt.plot(R_90*np.cos(theta), R_90*np.sin(theta), color = 'grey', linestyle = '--')
    plt.plot(R_66*np.cos(theta), R_66*np.sin(theta), color = 'grey', linestyle = '--')
    plt.plot(R_33*np.cos(theta), R_33*np.sin(theta), color = 'grey', linestyle = '--')
    plt.text(-5, R_90-10, '%d nm'%R_90, color = 'grey', fontsize=10)
    plt.text(-5, R_66-10, '%d nm'%R_66, color = 'grey', fontsize=10)
    plt.text(-5, R_33-10, '%d nm'%R_33, color = 'grey', fontsize=10)
    plt.xlabel('Error Y [nm]')
    plt.ylabel('Error X [nm]')
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.axes().set_aspect('equal')
    
    plt.savefig(save_folder + '/2D_Scatter_%1s.jpg'%method, dpi = 500)
    plt.close()
    
def open_data(folder):

    list_of_files = os.listdir(folder)
    list_of_files.sort()
    
    first_scan = io.imread(folder + "/" + list_of_files[0])
    stack = np.zeros((len(list_of_files), first_scan.shape[0], first_scan.shape[1]))
            
    for n in range(0,len(list_of_files)):
        stack[n,:,:]= io.imread(folder + "/"+ list_of_files[n])
        
    return stack

def center_of_mass_old(image, image_type):

    Z = image  #solo ida
    
    Zn = (Z-min(map(min,Z)))/(max(map(max,Z))-min(map(min,Z)))
        
    if not image_type:
        Zn = Zn - 1 #filtro
        Zn = np.abs(Zn)
            
    for i in range(len(Z[:,1])):
        for j in range (len(Z[1,:])):
            if Zn[i,j] < 0.7:
                Zn[i,j] = 0
                                                 
    com = np.array(list(ndimage.measurements.center_of_mass(Zn)))
    
    return com

def center_of_mass(image, n_filter, rango_x, rango_y):
    
    new_image = (image- np.min(image) ) / (np.max(image) -  np.min(image))

    for i in range(len(new_image)):
        for j in range(len(new_image)):
            if new_image[i, j] <= n_filter:
                    new_image[i, j] = 0
                    com = ndimage.measurements.center_of_mass(new_image)
                    com = np.round(com, 2)

    Nx = image.shape[0]
    px_nm = rango_x/Nx
    
    Ny = image.shape[1]
    py_nm = rango_y/Ny

    com_nm = (np.array(com[0]) - Nx/2 + 1/2)*px_nm, (np.array(com[1]) - Ny/2 + 1/2)*py_nm
    com_nm = np.around(com_nm, 2)
    
 #   print('center of mass px (x0, y0) = ({}, {})'.format(*com))
 #   print('center of mass nm (x0, y0) = ({}, {})'.format(*com_nm))
    
    return com, com_nm

def gaussian(x, Io, xo, wo, C):

    return Io*np.exp(-2*(x-xo)**2/(wo**2)) + C

def fit_gaussian(gaussian, x, y):
    
    #mean = sum(x * y)
    #sigma = np.sqrt(sum(y*(x - mean)**2))
    #xo = np.argmax(y)*2000/34
    
    xo = 0
    
    init_params = [1, xo, 250, 0]
    
    popt, pcov = curve_fit(gaussian, x, y, p0 = init_params)
    
    x_fit = np.linspace(x[0], x[-1], 1000)
    #x_fit = np.arange(x[0], x[-1], 100)
    y_fit = gaussian(x_fit, popt[0], popt[1], popt[2], popt[3])

    return [x_fit, y_fit], popt

def curve_gauss(image, rango):

	profile_x = np.mean(image, axis = 1)
	profile_y = np.mean(image, axis = 0)

	#pixel_size = rango/image.shape[0]
	#axe = np.arange(0, rango, pixel_size)
	axe = np.linspace(-rango/2, rango/2, image.shape[0])

	return axe, profile_x, profile_y

def plot_scan_cm(stack_scan, plot_bool, fit_psf, fit_psf_2D):
    
    target_x = []
    target_y = []
    
    xo_gauss = []
    yo_gauss = []
    wx_gauss = []
    wy_gauss = []
    
    n_filter = 0.7
    range_x = 2000
    range_y = 2000
    
    for i in range(stack_scan.shape[0]):
        
        image = stack_scan[i,:,:]
        
        center_mass_px, center_mass = center_of_mass(image, n_filter, range_x, range_y)
        
        target_x.append(center_mass[0])
        target_y.append(center_mass[1])
            
        if fit_psf:
            
            ajuste_x, ajuste_y = psf(image, False)
            
            xo_gauss.append(-ajuste_x[1])
            wx_gauss.append(ajuste_x[2])
            yo_gauss.append(ajuste_y[1])
            wy_gauss.append(ajuste_y[2])
            
            factor_scale = 2000/34
            center_gauss = (np.array([-ajuste_x[1], ajuste_y[1]]) + 1000)/factor_scale
            
        if fit_psf_2D:
            
            center_gauss_2D_px, center_gauss_2D, w_gauss_2D = center_of_gauss(image, center_mass_px, range_x, range_y)
            
            xo_gauss.append(center_gauss_2D[0])
            wx_gauss.append(w_gauss_2D[0])
            yo_gauss.append(center_gauss_2D[1])
            wy_gauss.append(w_gauss_2D[1])
            
        if plot_bool:
            
            f, ax1 = plt.subplots(1, 1)
        
            ax1.imshow(image)
            ax1.scatter(*center_mass_px[::-1], color = 'r')
            
            if fit_psf:
                ax1.scatter(*center_gauss[::-1], color = 'g')
                
            if fit_psf_2D:
                ax1.scatter(*center_gauss_2D_px[::-1], color = 'b')


    return target_x, target_y, xo_gauss, yo_gauss, wx_gauss, wy_gauss

def psf(image, plot_bool):
    
    axe, profile_x, profile_y = curve_gauss(image, 2000)
    profile_x = (profile_x - min(profile_x)) /(max(profile_x) - min(profile_x))
    profile_y = (profile_y - min(profile_y)) /(max(profile_y) - min(profile_y))
    
    image_fitx, ajuste_x = fit_gaussian(gaussian, axe, profile_x)
    image_fity, ajuste_y = fit_gaussian(gaussian, axe, profile_y)
    
    #print('Gauss ajuste:', 'axe x', ajuste_x, 'axe y', ajuste_y)
    
    if plot_bool:
    
        plt.plot(axe, profile_x, label = 'axe x')
        plt.plot(axe, profile_y, label = 'axe y')
        plt.plot(image_fitx[0], image_fitx[1], 'r--')
        plt.plot(image_fity[0], image_fity[1], 'b--')
        plt.legend()
        plt.show()
        
    return ajuste_x, ajuste_y


def plot_printing_error_scan(direction, save_folder, type_scan, fit_psf, fit_psf_2D):
    
    direction_centerscan = direction + "/" + type_scan
    
    stack_centerscan = open_data(direction_centerscan)
    
    target_x, target_y, xo_gauss, yo_gauss, wx_gauss, wy_gauss = plot_scan_cm(stack_centerscan, plot_bool = False, fit_psf = fit_psf, fit_psf_2D = fit_psf_2D)
    
    print('Cantidad printing error scan:', len(target_x), len(target_y))
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    ax1.plot(target_y, target_x, '.')
    ax1.set_xlabel('Error Y [nm]')
    ax1.set_ylabel('Error X [nm]')
    ax1.set_xlim(-100,100)
    ax1.set_ylim(-100,100)
    ax1.set_aspect('equal')
    
    ax2.hist(target_x, bins=15, range=[-150,150], density = True, rwidth=0.9, color='C1')
    ax2.set_xlabel('Error X [nm]')
    #ax2.set_ylabel('counts')
    ax2.set_ylim(0, 0.01)
    ax2.set_xlim(-150, 150)
    
    ax3.hist(target_y, bins=15, range=[-150,150], density = True, rwidth=0.9, color='C2')
    ax3.set_xlabel('Error Y [nm]')
    
    ax3.set_ylim(0, 0.01)
    ax3.set_xlim(-150, 150)
    
    f.set_tight_layout(True)
    plt.savefig(save_folder + '/Printing_error_%1s.jpg'%(type_scan), dpi = 500)
    plt.close()
    
    if fit_psf:
        
        plot_histogram_psf(save_folder, type_scan, xo_gauss, yo_gauss, wx_gauss, wy_gauss)
        
    if fit_psf_2D:
        
        plot_histogram_psf(save_folder, type_scan, xo_gauss, yo_gauss, wx_gauss, wy_gauss)
        
    name = os.path.join(save_folder + '/Printing_error_%1s.txt'%(type_scan))
    data = np.array([target_x, target_y, xo_gauss, yo_gauss, wx_gauss, wy_gauss]).T
    header_txt = 'x_cm, y_cm, x_gauss, y_gauss, w_x, w_y'
    np.savetxt(name, data, header = header_txt)

    return target_x, target_y

def plot_difference(save_folder, x_1, x_2, y_1, y_2):
    
    difference_x = x_2 - x_1
    difference_y = y_2 - y_1
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.plot(difference_x, 'o')
    ax2.plot(difference_y, 'o')
    
    ax1.set_ylim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    ax1.set_xlabel('Difference Printing Error X [nm]')
    ax1.set_ylabel('Difference Printing Error Y [nm]')
    
    f.set_tight_layout(True)
    plt.savefig(save_folder + '/Printing_error_difference_txt_Scan.jpg', dpi = 500)
    plt.close()
    
    return
    
def plot_histogram_psf(save_folder, type_scan, xo, yo, wx, wy):
    
   # center_image = 1000*np.ones(len(xo))
    xo = np.array(xo) #- center_image
    yo = np.array(yo) #- center_image
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.title('Xo and Yo from Fit Gauss')
    
    ax1.plot(yo, xo, '.')
    ax1.set_xlabel('Error Y [nm]')
    ax1.set_ylabel('Error X [nm]')
    ax1.set_xlim(-100,100)
    ax1.set_ylim(-100,100)
    ax1.set_aspect('equal')
    
    ax2.hist(xo, bins=15, range=[-150,150], density = True, rwidth=0.9, color='C1')
    ax2.set_xlabel('Error X [nm]')
    ax2.set_ylim(0, 0.015)
    ax2.set_xlim(-150, 150)
    
    ax3.hist(yo, bins=15, range=[-150,150], density = True, rwidth=0.9, color='C2')
    ax3.set_xlabel('Error Y [nm]')
    ax3.set_ylim(0, 0.015)
    ax3.set_xlim(-150, 150)
    
    f.set_tight_layout(True)
    plt.savefig(save_folder + '/Gauss_Printing_error_%1s.jpg'%(type_scan), dpi = 500)
    plt.close()
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.title('Wx and Wy from Fit Gauss')
    
    ax1.plot(wy, wx, '.')
    ax1.set_xlabel('Wy [nm]')
    ax1.set_ylabel('Wx [nm]')
    ax1.set_xlim(250,350)
    ax1.set_ylim(250,350)
    ax1.set_aspect('equal')

    ax2.hist(wx, bins=15, range=[250,350], density = True, rwidth=0.9, color='C1')
    ax2.set_xlabel('Wx [nm]')
    ax2.set_ylim(0, 0.015)
    ax2.set_xlim(250,350)
    
    ax3.hist(wy, bins=15, range=[250,350], density = True, rwidth=0.9, color='C2')
    ax3.set_xlabel('Wy [nm]')
    
    ax3.set_ylim(0, 0.015)
    ax3.set_xlim(250,350)
    
    f.set_tight_layout(True)
    plt.savefig(save_folder + '/Gauss_omega_%1s.jpg'%(type_scan), dpi = 500)
    plt.close()
    
    percentile(xo, yo, save_folder, 'Gauss_' + type_scan)
    
    w = np.sqrt(np.array(wx)**2+np.array(wy)**2)
    print(type_scan, 'mean, std:', 'N', len(wx), 'wx', np.mean(wx), np.std(wx), 'wy', np.mean(wy), np.std(wy), 'w ratio', np.mean(w), np.std(w))
    
    
def gaussian2D(grid, amplitude, x0, y0, σ_x, σ_y, offset, theta=0):
    
    # TO DO (optional): change parametrization to this one
    # http://mathworld.wolfram.com/BivariateNormalDistribution.html  
    # supposed to be more robust for the numerical fit
    
    x, y = grid
    x0 = float(x0)
    y0 = float(y0)   
    a = (np.cos(theta)**2)/(2*σ_x**2) + (np.sin(theta)**2)/(2*σ_y**2)
    b = -(np.sin(2*theta))/(4*σ_x**2) + (np.sin(2*theta))/(4*σ_y**2)
    c = (np.sin(theta)**2)/(2*σ_x**2) + (np.cos(theta)**2)/(2*σ_y**2)
    G = offset + amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                            + c*((y-y0)**2)))
    return G.ravel()

def center_of_gauss(image, initial_position, range_x, range_y):
    
    Nx = image.shape[0]
    Ny = image.shape[1]
    
 #   print('number pixel', Nx, Ny)
    
    px_nm = range_x/Nx
    py_nm = range_y/Ny
    
    x = np.arange(-Nx/2 + 1/2, Nx/2)
    y = np.arange(-Ny/2 + 1/2, Ny/2) 
    
    	#x = np.arange(0, Nx)
    	#y = np.arange(0, Ny)
    [Mx, My] = np.meshgrid(x, y)
    
    dataG_2d = (image- np.min(image) ) / (np.max(image) -  np.min(image))
    dataG_ravel = dataG_2d.ravel()
    
    initial_sigma = [150/px_nm, 150/py_nm]
    
    #primero va lo de y, luego lo de x
    initial_guess_G = [1, initial_position[1] + y[0],  initial_position[0] + x[0], initial_sigma[1], initial_sigma[0], 0]
    bounds = ([0, y[0], x[0], 0, 0, 0], [1, -y[0], -x[0], 4*initial_sigma[1], 4*initial_sigma[0], 1])
    
    	#initial_guess_G = [1, initial_position[0],  initial_position[1], initial_sigma[0], initial_sigma[1], 0]
    	#bounds = ([0, 0, 0, 0, 0, 0], [1, Nx, Ny, 3*initial_sigma[0], 3*initial_sigma[0], 1])
    
    poptG, pcovG = curve_fit(gaussian2D, (Mx, My), dataG_ravel, p0= initial_guess_G, bounds = bounds)
    
    poptG = np.around(poptG, 2)
    #print('A = {}, (y0, x0) = ({}, {}), σ_y = {}, σ_x = {}, bkg = {}'.format(*poptG))
    
    cog =  poptG[2] - x[0], poptG[1] - y[0] 
    
    cog_nm =  (np.array(cog[0]) + x[0])*px_nm, (np.array(cog[1]) + y[0])*py_nm
    cog_nm = np.around(cog_nm, 2)
        
   # print('center of gauss: px (x0, y0) = ({}, {})'.format(*cog))
   # print('center of gauss: nm (x0, y0) = ({}, {})'.format(*cog_nm))
    
    w_nm = 2*poptG[4]*px_nm, 2*poptG[3]*py_nm
    w_nm = np.around(w_nm, 2)
   # print('center of gauss: nm (w_x w_x) = ({}, {})'.format(*w_nm))
    
    return cog, cog_nm, w_nm
    
#%%
    
#root = Tk()
#root.withdraw()
#direction = filedialog.askdirectory()

#prefix_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2019/Mediciones_PyPrinting/'

prefix_folder = '/media/luciana/F650A60950A5D0A3/Ubuntu_archivos/Printing/Mediciones_PyPrinting (2019)/'

local_folder  = '2019-10-30/20191030-171212_Printing_10x10_2.45mW/'
#local_folder  = '2019-10-30/LabView_Matlab_scan_scanestadistica/printing-20191030-183854_ Grilla 10x10 2.45 mW/tiff_NPscan/'

#local_folder  = '2019-10-30/Pree_Scan/'

direction = os.path.join(prefix_folder, local_folder)
print(direction)

save_folder = direction + 'figures_printing_error_2021'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#%%
    
bool_fit_psf = False
bool_fit_psf_2D = True

error_x_scan, error_y_scan = plot_printing_error_scan(direction, save_folder, "Scan", fit_psf = bool_fit_psf, fit_psf_2D = bool_fit_psf_2D)
error_x_scan_back, error_y_scan_back = plot_printing_error_scan(direction, save_folder, "Scan_back", fit_psf = bool_fit_psf, fit_psf_2D = bool_fit_psf_2D)
error_x_scan_gone, error_y_scan_gone = plot_printing_error_scan(direction, save_folder, "Scan_gone", fit_psf = bool_fit_psf, fit_psf_2D = bool_fit_psf_2D)

#error_x, error_y = plot_printing_error_txt(direction, save_folder)
#plot_difference(save_folder, error_x_scan, error_x, error_y_scan, error_y)
