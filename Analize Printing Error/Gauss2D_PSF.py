
import numpy as np
import matplotlib.pyplot as plt

import time
import os

from skimage import io
from scipy.optimize import curve_fit
from scipy import ndimage

def image_NP(N, rango, Io, sigma_x, sigma_y, xo, yo, C):

	image = np.zeros((N,N))
	#x = np.linspace(-rango/2, rango/2, N)
	x = np.arange(-rango/2, rango/2, rango/N)
	y=x

	p = Io, sigma_x, sigma_y, xo, yo, C

	for i in range(N):
		for j in range(N):
			image[i, j] = np.random.poisson(gaussian_2D(x[i], y[j], *p))
			#image[i, j] = gaussian_2D(x[i], y[j], *p)

	return image

def gaussian_2D(x, y, *p):

    Io, sigma_x, sigma_y, xo, yo, C = p

    g2D = Io*np.exp(   -(x-xo)**2/(2*sigma_x**2) - (y-yo)**2/(2*sigma_y**2)   ) + C

    return g2D

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

def center_mass(image, n_filter, rango_x, rango_y):
    
    
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
    
    com_nm = np.array(com[0])*px_nm - rango_x/2 + px_nm/2 , np.array(com[1])*py_nm  - rango_y/2 + py_nm/2
    com_nm = np.around(com_nm, 2)
    
    print('px (x0, y0) = ({}, {})'.format(*com))
    print('nm (x0, y0) = ({}, {})'.format(*com_nm))
    
    return com

def center_gauss(image, initial_position, range_x, range_y):
    
    Nx = image.shape[0]
    Ny = image.shape[1]
    
    px_nm = range_x/Nx
    py_nm = range_y/Ny
    
    x = np.arange(-Nx/2 +1/2, Nx/2)
    y = np.arange(-Ny/2 +1/2, Ny/2) 
    
    	#x = np.arange(0, Nx)
    	#y = np.arange(0, Ny)
    [Mx, My] = np.meshgrid(x, y)
    
    dataG_2d = (image- np.min(image) ) / (np.max(image) -  np.min(image))
    dataG_ravel = dataG_2d.ravel()
    
    initial_sigma = [150/px_nm, 150/py_nm]
    
    #primero va lo de y, luego lo de x
    initial_guess_G = [1, initial_position[1] + y[0],  initial_position[0] + x[0], initial_sigma[1], initial_sigma[0], 0]
    bounds = ([0, y[0], x[0], 0, 0, 0], [1,-y[0], -x[0], 4*initial_sigma[1], 4*initial_sigma[0], 1])
    
    	#initial_guess_G = [1, initial_position[0],  initial_position[1], initial_sigma[0], initial_sigma[1], 0]
    	#bounds = ([0, 0, 0, 0, 0, 0], [1, Nx, Ny, 3*initial_sigma[0], 3*initial_sigma[0], 1])
    
    poptG, pcovG = curve_fit(gaussian2D, (Mx, My), dataG_ravel, p0= initial_guess_G, bounds = bounds)
    
    poptG = np.around(poptG, 2)
    #print('A = {}, (y0, x0) = ({}, {}), σ_y = {}, σ_x = {}, bkg = {}'.format(*poptG))
    
    cog =  poptG[2] - x[0], poptG[1] - y[0]
    
    cog_nm =  (np.array(cog[0]) + x[0])*px_nm, (np.array(cog[1]) + y[0])*py_nm
    cog_nm = np.around(cog_nm, 2)
        
    print('px (x0, y0) = ({}, {})'.format(*cog))
    print('nm (x0, y0) = ({}, {})'.format(*cog_nm))
    
    sigma_nm = poptG[4]*px_nm, poptG[3]*py_nm
    w_nm = 2*np.around(sigma_nm, 2)
    print('nm (w_x, w_y) = ({}, {})'.format(*w_nm))
    
    return cog
	

if __name__ == '__main__':
    
    rango_x = 2000 # in nm
    rango_y = rango_x
    Nx = 34
    Ny = Nx
    
    image_real = True
    
    if image_real:
    
        #image real
        prefix_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2019/Mediciones_PyPrinting/'
        
      #  local_folder  = '2019-10-30/20191030-171212_Printing_10x10_2.45mW/'
        #local_folder  = '2019-10-30/LabView_Matlab_scan_scanestadistica/printing-20191030-183854_ Grilla 10x10 2.45 mW/tiff_NPscan/'
        #local_folder  = '2019-10-30/Pree_Scan/'
    
       # file = 'Scan/NPscan_001.tiff'
       
        local_folder  = '2019-11-11/'
        
        #file = 'scan_ramp_2/image_back_20191111-162203.tiff'
        #file = 'scan_ramp/image_gone_20191111-162147.tiff'
        #file = 'step_by_step/scan_20191111-162130.tiff'
        #file = 'labview_34pixel/tiff_NPscan/Scan_back/back_NPscan_image-20191111-163846.tiff'
        #file = 'labview_34pixel/tiff_NPscan/Scan_back/back_NPscan_image-20191111-164040.tiff'
        
        file = 'scan_stepbystep_mapa_Z_0.2um/z_-1.tiff'
        
        name_file = os.path.join(prefix_folder, local_folder, file)
        
        print(name_file)
        
        image = io.imread(name_file)
        
    else:

        #image false
        
        N_G = 1000 # number of counts at max
        x0_nm = 500 # in nm
        y0_nm = -500 # in nm
        σx_nm = 140
        σy_nm = 140
        bkg = 100 # in number of counts
        image = image_NP(Nx, rango_x, N_G, σx_nm, σy_nm, x0_nm, y0_nm, bkg)

	#center of mass
    
    n_filter = 0.2
    
    t0 = time.time()
    com = center_mass(image, n_filter, rango_x, rango_y)
    t1 = time.time()
    t_m = (t1-t0)*1e3 
    print('center of mass fit took {} ms'.format(np.around(t_m, 2)))
    
	#center of gauss
    
    initial_pos = com
    
    t0 = time.time()
    cog = center_gauss(image, initial_pos, rango_x, rango_y)
    t1 = time.time()
    t_G = (t1-t0)*1e3 
    
    print('gaussian fit took {} ms'.format(np.around(t_G, 2)))

	#plotear
    
    plt.figure('gaussian data')
    plt.imshow(image)
    plt.scatter(*cog[::-1], color = 'b')
    plt.scatter(*com[::-1], color = 'r')
    plt.show()
    
