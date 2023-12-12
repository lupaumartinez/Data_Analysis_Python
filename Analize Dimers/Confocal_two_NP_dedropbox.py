import matplotlib.pyplot as plt
import os
import numpy as np
import time
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage import io


def image_NP(N, rango, Io, sigma_x, sigma_y, xo, yo, C):

	image = np.zeros((N,N))
	#x = np.linspace(-rango/2, rango/2, N)
	px = rango/N
	x = np.arange(-rango/2 + px/2, rango/2, px)
	y=x

	p = Io, sigma_x, sigma_y, xo, yo, C

	for i in range(N):
		for j in range(N):
			image[i, j] = gaussian_2D(x[i], y[j], *p)  + 0.01*Io*np.random.normal(0, 1)

	return image


def filter_image(image, n):

	Z = image
	#Zn = norm_image(Z, type_image)

	for i in range(len(Z[:,1])):
		for j in range (len(Z[1,:])):
			if Z[i,j] < n:
				Z[i,j] = 0

	return Z

def norm_image(image, type_image):

	Z = image

	if type_image == 'maximum':

		Zn = (image-np.min(image))/(np.max(image)-np.min(image)) #min = 0, max = 1

	if type_image == 'minimum':

		Zn = (image-np.min(image))/(np.max(image)-np.min(image)) #min = 0, max = 1

		Zn = Zn - 1 #min = -1, max = 0
		Zn = np.abs(Zn) #min = 1, max = 0

	if type_image == 'maximum_minimum':

		bkg = np.mean(Z)

		Zn = (image-bkg)/(np.max(image)-bkg)  #min = -1, bkg = 0, max = 1

		Zn = np.abs(Zn)  #min, max = 1, bkg = 0

	return Zn

def gaussian_2D(x, y, *p):

    Io, sigma_x, sigma_y,  xo, yo, C = p

    g2D = Io*np.exp(- (x-xo)**2/(2*sigma_x**2) - (y-yo)**2/(2*sigma_y**2) ) + C

    return g2D

def gaussian(x, Io, xo, sigma, C):

    return Io*np.exp(-(x-xo)**2/(2*sigma**2)) + C

def fit_gaussian(gaussian, x, y):

	#mean = sum(x * y)
#	sigma = np.sqrt(sum(y*(x - mean)**2))

	xo = np.argmax(y)

	init_params = [1, xo, 135, 0]

	popt, pcov = curve_fit(gaussian, x, y, p0 = init_params)

	#x_fit = np.linspace(x[0], x[-1], 100)
	x_fit = np.arange(x[0], x[-1], 10)

	y_fit = gaussian(x_fit, popt[0], popt[1], popt[2], popt[3])

	return [x_fit, y_fit], popt

def curve_gauss(image, rango):

	profile_x = np.mean(image, axis = 1)
	profile_y = np.mean(image, axis = 0)

	pixel_size = rango/image.shape[0]
	axe = np.arange(-rango/2 + pixel_size/2, rango/2, pixel_size)
	#axe = np.linspace(-rango/2, rango/2, image.shape[0])

	return axe, profile_x, profile_y

def find_peaks(x, y, threshold_rel, number):

	index = peak_local_max(y, min_distance=1, threshold_rel = threshold_rel, num_peaks = number, indices=True)

	return index, x[index], y[index]

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


def two_gaussian2D(grid, amplitude, x0, y0, σ_x, σ_y, offset, amplitude1, x1, y1, theta=0):

    x, y = grid

    x0 = float(x0)
    y0 = float(y0) 

    x1 = float(x1)
    y1 = float(y1)   

    a = (np.cos(theta)**2)/(2*σ_x**2) + (np.sin(theta)**2)/(2*σ_y**2)
    b = -(np.sin(2*theta))/(4*σ_x**2) + (np.sin(2*theta))/(4*σ_y**2)
    c = (np.sin(theta)**2)/(2*σ_x**2) + (np.cos(theta)**2)/(2*σ_y**2)


    G0 = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))

    G1 = amplitude1*np.exp( - (a*((x-x1)**2) + 2*b*(x-x1)*(y-y1) + c*((y-y1)**2)))

    G =  offset + G0 + G1

    return G.ravel()

def center_mass(image, range_x, range_y):

	com = ndimage.measurements.center_of_mass(image)

	Nx = image.shape[0]
	px_nm = range_x/Nx

	Ny = image.shape[1]
	py_nm = range_y/Ny

	com_nm = (np.array(com[0]) - Nx/2 + 1/2)*px_nm, (np.array(com[1]) - Ny/2 + 1/2)*py_nm
	com_nm = np.around(com_nm, 2)

	print(' (x0, y0) = ({}, {})'.format(*com_nm))

	return com[::-1]

def center_gauss(image, initial_position, range_x, range_y):

	Nx = image.shape[0]
	Ny = image.shape[1]

	px_nm = range_x/Nx
	py_nm = range_y/Ny

	x = np.arange(-Nx/2 + 1/2, Nx/2)
	y = np.arange(-Ny/2 + 1/2, Ny/2) 

	#x = np.arange(0, Nx)
	#y = np.arange(0, Ny)
	[Mx, My] = np.meshgrid(x, y)

	dataG_2d = image
	dataG_ravel = dataG_2d.ravel()

	initial_sigma = [125/px_nm, 135/py_nm]

	initial_guess_G = [1, initial_position[0] - Nx/2,  initial_position[1] - Ny/2, initial_sigma[0], initial_sigma[1], 0]
	bounds = ([0, -Nx/2,- Ny/2, 0, 0, 0], [1, Nx/2, Ny/2, 4*initial_sigma[0], 4*initial_sigma[0], 1])

	#initial_guess_G = [1, initial_position[0],  initial_position[1], initial_sigma[0], initial_sigma[1], 0]
	#bounds = ([0, 0, 0, 0, 0, 0], [1, Nx, Ny, 3*initial_sigma[0], 3*initial_sigma[0], 1])

	poptG, pcovG = curve_fit(gaussian2D, (Mx, My), dataG_ravel, p0= initial_guess_G, bounds = bounds)

	poptG = np.around(poptG, 2)
	#print('A = {}, (y0, x0) = ({}, {}), σ_y = {}, σ_x = {}, bkg = {}'.format(*poptG))

	cog = poptG[2] + Nx/2 - 1/2, poptG[1] + Ny/2 - 1/2
	#cog = poptG[2], poptG[1]

	cog_nm = (np.array(cog[0]) - Nx/2 + 1/2)*px_nm, (np.array(cog[1]) - Ny/2 + 1/2)*py_nm
	cog_nm = np.around(cog_nm, 2)

	print(' (x0, y0) = ({}, {})'.format(*cog_nm))

	return cog[::-1]

def two_center_gauss(image, initial_position_1, initial_position_2, range_x, range_y):

	Nx = image.shape[0]
	Ny = image.shape[1]

	px_nm = range_x/Nx
	py_nm = range_y/Ny

	x = np.arange(-Nx/2 + 1/2, Nx/2)
	y = np.arange(-Nx/2 + 1/2, Ny/2) 

	#x = np.arange(0, Nx)
	#y = np.arange(0, Ny)
	[Mx, My] = np.meshgrid(x, y)

	dataG_2d = image
	dataG_ravel = dataG_2d.ravel()

	initial_sigma = [125/px_nm, 135/py_nm]
	bounds = ([0, x[0], y[0], 0, 0, 0, 0, x[0], y[0]], [1, x[-1], y[-1], 4*initial_sigma[0], 4*initial_sigma[0], 1, 1,x[-1], y[-1]])

	initial_guess_G = [1, initial_position_1[0] + x[0],  initial_position_1[1]  + y[0], initial_sigma[0], initial_sigma[1], 0, 1, initial_position_2[0] + x[0],  initial_position_2[1] +  y[0]]

	poptG, pcovG = curve_fit(two_gaussian2D, (Mx, My), dataG_ravel, p0= initial_guess_G, bounds = bounds)

	poptG = np.around(poptG, 2)
	#print('A = {}, (y0, x0) = ({}, {}), σ_y = {}, σ_x = {}, bkg = {}'.format(*poptG))

	cog_1 = poptG[2] - x[0], poptG[1] - y[0]
	cog_2 = poptG[8] - x[0], poptG[7] - y[0]
	#cog = poptG[2], poptG[1]

	cog_1_nm = (np.array(cog_1[0]) + x[0])*px_nm, (np.array(cog_1[1]) + y[0])*py_nm
	cog_1_nm = np.around(cog_1_nm, 2)

	cog_2_nm = (np.array(cog_2[0]) + x[0])*px_nm, (np.array(cog_2[1]) + y[0])*py_nm
	cog_2_nm = np.around(cog_2_nm, 2)

	print(' center gauss np1 (x0, y0) = ({}, {})'.format(*cog_1_nm))
	print(' center gauss np2 (x0, y0) = ({}, {})'.format(*cog_2_nm))

	return cog_1[::-1], cog_2[::-1]


if __name__ == '__main__':

    N= 34
    rango= 2000
    sigma_x = 140
    sigma_y = 140
   
    
    image_real = False
    
    plot_one_NP = False
    
    fit_Gauss_1D = False
    
    plot_two_NP = True
    
    if plot_one_NP:
        
        image = image_NP(N= N, rango=rango, Io= - 0.1, sigma_x =sigma_x, sigma_y = sigma_y, xo = 0, yo = 0, C = 0.5)

        image_norm = norm_image(image, 'minimum')

        #center of mass

        n_filter = 0.7
        image_filt = filter_image(image_norm, n_filter)
       
        t0 = time.time()    
        com = center_mass(image_filt, range_x = rango , range_y = rango)
        t1 = time.time()
        t_m = (t1-t0)*1e3
        print('center of mass fit took {} ms'.format(np.around(t_m, 2)))
        
        #center of gauss
        
        initial_pos = com
        
        t0 = time.time()
        cog = center_gauss(image_norm, initial_pos, range_x = rango, range_y = rango)
        t1 = time.time()
        t_G = (t1-t0)*1e3 
        
        print('gaussian fit took {} ms'.format(np.around(t_G, 2)))
        
        #plotear
        
        plt.figure('gaussian and center of mass data')
        plt.imshow(image, cmap='viridis', interpolation='None')
        plt.scatter(*cog, color = 'b')
        plt.scatter(*com, color = 'r')
        plt.show()


        if fit_Gauss_1D:
    
            axe, profile_x, profile_y = curve_gauss(image, rango = rango)
            
            profile_x = (profile_x - min(profile_x)) /(max(profile_x) - min(profile_x))
            profile_y = (profile_y - min(profile_y)) /(max(profile_y) - min(profile_y))
            
            image_fitx, ajuste_x = fit_gaussian(gaussian, axe, profile_x)
            image_fity, ajuste_y = fit_gaussian(gaussian, axe, profile_y)
            
            print('Gauss ajuste:', 'axe x', ajuste_x, 'axe y', ajuste_y)
            
            plt.plot(axe, profile_x, label = 'axe x')
            plt.plot(image_fitx[0],image_fitx[1], 'r--')
            plt.plot(axe, profile_y, label = 'axe y')
            plt.plot(image_fity[0],image_fity[1], 'b--')
            plt.legend()
            plt.show()
            
    if plot_two_NP:
        
        if image_real:
            
            #image real
            prefix_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2019/Mediciones_PyPrinting/'
            
            #local_folder  = '2019-10-30/20191030-171212_Printing_10x10_2.45mW/'
            #local_folder  = '2019-10-30/LabView_Matlab_scan_scanestadistica/printing-20191030-183854_ Grilla 10x10 2.45 mW/tiff_NPscan/'
            #local_folder  = '2019-10-30/Pree_Scan/'
            local_folder = '2019-10-30/20191030-182644_Dimers_10x1/'
            #local_folder  = '2019-10-30/LabView_Matlab_scan_scanestadistica/Dimers2019-10-30DX= 500nmDY=   0nm/tiff_NPscan/'
            
            #file = 'Scan_gone/gone_NPscan_002.tiff'
            file = 'Dimer_Scan/NPscan_003.tiff'
            
            name_file = os.path.join(prefix_folder, local_folder, file) 
            image = io.imread(name_file)
            
        else:
            
                image_2 = image_NP(N= N, rango=rango, Io = 0.8, sigma_x =sigma_x, sigma_y = sigma_y, xo = -300, yo =  0, C = 0)
                image_1 = image_NP(N= N, rango=rango, Io = -0.4, sigma_x =sigma_x, sigma_y = sigma_y, xo = 0, yo = 0, C = 0)
                image = image_1 + image_2 + 0.41
    
        t0 = time.time()
        image_norm = norm_image(image, 'maximum_minimum')
        image_filt = filter_image(image_norm, 0.1)
        t1 = time.time()
        t_f = (t1-t0)*1e3 
        print('filter image took {} ms'.format(np.around(t_f, 2)))
        
        plt.imshow(image_filt)
        plt.show()
        
        t0 = time.time()
        
        axe, profile_x, profile_y = curve_gauss(image_filt, rango = rango)
        
        index_x, axe_x, profile_x_max = find_peaks(axe, profile_x, threshold_rel = 0.1, number = 2)
        index_y, axe_y, profile_y_max = find_peaks(axe, profile_y, threshold_rel = 0.1, number = 2)
        
        print('local max index x', index_x, 'local max nm x', axe_x)
        print('local max index y', index_y, 'local max nm y', axe_y)
        
        if len(index_x) == 2 and len(index_y) == 1:
        
            pos_NP_1 = index_y[0][0], index_x[1][0]
            pos_NP_2 = index_y[0][0], index_x[0][0]
        
        if len(index_x) == 1 and len(index_y) == 2:
        
            pos_NP_1 = index_y[1][0], index_x[0][0]
            pos_NP_2 = index_y[0][0], index_x[0][0]
        
        if len(index_x) == 2 and len(index_y) == 2:
            
            pos_NP_1 = index_y[1][0], index_x[1][0]
            pos_NP_2 = index_y[0][0], index_x[0][0]
        
        if len(index_x) == 1 and len(index_y) == 1:
        
            pos_NP_1 = index_y[0][0], index_x[0][0]
            pos_NP_2 = index_y[0][0], index_x[0][0]
        
        t1 = time.time()
        t_plm = (t1-t0)*1e3 
        print('peak local max fit took {} ms'.format(np.around(t_plm, 2)))
        
        print('np1', pos_NP_1, 'np2', pos_NP_2)
        
        plt.plot(axe, profile_x, label = 'axe x')
        plt.plot(axe, profile_y, label = 'axe y')
        plt.plot(axe_x, profile_x_max, 'ro')
        plt.plot(axe_y, profile_y_max, 'bo')
        plt.legend()
        plt.show()
        
        #plotear
        
        plt.title('Local Maximum')
        plt.imshow(image, cmap='viridis', interpolation='None')
        plt.scatter(*pos_NP_1, color = 'k')
        plt.scatter(*pos_NP_2, color = 'm')
        plt.show()
        
        
        #center of gauss
        
        initial_pos_1 = pos_NP_1
        initial_pos_2 = pos_NP_2
        t0 = time.time()
        cog_1, cog_2 = two_center_gauss(image_norm, initial_pos_1, initial_pos_2, range_x = rango, range_y = rango)
        t1 = time.time()
        t_G = (t1-t0)*1e3 
        
        print('gaussian fit took {} ms'.format(np.around(t_G, 2)))
        
        #plotear gauss
        plt.title('Gauss 2D')
        plt.imshow(image, cmap='viridis', interpolation='None')
        plt.scatter(*cog_1, color = 'k')
        plt.scatter(*cog_2, color = 'm')
        plt.show()
        
        print(' gauus + peak + filter: complete {} ms'.format(np.around(t_f + t_plm + t_G, 2)))

        
            	   