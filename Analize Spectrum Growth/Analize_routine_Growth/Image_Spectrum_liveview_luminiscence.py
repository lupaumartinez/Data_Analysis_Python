
import matplotlib.pyplot as plt 
import numpy as np
import re
import os
import scipy.signal as sig

def smooth_Signal(signal, window, deg, repetitions):
    
    k = 0
    while k < repetitions:
        signal = sig.savgol_filter(signal, window, deg, mode = 'mirror')
        k = k + 1
        
    return signal

base_folder = 'C:/Ubuntu_archivos/Printing/'

daily_folder = '2021-08 (Growth PL circular)/2021-08-09 (growth, PL circular)/growth/20210809-180739_Growth_10x2_col11y12_580nm/'
   
file = 'Liveview_Spectrum_Col_001_NP_006'

folder = os.path.join(daily_folder, file)
parent_folder = os.path.join(base_folder, folder)

print(parent_folder)

save_folder = os.path.join(parent_folder, 'image')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
             

list_of_folders = os.listdir(parent_folder)

list_of_files = [f for f in list_of_folders if re.search('Line_Spectrum_step',f) and not re.search('Background',f)]
list_of_files.sort()

list_of_files_bkg = [f for f in list_of_folders if re.search('Background_Line',f)]
list_of_files_bkg.sort()

file_calibration = [f for f in list_of_folders if re.search('Calibration_Shamrock',f)][0]
name_file = os.path.join(parent_folder, file_calibration)
wavelength = np.loadtxt(name_file)

list_of_files_poly_fit = [f for f in list_of_folders if re.search('PolySpectrum',f) and not re.search('Max_Poly',f)]
list_of_files_poly_fit.sort()
name_file = os.path.join(parent_folder, list_of_files_poly_fit[0])
data = np.loadtxt(name_file)
wavelength_poly = data[:, 0]

list_of_files_max = [f for f in list_of_folders if re.search('Max_Poly',f)]
list_of_files_max.sort()

N = len(list_of_files)
L = len(wavelength)

first_wave = 501 #region integrate_antistokes
starts_notch = 521 #region integrate_antistokes

ends_notch = 542 #  #550 gold  #sustrate 700    #water  625 #antistokes 500
last_wave = 645# 645 # #570 gold  #sustrate 725   #water 670  #antistokes 521

window, deg, repetitions = 21, 0, 1

#exposure_time = 4 #s
#time = np.array(range(N))*exposure_time

range_stokes = np.where((wavelength >= ends_notch) & (wavelength <= last_wave))
wavelength_stokes = wavelength[range_stokes]

range_notch = np.where((wavelength >starts_notch) & (wavelength <ends_notch))
#wavelength_notch = wavelength[range_notch]

matrix = np.zeros((N, len(wavelength)))
matrix_smooth = np.zeros((N, len(wavelength_stokes)))
matrix_poly = np.zeros((N, len(wavelength_poly)))

max_times = np.zeros(N)
max_poly = np.zeros(N)

for i in range(N):
    
    file_specs_poly = os.path.join(parent_folder, list_of_files_poly_fit[i])
    data = np.loadtxt(file_specs_poly)
    poly = data[:, 1]
    matrix_poly[i, :] = poly
    
    file_specs = os.path.join(parent_folder, list_of_files[i])
    file_specs_bkg = os.path.join(parent_folder, list_of_files_bkg[i])
    
    specs = np.loadtxt(file_specs)
    specs_bkg = np.loadtxt(file_specs_bkg)
    
    spectrum = specs # - specs_bkg
    
    spectrum_stokes = spectrum[range_stokes]
  #  spectrum_stokes = smooth_Signal(spectrum_stokes, window, deg, repetitions)
    
    spectrum_notch = spectrum[range_notch]  
    
    #spectrum_stokes = (spectrum_stokes - np.mean(spectrum_notch))/(max(spectrum_stokes) - np.mean(spectrum_notch))
   # matrix_stokes[i, :] = spectrum_stokes
    
    matrix[i, range_stokes] = spectrum_stokes
    matrix[i, range_notch] = spectrum_notch
    
    matrix_smooth[i, :] = smooth_Signal(spectrum_stokes, window, deg, repetitions)
   # matrix[i, range_notch] = spectrum_notch
  
    file_max = os.path.join(parent_folder, list_of_files_max[i])
    data = np.loadtxt(file_max, skiprows = 1)
    max_times[i] = data[0]
    max_poly[i] = data[1]
    
    
fig = plt.figure(figsize=(18, 4))
axes = fig.subplots(ncols=3, nrows=1)

#axes[1].plot(wavelength, matrix[0, :], 'b')
#axes[1].plot(wavelength_poly, matrix_poly[0, :], 'k--')
#axes[1].plot(wavelength, matrix[-1, :], 'r')
#axes[1].plot(wavelength_poly, matrix_poly[-1, :], 'k--')
    
#axes.set_title()
ims = axes[0].imshow(matrix, origin = 'lower', extent=[np.min(wavelength), np.max(wavelength),
                                 np.min(max_times), np.max(max_times)], cmap= 'inferno')

range_plot = np.where((max_poly > 548))
axes[0].plot(max_poly[range_plot], max_times[range_plot], 'k.--')

axes[0].grid(False)
axes[0].set_ylabel("Time (s)")
axes[0].set_xlabel("Wavelength (nm)")

cax = axes[0].inset_axes([1.04, 0, 0.03, 1])
cbar = fig.colorbar(ims, ax=axes[0], cax=cax)
cbar.set_label("Counts Stokes PL")

ims = axes[1].imshow(matrix_smooth, origin = 'lower', extent=[np.min(wavelength_stokes), np.max(wavelength_stokes),
                                 np.min(max_times), np.max(max_times)], cmap= 'inferno')

range_plot = np.where((max_poly > 548))
axes[1].plot(max_poly[range_plot], max_times[range_plot], 'k.--')

axes[1].grid(False)
axes[1].set_ylabel("Time (s)")
axes[1].set_xlabel("Wavelength (nm)")

cax = axes[1].inset_axes([1.04, 0, 0.03, 1])
cbar = fig.colorbar(ims, ax=axes[0], cax=cax)
cbar.set_label("Counts Stokes PL")


#axes.set_title()
ims = axes[2].imshow(matrix_poly, origin = 'lower', extent=[np.min(wavelength_poly), np.max(wavelength_poly),
                                 np.min(max_times), np.max(max_times)], cmap= 'inferno')

range_plot = np.where((max_poly > 548))
axes[2].plot(max_poly[range_plot], max_times[range_plot], 'k.--')

axes[2].grid(False)
axes[2].set_ylabel("Time (s)")
axes[2].set_xlabel("Wavelength (nm)")

cax = axes[2].inset_axes([1.04, 0, 0.03, 1])
cbar = fig.colorbar(ims, ax=axes[1], cax=cax)
cbar.set_label("Counts Stokes PL")

name_fig = os.path.join(save_folder, 'image_Max_2.png')
plt.savefig(name_fig, dpi = 400)