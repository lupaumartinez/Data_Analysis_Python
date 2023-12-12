# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:11:02 2021

@author: Ituzaingo
"""
import os
import numpy as np
import matplotlib.pyplot as plt

#La carpeta 1
folderlist=[]
combined_mean=np.array([])

base_folder = r'C:\Ubuntu_archivos\Printing'
daily_folder = r'2022-06-22 Au60 satelites Pd'

parent_folder = os.path.join(base_folder, daily_folder)

save_folder = os.path.join(parent_folder, 'combined_mean_beta_psf_298nm')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

folder1 = 'processed_data_298nm'
folder1 = os.path.join('20220622-140802_Luminescence_Load_grid_in_best_center', folder1)


folder2 = 'processed_data_298nm'
folder2 = os.path.join('20220622-115036_Luminescence_Load_grid_in_best_center', folder2)


folder3 = 'processed_data_298nm'
folder3 = os.path.join('20220622-140802_Luminescence_Load_grid_2_in_best_center', folder3)


folderlist = [folder1, folder2, folder3]

# folderlist.append('20211124-112922_Luminescence_10x10')
# folderlist.append('20211123-163810_Luminescence_10x10')

for x in folderlist:
    
    path_to = os.path.join(parent_folder,x,'stats','list_of_mean.dat')
    data=np.loadtxt(path_to, skiprows=1)
    combined_mean=np.concatenate((combined_mean, data), axis=None)

combined_mean=combined_mean[combined_mean != 0]

##########

plt.figure()
nbins = 10
range_tuple = [0,120]
plt.hist(combined_mean, bins=nbins, range=range_tuple, rwidth = 1, \
              align='mid', alpha = 1, edgecolor='k')#, normed = False,
              #histtype = 'barstacked', stacked=False)
plt.legend(loc=1, handlelength=0)
ax = plt.gca()
ax.axvline(np.median(combined_mean), ymin = 0, ymax = 1, color='k', linestyle='--', linewidth=2,label="{0} ± {1} - N={2}".format(round(np.median(combined_mean),1), round((np.std(combined_mean)/(np.sqrt(len(combined_mean)))),1),len(combined_mean)))
ax.set_xlim(range_tuple)
ax.set_xlabel(u'Photothermal coefficient (K µm$^{2}$/mW)')
ax.set_ylabel('Entries')
plt.legend(loc='upper right')
figure_name = os.path.join(save_folder,'combined_mean_beta_TOTAL.png')
plt.legend(loc='upper right')
plt.savefig(figure_name)
plt.show()
plt.close()

###Data Save
core=67
T=0 #The tickness
Thickness=np.ones_like(combined_mean)*T
data_name = os.path.join(save_folder,'combined_mean_beta_TOTAL_Au%s_Pl%s.txt'%(core,T))
np.savetxt(data_name, (combined_mean,Thickness), fmt='%.3e')

test=np.loadtxt(data_name)
