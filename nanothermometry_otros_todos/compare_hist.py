# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:11:02 2021

@author: Ituzaingo
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

#La carpeta 1
folderlist=[]
combined_mean=np.array([])

base_folder = r'C:\Users\lupau\OneDrive\Documentos'

daily_folder_1 = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-161433_Luminescence_Load_grid_in_best_center_size_13'

daily_folder_2 = r'2022-07-15 Nanostars R20 drop and cast PL\20220715-174344_Luminescence_Load_grid_in_best_center_size_13'

parent_folder = os.path.join(base_folder,'2022-07-15 Nanostars R20 drop and cast PL')

save_folder = os.path.join(parent_folder, 'Compare_betha')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
folderlist = [daily_folder_1, daily_folder_2]
colors = ['green', 'orange']
labels = ['532 nm', '592 nm']

plt.figure()
ax = plt.gca()
ax.set_xlabel(u'Photothermal coefficient (K µm$^{2}$/mW)')
ax.set_ylabel('Entries')
figure_name = os.path.join(save_folder,'compare_mean_beta.png')
plt.legend(loc='upper right')

nbins = 10
range_tuple = [0,120]
ax.set_xlim(range_tuple)

for i in range(2):
    
    path_to = os.path.join(base_folder,folderlist[i], r'processed_data\stats')
    #files = os.listdir(path_to)
   # file = [f for f in files if re.search('txt',f)][0]
    #print(file)
    file = 'list_of_mean.dat'
    file = os.path.join(path_to, file)
    
    data=np.loadtxt(file)
    combined_mean= data#[0]
    
    print(np.median(combined_mean))

    plt.hist(combined_mean, bins=nbins, density=None,range=range_tuple, rwidth = 1, \
                  align='mid', color = colors[i], alpha = 0.5)#, edgecolor ='k')#, normed = False,
                  #histtype = 'barstacked', stacked=False)

  
    ax.axvline(np.median(combined_mean), ymin = 0, ymax = 1, color= colors[i], linestyle='--', linewidth=2,label= "{0} = {1} ± {2} - N={3}".format(labels[i], round(np.median(combined_mean),1), round((np.std(combined_mean)/(np.sqrt(len(combined_mean)))),1),len(combined_mean)))

    plt.show()

plt.legend(loc=1, handlelength=0)    
plt.savefig(figure_name)