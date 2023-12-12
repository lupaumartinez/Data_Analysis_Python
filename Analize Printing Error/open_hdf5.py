import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

from PIL import Image

def save_scan(filepath, image, image_gone, image_back, name_NP):
    
    timestr = name_NP
    
    name = str(filepath + "/" + "Scan" + "/" + timestr + ".tiff")
    guardado = Image.fromarray(np.transpose(image))
    guardado.save(name)  
    
    name_gone = str(filepath +  "/" + "Scan_gone" + "/" + "/" + "gone_" + timestr + ".tiff")
    guardado_gone = Image.fromarray(np.transpose(image_gone))
    guardado_gone.save(name_gone)
    
    name_back = str(filepath + "/" + "Scan_back" + "/" + "back_" + timestr + ".tiff")
    guardado_back = Image.fromarray(np.transpose(image_back))
   # guardado_back = Image.fromarray((np.fliplr(np.transpose( np.flip(image_back)))))
    guardado_back.save(name_back)
    
    print("\n Sacans saved\n")
    
#%%
        
prefix_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2019/Mediciones_PyPrinting/'
local_folder  = '2019-11-11/labview_34pixel'

direction = os.path.join(prefix_folder, local_folder)

save_folder = os.path.join(direction, 'tiff_NPscan')

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
    folder_scan = os.path.join(save_folder, 'Scan')
    folder_scan_back = os.path.join(save_folder, 'Scan_back')
    folder_scan_gone = os.path.join(save_folder, 'Scan_gone')
    
    if not os.path.exists(folder_scan):
        
        os.makedirs(folder_scan)
        os.makedirs(folder_scan_back)
        os.makedirs(folder_scan_gone)
        
print(direction)

for files in os.listdir(direction):
    
    if files.startswith("NPscan"):
        
        name_NP = files.split('.')[0]
        
        print(name_NP)

        name = direction + "/" + files
        f= h5py.File(name, 'r')
        
        #Get the key
        scan_go =  list(f.keys())[1]
        scan_return = list(f.keys())[2]

        # Get the data
        image_gone = list(f[scan_go])
        image_back = list(f[scan_return])
    
        #image= image_gone + np.fliplr(image_back) 
        image= image_gone + np.array(image_back)
        
        save_scan(save_folder, image, image_gone, image_back, name_NP)
        
        
