# -*- coding: utf-8 -*-
"""
Spyder Editor

MODUS OPERANDI:
    Correr el script de forma similar a scan y scan_delete. 
    Esto es, correr el script una vez.
    Luego cargar list de timetraces malas en "bad_np_list".
    Esos timetraces serán eliminados de la estadística.
    Importante: la carpeta de de las trazas a analizar se carga en "working_folder".
    Dentro de esa carpeta se creará una nueva carpeta llamada "figures" donde
    se guardarán las trazas.
    CARGAR TODOS los inputs.

Mariano Barella marianobarella@gmail.com
Agosto 2018
"""

import os, time, sys, shutil, re
import numpy as np
import matplotlib.pyplot as plt

# INPUTS

calibration_BS = [3.9, -0.0] # slope mW/V, intercept #mW laser rojo

umbral = 1.10
umbral_down = 0.90
t_max = 60 # variable utilizada para filtrar SIN mirar traza por traza
error = 0#0.1 #error tolerable de la rutina de impresión, depende del time_steps 0.02 s y sincronización de eventos
t_max = t_max + error
steps_after_umbral = 40 #PyPrinting por default 10
steps_before_umbral = 40 #PyPrinting por default 10

# LISTA DE TRAZAS FALLADAS
# en caso de que no haya ningún evento "malo" usar la de [9999]
bad_np_list = [6000]
#bad_np_list = [14,9,15]

# para grabar
save_figure = True

# PATH
#prefix_folder = '/run/user/1000/gvfs/smb-share:server=fileserver,share=lmartinez/Luciana doctorado/2020/Mediciones_PyPrinting/'
#local_folder  = '2020-02-05 (AuNPz 60nm sin NaCl4)/20200205-165546_Printing_10x10/'

prefix_folder = '/home/luciana/Cuarentena/2020-03-18 (P2 R5)/'
local_folder = '2020-03-18 (nanostars P2 R5 laser verde)/20200318-115703_Printing_10x10_4mW/'

working_folder = prefix_folder + local_folder

print(working_folder)
#############################################################
#############################################################
#############################################################

plt.close('all')
plt.ioff()

save_folder = working_folder + 'figures_traces/'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

list_of_timetraces = [g for g in os.listdir(working_folder) if os.path.isfile(working_folder + g)]
list_of_timetraces = [g for g in list_of_timetraces if re.search('.txt',g)]
list_of_timetraces = [g for g in list_of_timetraces if re.search('NP',g)]
list_of_timetraces.sort()

print(len(list_of_timetraces))

counter = 1
for g in list_of_timetraces:
        print(str(counter) + ': ' + g)
        counter += 1
print("0: for all timetraces. WARNING: all previous analysis would be deleted.")
time.sleep(1)
keyboard_input = input('Option: ')
user_input  = int(keyboard_input)
if not type(user_input):
    print('Error. Input MUST be an integer. Program closed.')
    sys.exit()
else:
    if user_input == 0:
        print('Are you sure you want to delete all previous analysis?')
        keyboard_input = input('Y/N: ')
        if keyboard_input == 'Y':
            timetraces_list = list_of_timetraces
            if os.path.exists(save_folder):
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
        else:
            print('Run the program again. Closing...')
            sys.exit()
    else:
        timetraces_list = [list_of_timetraces[user_input-1]]
if not timetraces_list:
    print('Error. No timetrace selected. Closing program.')
    sys.exit()


bad_np_array = np.array(['NP%d.txt' % i for i in bad_np_list])

#toda la data menos los bad_np_array
time_step_array = np.array([])
power_bfp_array = np.array([])
list_time_tots_complete = np.array([])
list_umbral_complete = np.array([])
NP_array = np.array([])

#los que cumplieron condicion de Printing por umbrales
list_time_tots = np.array([])
list_umbral = np.array([])

for g in timetraces_list:
    if not np.where(bad_np_array == g)[0].size == 0: continue

    NP = int(g.split('NP_')[-1].split('.txt')[0])
    
    NP_array = np.append(NP_array, NP)

    filename = working_folder + g
#    print g
    #f = open(filename, 'r')
    f = np.loadtxt(filename)
    
    # List all keys
#    print("Keys: %s" % f.keys())
    times = f[:,0]
    voltage = f[:,1]
    power_bfp = f[:,2]*calibration_BS[0] + calibration_BS[1]
    
    # Get data
    N = len(times)
    # Get trace time
    time_tot = times[-1] # in seconds}
    time_step = time_tot/N
    time_step_array = np.append(time_step_array,time_step)
    
    power_bfp_mean = np.mean(power_bfp)
    power_bfp_array = np.append(power_bfp_array,power_bfp_mean)
    
    voltage_after_umbral = np.mean(voltage[-steps_after_umbral:])
    voltage_before_umbral = np.mean(voltage[-(steps_before_umbral+steps_after_umbral):])

    real_umbral = voltage_after_umbral/voltage_before_umbral
    
    events_by_umbral = voltage > real_umbral*voltage_before_umbral
    events_by_umbral_down = voltage < real_umbral*voltage_before_umbral
    
    list_time_tots_complete = np.append(list_time_tots_complete,time_tot)
    list_umbral_complete = np.append(list_umbral_complete, real_umbral)

    if real_umbral > umbral or real_umbral < umbral_down: pass

    print(NP, real_umbral, 'yes')
   
    list_time_tots = np.append(list_time_tots,time_tot)
    list_umbral = np.append(list_umbral, real_umbral)
    
    voltage_ok = voltage[events_by_umbral]
    times_ok = times[events_by_umbral]
    
    voltage_ok_down = voltage[events_by_umbral_down]
    times_ok_down = times[events_by_umbral_down]

    fig, ax = plt.subplots()
#    plt.hold(True)
    ax.plot(times,voltage, 'r.-', label=g)
    ax.plot(times_ok, voltage_ok, 'go')
    ax.plot(times_ok_down, voltage_ok_down, 'bo')
    ax = plt.gca()
    plt.grid(True)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16, labelcolor = 'r')
    ax.set_ylabel(r'Intensity (arb)', fontsize=16, color = 'r')
    ax.set_xlabel(u'Time (s)', fontsize=16)
#    ax.set_ylim([0.05,0.275])

    ax2 = ax.twinx()    
    ax2.plot(times,power_bfp, 'k.-', label=g)
    #ax2 = plt.gca()
    #ax2.grid(True)
    ax2.tick_params(axis='y', which='major', labelsize=16, labelcolor = 'k')
    ax2.set_ylabel(r'Power BFP (mW)', fontsize=16, color = 'k')
#    ax.set_ylim([0.05,0.275])
    plt.legend()
    
    if save_figure:
        fig.savefig(save_folder + '%s.png' % g[:-4], dpi = 100, bbox_inches='tight')
    plt.close() # comentar en caso de querer ver todes les gráficxs 
    
plt.show()

#%%

# Compute mean and std dev
mean_time_tots = np.mean(list_time_tots)
std_time_tots = np.std(list_time_tots,ddof=1)

print('Mean:', mean_time_tots)
print('Std:', std_time_tots)
print('N:', len(list_time_tots))

x = np.arange(0,60,1)
beta = mean_time_tots
y_exp = (1/beta)*np.exp((-x/beta))

fig = plt.figure()
plt.hist(list_time_tots, bins=15, range=[0,60], density = True, rwidth=0.9, color='C0')
plt.plot(x,y_exp, 'C3-', linewidth = 2)
ax = plt.gca()
ax.axvline(beta, color='k', linestyle = '--')
plt.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_ylabel(r'Event frequency', fontsize=16)
ax.set_xlabel(u'Printing time (s)', fontsize=16)
ax.set_axisbelow(True)
if save_figure:
    fig.savefig(save_folder + 'histogram_printing_time.png', dpi = 300, bbox_inches='tight')
plt.close() # comentar en caso de querer ver todes les gráficxs 

fig = plt.figure()
plt.hist(time_step_array*1000, range =[1,40] ,bins=15, rwidth=0.9, color='C0')
ax = plt.gca()
mean_step = np.mean(time_step_array*1000)
std_step = np.std(time_step_array*1000)
ax.axvline(mean_step, color='C3', linestyle = '--', linewidth=2)
plt.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_ylabel(r'Frequency', fontsize=16)
ax.set_xlabel(u'Time step (ms)', fontsize=16)
ax.set_axisbelow(True)
if save_figure:
    fig.savefig(save_folder + 'histogram_time_step.png', dpi = 300, bbox_inches='tight')
plt.close() # comentar en caso de querer ver todes les gráficxs 


fig = plt.figure()
plt.hist(list_umbral, bins=15, rwidth=0.9, color='C0')
ax = plt.gca()
mean_umbral = np.mean(list_umbral)
std_umbral = np.std(list_umbral)
ax.axvline(mean_umbral, color='C3', linestyle = '--', linewidth=2)
plt.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_ylabel(r'Event Frequency', fontsize=16)
ax.set_xlabel(u'Umbral Printing', fontsize=16)
ax.set_axisbelow(True)
if save_figure:
    fig.savefig(save_folder + 'histogram_umbral_printing.png', dpi = 300, bbox_inches='tight')
plt.close() # comentar en caso de querer ver todes les gráficxs 

mean_power_bfp = np.mean(power_bfp_array)
std_power_bfp = np.std(power_bfp_array,ddof=1)

print('Mean:', mean_power_bfp)
print('Std:', std_power_bfp)

beta = mean_power_bfp

fig = plt.figure()
plt.hist(power_bfp_array, bins=15, density = True, rwidth=0.9, color='C0')
#plt.plot(x,y_exp, 'C3-', linewidth = 2)
ax = plt.gca()
ax.axvline(beta, color='r', linestyle = '--')
plt.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_ylabel(r'Frequency', fontsize=16)
ax.set_xlabel(u'Power BFP mean (mW)', fontsize=16)
ax.set_axisbelow(True)
if save_figure:
    fig.savefig(save_folder + 'histogram_Power_BFP_mean.png', dpi = 300, bbox_inches='tight')
plt.close() # comentar en caso de querer ver todes les gráficxs 

fig = plt.figure()
plt.plot(NP_array, power_bfp_array, 'o')
ax = plt.gca()
plt.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel(r'NP', fontsize=16)
ax.set_ylabel(u'Power BFP mean (mW)', fontsize=16)
ax.set_axisbelow(True)
if save_figure:
    fig.savefig(save_folder + 'Power_BFP_mean_vs_NP.png', dpi = 300, bbox_inches='tight')
plt.close() # comentar en caso de querer ver todes les gráficxs 

plt.show()

fig = plt.figure()
plt.plot(NP_array, list_time_tots_complete, 'o')
ax = plt.gca()
plt.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel(r'NP', fontsize=16)
ax.set_ylabel(u'Printing time (s)', fontsize=16)
ax.set_axisbelow(True)
if save_figure:
    fig.savefig(save_folder + 'Printing_time_vs_NP.png', dpi = 300, bbox_inches='tight')
plt.close() # comentar en caso de querer ver todes les gráficxs 

fig = plt.figure()
plt.plot(NP_array, list_umbral_complete, 'o')
ax = plt.gca()
plt.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel(r'NP', fontsize=16)
ax.set_ylabel(u'Umbral Printing', fontsize=16)
ax.set_axisbelow(True)
if save_figure:
    fig.savefig(save_folder + 'Umbral_Printing_vs_NP.png', dpi = 300, bbox_inches='tight')
plt.close() # comentar en caso de querer ver todes les gráficxs 

fig = plt.figure()
plt.plot(power_bfp_array, list_time_tots_complete, 'o')
ax = plt.gca()
plt.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel(r'Power BFP mean (mW)', fontsize=16)
ax.set_ylabel(u'Printing time (s)', fontsize=16)
ax.set_axisbelow(True)
if save_figure:
    fig.savefig(save_folder + 'Printing_time_vs_Power_BFP_mean.png', dpi = 300, bbox_inches='tight')
plt.close() # comentar en caso de querer ver todes les gráficxs 

fig = plt.figure()
plt.plot(power_bfp_array, list_umbral_complete, 'o')
ax = plt.gca()
plt.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel(r'Power BFP mean (mW)', fontsize=16)
ax.set_ylabel(u'Umbral Printing', fontsize=16)
ax.set_axisbelow(True)
if save_figure:
    fig.savefig(save_folder + 'Umbral_Printing_vs_Power_BFP_mean.png', dpi = 300, bbox_inches='tight')
plt.close() # comentar en caso de querer ver todes les gráficxs 

plt.show()

#plt.close('all')

header_txt = 'Mean Printing time (s), STD Printing time (s), Umbral Printing, STD Umbral Printing, N, Mean Time Steps (ms), STD Time Steps (ms), Mean Power BFP mean (mW), STD Mean BFP mean (mW), N'
data = np.array([mean_time_tots, std_time_tots, mean_umbral, std_umbral, len(list_time_tots), mean_step, std_step, mean_power_bfp, std_power_bfp, len(power_bfp_array)]).T
name = os.path.join(save_folder, 'Histogram_data.txt')             
np.savetxt(name, data, header = header_txt)