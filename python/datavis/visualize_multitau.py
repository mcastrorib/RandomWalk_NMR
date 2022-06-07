import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def magnetization(D, gamma, grad, te):
	return np.exp((-1.0/12.0)*(gamma*grad)**2*D*te**3) 

DB_DIR = './db/'
SIM_DIR = ['NMR_Simulation_slab_r=20.0um_res=0.1um_f=5.0Tm_axis=z_shift=1_bc=noflux',
           'NMR_Simulation_slab_r=20.0um_res=0.1um_f=5.0Tm_axis=z_shift=1_bc=periodic',
           'NMR_Simulation_slab_r=20.0um_res=0.1um_f=5.0Tm_axis=z_shift=1_bc=mirror']

labels = [r'no flux', r'periodic', r'mirror']
colors = ['blue', 'green', 'orange']

DATA_DIR = 'NMR_multitau_min=0.01ms_max=1.00ms_pts=10_scale=log'
FILENAME = 'multitau_decay.csv'

fig, axs =  plt.subplots(1, 1, figsize=(10.0, 7.0), constrained_layout=True)

for sim, label, color in zip(SIM_DIR, labels, colors):
	sim_path = os.path.join(DB_DIR, sim, DATA_DIR, FILENAME)
	df = pd.read_csv(sim_path)
	t_echo = 1e-3*df['echo_time'].values
	signal = df['signal'].values
	axs.scatter(t_echo, signal, marker='s', color=color, s=50, label=label)

D0 = 2.5e-09
gamma = 2.67e08 
grad = 5.0
te = np.logspace(np.log10(0.008e-3), np.log10(1.1e-3), 1000)
analytical_signal = magnetization(D0, gamma, grad, te)
axs.plot(te, analytical_signal, 'r--', label="analytical")

axs.set_xscale('log')
axs.set_xlabel('Time (s)')
# axs.set_xlim(0,0.03)

axs.set_yscale('log')
axs.set_ylabel('Normalized amplitude')
# axs.set_ylim(0.01,1.0)

axs.legend(loc='best', frameon=False)
plt.show()