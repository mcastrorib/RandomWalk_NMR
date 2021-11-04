import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_ReadFromFile import *

def read_collision_histogram(filepath, scale):
	xi_rates = []
	xi_occurs = []

	with open(filepath, 'r') as file:
		file_lines = file.readlines()
		for line in file_lines:
			data = line.split('\t')
			xi_rates.append(float(data[0]))
			xi_occurs.append(float(data[1]))

	if(scale == 'log'):
		xi_rates[0] = pow(10, xi_rates[0])

	collision_hist = {
		'rates': np.array(xi_rates),
		'occurs': np.array(xi_occurs)
	}
	return collision_hist

def compress_histogram(histogram, factor, scale):
	rebalance = 0
	if(scale == 'log'):
		rebalance += 1

	if((histogram['rates'].shape[0] - rebalance) % factor == 0):
		size = (histogram['rates'].shape[0] - rebalance) // factor
		size += rebalance

		rates = np.zeros(size)
		occurs = np.zeros(size)

		if(scale == 'log'):
			rates[0] = histogram['rates'][0]
			occurs[0] = histogram['occurs'][0]

			for idx in range(size-1):
				begin = idx*factor + 1
				rates[idx+1] = histogram['rates'][begin + factor - 1]
				occurs[idx+1] = histogram['occurs'][begin:(begin + factor)].sum()
		else:
			for idx in range(size):
				begin = idx*factor
				rates[idx] = histogram['rates'][begin:(begin + factor)].sum() / factor
				occurs[idx] = histogram['occurs'][begin:(begin + factor)].sum()

		compressed_histogram = {
			'rates': rates,
			'occurs': occurs
		}

		return compressed_histogram
	else:
		print('Compression factor is not valid')
		return None


dbdir = '/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/db/xirate_tests'
filename = 'NMR_histogram.txt'
rtag = 'EW'

filedirs = []
filedirs.append(rtag + '_shift=0/' + rtag + '-1.0_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=0_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=1/' + rtag + '-1.0_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=1_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=2/' + rtag + '-1.0_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=2_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=3/' + rtag + '-1.0_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=3_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')

filedirs.append(rtag + '_shift=0/' + rtag + '-0.5_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=0_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=1/' + rtag + '-0.5_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=1_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=2/' + rtag + '-0.5_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=2_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=3/' + rtag + '-0.5_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=3_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')

filedirs.append(rtag + '_shift=0/' + rtag + '-0.2_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=0_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=1/' + rtag + '-0.2_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=1_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=2/' + rtag + '-0.2_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=2_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=3/' + rtag + '-0.2_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=3_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')

filedirs.append(rtag + '_shift=0/' + rtag + '-0.1_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=0_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=1/' + rtag + '-0.1_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=1_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=2/' + rtag + '-0.1_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=2_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=3/' + rtag + '-0.1_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=3_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')

filedirs.append(rtag + '_shift=0/' + rtag + '0.0_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=0_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=1/' + rtag + '0.0_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=1_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=2/' + rtag + '0.0_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=2_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')
filedirs.append(rtag + '_shift=3/' + rtag + '0.0_logmaptimes_PFGSE_NMR_RTAG=DP_res=0.91_rho=0.0_shift=3_a=10.0_w=100k_ws=100_bc=mirror/NMR_pfgse_timesample_0')


tags = ['shift=0', 'shift=1', 'shift=2', 'shift=3']
times = ['4 ms', '13 ms', '25 ms', '33 ms', '40 ms']
labels = []
for time in times:
	for tag in tags:
		labels.append(r'$\Delta=$'+time+', '+tag)

samples = 5
shifts = 4
scale = samples * shifts * ['log']
compress_factor = np.array(samples * shifts * [4])

cm_hists = []
for idx in range(len(filedirs)):
	complete_path = os.path.join(dbdir, filedirs[idx], filename)
	new_hist = read_collision_histogram(complete_path, scale[idx])

	
	cm_hists.append(compress_histogram(new_hist, compress_factor[idx], scale[idx]))
	# cm_hists.append(new_hist)

rows = samples
cols = shifts
fig, axs = plt.subplots(rows, cols,figsize=[cols*4.5, rows*3.0])
for row in range(rows):
	for col in range(cols):
		idx = row*cols + col

		widths = np.zeros(cm_hists[idx]['rates'].shape[0])
		offsets = np.zeros(cm_hists[idx]['rates'].shape[0])
		if(scale[idx] == 'log'):
			for widx in range(cm_hists[idx]['rates'].shape[0] - 1):
				widths[widx] = 0.9*(cm_hists[idx]['rates'][widx+1] - cm_hists[idx]['rates'][widx])
				# offsets[widx] = 0.5*widths[widx]
			widths[-1] = widths[-2]
			widths[0] *= 0.03 
		else:
			for widx in range(cm_hists[idx]['rates'].shape[0]):
				widths[widx] = (cm_hists[idx]['rates'][1] - cm_hists[idx]['rates'][0])
		
		if(scale[idx] == 'log'):
			new_label = r'$\xi=0$: ' + '{:.2}'.format(cm_hists[idx]['occurs'][0])
			axs[row][col].bar((cm_hists[idx]['rates'] - offsets)[1:], cm_hists[idx]['occurs'][1:], widths[1:], label=new_label)
			axs[row][col].set_xscale('log')
			axs[row][col].set_xlim([0.99*cm_hists[idx]['rates'][0], 1.0])
			axs[row][col].set_ylim([0.0, 1.5*cm_hists[idx]['occurs'][1:].max()])
			axs[row][col].legend(loc='upper left')

		else:
			axs[row][col].bar((cm_hists[idx]['rates'] - offsets), cm_hists[idx]['occurs'], widths)
			axs[row][col].set_xlim([0,1])

		new_title = labels[idx]
		axs[row][col].set_title(new_title, y=0.95, x=0.75, pad=-16) #, size=16)

plt.tight_layout()
plt.show()