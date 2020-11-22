import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_data import *
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import *

def main():
	# simulation parameters
	walkers_str = r'10k'
	edge_length = [1.0, 2.5, 5.0, 10.0, 20.0]
	edge_length_str = ['1um', '2.5um', '5um', '10um', '20um']
	rho = 0.05
	relaxation_strength = 0
	Dfree = 2.5
	res = 1.0

	# Plot title
	line1 = r'$\bf{Spherical Pore}$: resolution = ' + str(res) + r' $\mu$m/voxel'
	line2 = r'$\rho a/D$= ' + str(relaxation_strength) + r' $\mu$m, D = ' + str(Dfree)  + r' $\mu$m²/s, walkers=' + walkers_str
	plot_title =  line1 + '\n' + line2

	# Set simulation data files
	sim_data_dir = []
	for edge in edge_length_str:
		sim_data_dir.append(r"/home/matheus/Documentos/doutorado_ic/tese/saved_data/callaghan_test/relaxation_strength="+ str(relaxation_strength) +"/a="+ edge + r"/isolated_sphere/using_q/res=" + str(res) + r"/")

	sim_experiment_dir = []
	for edge in edge_length:
		rho_sim = (1000.0 * relaxation_strength * Dfree)/edge
		sim_experiment_dir.append(r"PFGSE_NMR_sphere_r=" + str(edge) + r"_rho=" + str(rho_sim) + r"_res=1.0_shift=0_w=" + walkers_str + r"/")

	sim_dirs = []
	sim_datafiles = []
	for dataset in range(len(edge_length)):
		sim_dirs.append(sim_data_dir[dataset] + sim_experiment_dir[dataset])
		sim_echoes_file = r"consoleLog"
		filepath = sim_dirs[dataset] + sim_echoes_file
		if(os.path.isfile(filepath)):
			sim_datafiles.append(filepath)
		

	# print("sim_dirs = \n", sim_dirs)
	# print("sim_datafiles = \n", sim_datafiles)

	# read data from consoleLog files
	console_data = []
	for file in sim_datafiles:
		console_data.append(read_console_log_data(file))

	# for idx in range(len(console_data)):
	# 	print("console data for a = {}: \n {}".format(edge_length[idx], console_data[idx]))

	# estimate pore sizes
	pore_sizes = []
	for idx in range(len(console_data)):
		line = []
		for data in console_data[idx]["S_volume"]:
			line.append(3.0/data)
		pore_sizes.append(line)

	# for row in range(len(pore_sizes)):
	# 	print(pore_sizes[row])

	# # normalize data
	# for i in range(len(edge_length)):
	# 	for j in range(len(pore_sizes[i])):
	# 		pore_sizes[i][j] = pore_sizes[i][j]/edge_length[i]

	# print('after normalization:')
	# for row in range(len(pore_sizes)):
	# 	print(pore_sizes[row])

	bar_data = []
	for col in range(len(pore_sizes[0])):
		datarow = []
		for row in range(len(pore_sizes)):
			datarow.append(pore_sizes[row][col])
		bar_data.append(datarow)

	# print(bar_data)

	bar_labels = []
	bar_labels.append(r"$\Delta$ = 0.2 a²/D")
	bar_labels.append(r"$\Delta$ = 0.5 a²/D")
	bar_labels.append(r"$\Delta$ = 1.0 a²/D")
	bar_labels.append(r"$\Delta$ = 2.0 a²/D")

	tick_labels = []
	for edge in edge_length:
		tick_labels.append(str(edge) + r" $\mu$m")
	barplot(bar_data, bar_labels, tick_labels)


	return

if __name__ == '__main__':
 	main()

 	# dirpath =  r"/home/matheus/Documentos/Doutorado IC/tese/saved_data/callaghan_test/relaxation_strength=0/a=2.5um/isolated_sphere/using_q/res=1.0/"
 	# datadir = r"PFGSE_NMR_sphere_r=2.5_rho=0.0_res=1.0_shift=0_w=10M/"
 	# filename = r"consoleLog"
 	# filepath = dirpath + datadir + filename
 	# data = read_console_log_data(filepath)
 	# print(data)