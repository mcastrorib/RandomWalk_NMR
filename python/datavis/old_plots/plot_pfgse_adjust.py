import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_data import NMR_data
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import NMR_PlotProperties
from LeastSquaresRegression import LeastSquaresRegression

def main():
	# simulation parameters
	res = 1.0
	Dfree = 2.5
	
	walker_str = r'1M'
	edge_length = 5.0
	edge_str = '5'
	
	relaxation_strength = 0
	rho_analytic = 0.0
	rho_sim = (1000.0 * relaxation_strength * Dfree)/edge_length

	# Plot title
	line1 = r'$\bf{Cylindrical Pore}$: resolution = ' + str(res) + r' $\mu$m/voxel'
	line2 = r'$\rho a/D$= ' + str(relaxation_strength) + r', a = ' + str(edge_length) + r'$\mu$m, D = ' + str(Dfree)  + r' $\mu$m²/s, walkers=' + walker_str
	plot_title =  line1 + '\n' + line2

	# Set simulation data files
	sim_data_dir = r"/home/matheus/Documentos/doutorado_ic/tese/saved_data/callaghan_test/relaxation_strength="+ str(relaxation_strength) +"/a=" + edge_str + r"um/isolated_cylinder/using_q/res=1.0"
	sim_experiment_dir = r"/PFGSE_NMR_cylinder_r=" + str(edge_length) + r"_rho=" + str(rho_sim) + r"_res=1.0_shift=0_w=" + walker_str
	sim_dir = sim_data_dir + sim_experiment_dir
	pfgse_echoes_file = r"/PFGSE_echoes.txt"
	
	# collect via os list
	dirlist = [ item for item in os.listdir(sim_dir) if os.path.isdir(os.path.join(sim_dir, item)) ]
	sim_datafiles = []
	for item in dirlist:
		filepath = r'/' + item + pfgse_echoes_file
		print(filepath)
		if(os.path.isfile(sim_dir + filepath)):
			sim_datafiles.append(sim_dir + filepath)


	# read data from PFGSE echoes files
	pfgse_data = []
	for file in sim_datafiles:
		pfgse_data.append(read_data_from_pfgse_echoes_file(file))

	# read data from Console log
	sim_consolelog_files = []
	log_file = r"/consoleLog"
	filepath = sim_dir + log_file
	if(os.path.isfile(filepath)):
		sim_consolelog_files.append(filepath)
	
	# read data from consoleLog files
	consolelog_data = []
	for file in sim_consolelog_files:
		consolelog_data.append(read_console_log_data(file))

	for dataset in range(len(pfgse_data)):
		# create and solve least square regression
		lsa = LeastSquaresRegression()
		lsa_threshold = 0.8
		lsa_points = count_points_to_apply_lhs_threshold(pfgse_data[dataset]["lhs"], lsa_threshold)
		lsa.config(pfgse_data[dataset]["rhs"], pfgse_data[dataset]["lhs"], lsa_points)
		lsa.solve()
		lsa_results = lsa.results()
		D_sat = lsa_results["B"]
		lsa_title = plot_title


		print("t= {:.2f}, D(t)= {:.4f}".format(pfgse_data[dataset]["delta"], D_sat))

		# plot adjust
		plot_least_squares_adjust(
			pfgse_data[dataset]["rhs"], 
			pfgse_data[dataset]["lhs"], 
			D_sat, pfgse_data[dataset]["delta"], 
			lsa_points, 
			lsa_threshold, 
			lsa_title)

	return

if __name__ == '__main__':
	main()