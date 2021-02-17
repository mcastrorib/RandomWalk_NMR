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
	walkers_str = r'10k'
	edge_length = [10.0] #, 2.5, 5.0, 10.0, 20.0]
	edge_length_str = ['10um'] #, '2.5um', '5um', '10um', '20um']
	pore_type = 'isolated_sphere'
	rho = 0.0
	relaxation_strength = 0
	Dfree = 2.50
	res = 1.0
	use_raw_data = True

	# Root dir
	# root_dir = r"/home/matheus/Documentos/doutorado_ic/tese/saved_data/callaghan_test"
	root_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/free_diffusion'

	# Plot title
	line1 = r'$\bf{Free media}$: resolution = ' + str(res) + r' $\mu$m/voxel'
	line2 = r'$\rho a/D$= ' + str(relaxation_strength) + r' $\mu$m, D = ' + str(Dfree)  + r' $\mu$m²/s, walkers=' + walkers_str
	plot_title =  line1 + '\n' + line2

	# Set simulation data files
	sim_data_dir = []
	for edge in range(len(edge_length_str)):
		rho_sim = (1000.0 * relaxation_strength * Dfree)/edge_length[edge]
		
		# Callaghan test
		# sim_data_dir = root_dir + r"/relaxation_strength="+ str(relaxation_strength) +"/a="+ str(edge_length_str[edge]) + r"/" + pore_type + "/using_q/res=" + str(res)
		# sim_experiment_dir = r"/PFGSE_NMR_sphere_r=" + str(edge_length[edge]) + r"_rho=" + str(rho_sim) + r"_res=1.0_shift=0_w=" + walkers_str
		
		# Free media
		sim_data_dir = root_dir + r"/Dt_recover"
		sim_experiment_dir = r"/PFGSE_NMR_FreeMedia_rho=" + str(rho_sim) + r"_res=1.0_shift=0_w=" + walkers_str
		
		sim_dir = sim_data_dir + sim_experiment_dir
		
		if (use_raw_data):
			# collect via os list
			pfgse_echoes_filename = r"/PFGSE_echoes.txt"
			dirlist = [ item for item in os.listdir(sim_dir) if os.path.isdir(os.path.join(sim_dir, item)) ]
			pfgse_echoes_files = []
			for item in dirlist:
				filepath = r'/' + item + pfgse_echoes_filename
				if(os.path.isfile(sim_dir + filepath)):
					pfgse_echoes_files.append(sim_dir + filepath)
			
			# read data from PFGSE echoes files
			pfgse_data = []
			for file in pfgse_echoes_files:
				pfgse_data.append(read_data_from_pfgse_echoes_file(file))

			# Get 'delta' times
			D_delta = []
			for dataset in range(len(pfgse_data)):
				D_delta.append(pfgse_data[dataset]["delta"])

			# Get D_msd
			D_msd = []
			for dataset in range(len(pfgse_data)):
				D_msd.append(pfgse_data[dataset]["diffusion_coefficient"])
			
			# Get D_sat
			D_sat = []
			for dataset in range(len(pfgse_data)):
				# apply threshold for least squares adjust
				lsa_threshold = 0.8
				lhs_min = min(pfgse_data[dataset]["lhs"])
				
				while(np.log(lsa_threshold) < lhs_min):
					lsa_threshold += 0.1 

				lsa_points = count_points_to_apply_lhs_threshold(pfgse_data[dataset]["lhs"], lsa_threshold)
				print("threshold = ", lsa_threshold)
				print("points = ", lsa_points)

				# create and solve least square regression
				lsa = LeastSquaresRegression()		
				lsa.config(pfgse_data[dataset]["rhs"], pfgse_data[dataset]["lhs"], lsa_points)
				lsa.solve()
				D_sat.append(lsa.results()["B"])
				# lsa_results = lsa.results()
				# D_sat = lsa_results["B"]
				print("t= {:.2f}, D(t)= {:.4f}".format(pfgse_data[dataset]["delta"], D_sat[dataset]))

				# # plot adjust
				# lsa_title = plot_title
				# plot_least_squares_adjust(
				# 	pfgse_data[dataset]["rhs"], 
				# 	pfgse_data[dataset]["lhs"], 
				# 	D_sat[dataset], 
				# 	pfgse_data[dataset]["delta"], 
				# 	lsa_points, 
				# 	lsa_threshold, 
				# 	lsa_title)

			# plot data
			t_adim = []
			for time in D_delta:
				t_adim.append(np.sqrt(time * Dfree))

			labels = ["S&T", "msd"]
			x_data = t_adim
			y_data = [D_sat, D_msd]
			markers = ['o', 'x']
			title = plot_title + '\n' + r'a = ' + str(edge_length[edge]) + r'$\mu$m'
			scatterplot(x_data, y_data, labels, markers, title) 

		else:
			# check if consoleLog file exists
			filename = r"consoleLog"
			filepath = sim_dir + filename
			datafile = ''
			if(os.path.isfile(filepath)):
				datafile = filepath
					
			# read data from consoleLog files
			console_data = read_console_log_data(datafile)
			print("console data for a = {}: \n {}".format(edge_length[edge], console_data))

			t_adim = []
			for time in console_data["D_times"]:
				t_adim.append(np.sqrt(time*Dfree))

			# plot data
			labels = ["S&T", "msd"]
			x_data = t_adim
			y_data = [console_data["D_sat"], console_data["D_msd"]]
			markers = ['o', 'x']
			title = plot_title + '\n' + r'a = ' + str(edge_length[edge]) + r'$\mu$m'
			scatterplot(x_data, y_data, labels, markers, title) 

	return

if __name__ == '__main__':
 	main()

 	# dirpath =  r"/home/matheus/Documentos/Doutorado IC/tese/saved_data/callaghan_test/relaxation_strength=0/a=2.5um/isolated_sphere/using_q/res=1.0/"
 	# datadir = r"PFGSE_NMR_sphere_r=2.5_rho=0.0_res=1.0_shift=0_w=10M/"
 	# filename = r"consoleLog"
 	# filepath = dirpath + datadir + filename
 	# data = read_console_log_data(filepath)
 	# print(data)