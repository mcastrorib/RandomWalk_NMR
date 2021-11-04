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
	Dp = 2.5
	edge_length = [20.0] # [200.0, 100.0, 40.0, 20.0, 10.0, 5.0]
	radius = [10.0] # [130.0, 65.0, 26.0, 13.0, 6.5, 3.25]
	true_phi = [0.472] # [0.104, 0.104, 0.104, 0.098, 0.104, 0.064]
	number_of_lengths = len(edge_length)
	walker_strings = [r'100k']
	time_scales = [edge**2 / Dp for edge in edge_length] 
	phi = 0.476
	De_bergman = 0.722
	relaxation_strength = 0.0
	rho = [relaxation_strength*Dp/edge for edge in edge_length]
	resolution = 1.0
	shift = 0
	markers = ['.', 'x', 's', 'v', '^', 'd']

	# Database dir
	# db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/bergman_test/'
	# db_dir += r'Dt_rho=' + str(relaxation_strength) + r'/'
	# db_dir += r'phi=' + str(phi) + r'_res=' + str(resolution) + r'/'
	
	# db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/bergman_test/new_version_tests/tests_periodic_bcs/'
	
	# db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/tortuosity_tests/tests_periodic/'
	# db_dir += r'phi=' + str(phi) + r'/'

	db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/db/'
	
	for walkers in walker_strings:
		# Plot title
		line1 = r'$\bf{Periodic Array}$ ($\phi$ = ' + str(phi) + '): resolution = ' + str(resolution) + r' $\mu$m/voxel'
		line2 = r'$\rho a/D$= ' + str(relaxation_strength) + r' $\mu$m, D = ' + str(Dp)  + r' $\mu$m²/s, walkers=' + walkers
		plot_title =  line1 + '\n' + line2

		# find data files 
		complete_paths = []
		for idx in range(number_of_lengths):
			# sim_dir = r'a=' + str(edge_length[idx]) + r'um_r=' + str(radius[idx]) + r'um/data/'
			sim_dir = r'PFGSE_NMR_periodic_a=' + str(edge_length[idx]) + r'_r=' + str(radius[idx]) 
			sim_dir += r'_rho=' + str(rho[idx]) + r'_res=' + str(resolution) + r'_shift=' + str(shift) + r'_w='+ walkers # + r'/'
			complete_paths.append(db_dir + sim_dir + '/')

		print(complete_paths)

		# List all data files in directory
		dataset_files = []
		for complete_path in complete_paths:
			exp_dirs = os.listdir(complete_path)
			number_of_sim_files = len(exp_dirs)
			echoes_files = [complete_path + exp_dir + '/PFGSE_echoes.txt' for exp_dir in exp_dirs]
			params_files = [complete_path + exp_dir + '/PFGSE_parameters.txt' for exp_dir in exp_dirs]

			data_files = []
			for i in range(number_of_sim_files):
				data_files.append([params_files[i], echoes_files[i]])

			# check if path exists
			for file in data_files:
				if(os.path.isfile(file[0]) and os.path.isfile(file[1])):
					print('FILES ' + file[0] + ' AND ' + file[1] + ' FOUND.')
				else:
					print('FILE ' + file[0] + ' AND ' + file[1] + ' NOT FOUND. Removed from list')
					data_files.remove(file)
	
			number_of_sim_files = len(data_files)
			dataset_files.append(data_files)

		rw_wrap = []
		for edge_idx in range(number_of_lengths):
			rw_times = []
			rw_D_sat = []
			rw_D_msd = []
			for time_dir in dataset_files[edge_idx]:
				rw_data = read_pfgse_data_from_rwnmr_file(time_dir)
				rw_times.append(np.sqrt(rw_data['delta'] / time_scales[edge_idx]))
				rw_D_sat.append(rw_data['D_sat'] / Dp)
				rw_D_msd.append(rw_data['D_msd'] / Dp)

			new_wrap = {
				't': rw_times,
				'D_sat': rw_D_sat,
				'D_msd': rw_D_msd
			}
			rw_wrap.append(new_wrap)

		# Plot 
		# plt.rcParams.update({'font.size': 14})
		plt.figure(figsize=[9.6, 7.2])
		plt.hlines(De_bergman, 0.8, 4.0, linestyle='dashed', color='black', label=r'$D_{e}/D_{p}=$'+str(De_bergman))
		for idx in range(number_of_lengths):
			new_label = 'a =' + str(edge_length[idx]) + '$\mu m$ ($\phi_{img}$ = ' +  str(true_phi[idx]) + ')'
			plt.plot(rw_wrap[idx]['t'], rw_wrap[idx]['D_msd'], markers[idx], label=new_label)
			# plt.plot(rw_wrap[idx]['t'], rw_wrap[idx]['D_sat'], markers[idx], label=new_label)
		plt.legend(loc='best')
		plt.title(plot_title)
		plt.xlim([0.0, 3.5])
		plt.ylim([0.0, 1.0])	
		plt.ylabel(r'$D(t)/D_{0}$')
		plt.xlabel(r'$(D_{0}\,t / a^{2})^{1/2}$')
		plt.show()
				

	return
			

if __name__ == '__main__':
	main()