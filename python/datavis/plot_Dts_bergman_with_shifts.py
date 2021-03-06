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
	edge_length = [10.0] #[5.0, 10.0, 20.0, 40.0, 100.0]
	radius = [5.0] #[2.5, 5.0, 10.0, 20.0, 50.0]
	true_phi = [0.448] #[0.352, 0.448, 0.472, 0.476, 0.476]
	number_of_lengths = len(edge_length)
	walker_strings = [r'1M']
	time_scales = [edge**2 / Dp for edge in edge_length] 
	phi = 0.47
	relaxation_strength = 0.0
	rho = [relaxation_strength*Dp/edge for edge in edge_length]
	resolution = 1.0
	# shift = 0
	shifts = [0, 1, 2, 3, 4]
	step_sizes = [(resolution/(2**shift)) for shift in shifts]
	number_of_divisions = len(shifts)

	# Database dir
	db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/bergman_test/'
	db_dir += r'Dt_rho=' + str(relaxation_strength) + r'/'
	db_dir += r'phi=' + str(phi) + r'_res=' + str(resolution) + r'/'

	for walkers in walker_strings:
		# Plot title
		line1 = r'$\bf{Periodic Array}$: resolution = ' + str(resolution) + r' $\mu$m/voxel, a = ' + str(edge_length[0]) + r'$\mu m$'
		line2 = r'$\rho a/D$= ' + str(relaxation_strength) + r' $\mu$m, D = ' + str(Dp)  + r' $\mu$m²/s, walkers=' + walkers
		plot_title =  line1 + '\n' + line2

		# find data files 
		complete_paths = []
		for idx in range(number_of_lengths):
			edge_data = []
			for shift in shifts:
				sim_dir = r'a=' + str(edge_length[idx]) + r'um_r=' + str(radius[idx]) + r'um/data/'
				sim_dir += r'PFGSE_NMR_periodic_a=' + str(edge_length[idx]) + r'_r=' + str(radius[idx]) 
				sim_dir += r'_rho=' + str(rho[idx]) + r'_res=' + str(resolution) + r'_shift=' + str(shift) + r'_w='+ walkers + r'/'

				edge_data.append(db_dir + sim_dir)
			# assembly complete path to data directory
			complete_paths.append(edge_data)	

		# List all data files in directory
		dataset_files = []
		for complete_path in complete_paths:
			for div in range(number_of_divisions):
				exp_dirs = os.listdir(complete_path[div])
				number_of_sim_files = len(exp_dirs)
				echoes_files = [complete_path[div] + exp_dir + '/PFGSE_echoess.txt' for exp_dir in exp_dirs]
				params_files = [complete_path[div] + exp_dir + '/PFGSE_parameters.txt' for exp_dir in exp_dirs]

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
			for div in range(number_of_divisions):
				rw_times = []
				rw_D_sat = []
				rw_D_msd = []
				for time_dir in dataset_files[edge_idx * number_of_divisions + div]:
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

		# Plot msd
		# plt.rcParams.update({'font.size': 16})
		plt.figure(figsize=[9.6, 7.2])
		plt.hlines(0.722, 0.5, 4.0, linestyle='dashed', color='black', label=r'$D_{e}/D_{p}=0.722$ ($\phi$ = 0.476)')
		for edge_idx in range(number_of_lengths):
			for div in range(number_of_divisions):
				idx = edge_idx * number_of_divisions + div
				new_label = r'$\phi_{img}$ = ' +  str(true_phi[edge_idx]) + ', step = ' + str(step_sizes[div]) + ' $\mu m$'
				plt.plot(rw_wrap[idx]['t'], rw_wrap[idx]['D_msd'], 'x', label='msd: ' + new_label)
		plt.legend(loc='upper right')
		plt.title(plot_title)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.3, 1.0])	
		plt.ylabel(r'$D(t)/D_{0}$')
		plt.xlabel(r'$(D_{0}\,t / a^{2})^{1/2}$')
		plt.show()

		
		# Plot S&T
		plt.figure(figsize=[9.6, 7.2])
		plt.hlines(0.722, 0.5, 4.0, linestyle='dashed', color='black', label=r'$D_{e}/D_{p}=0.722$ ($\phi$ = 0.476)')
		for edge_idx in range(number_of_lengths):
			for div in range(number_of_divisions):
				idx = edge_idx * number_of_divisions + div
				new_label = r'$\phi_{img}$ = ' +  str(true_phi[edge_idx]) + ', step = ' + str(step_sizes[div]) + ' $\mu m$'
				plt.plot(rw_wrap[idx]['t'], rw_wrap[idx]['D_sat'], 'x', label=new_label)
		plt.legend(loc='lower right')
		plt.title(plot_title)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])	
		plt.ylabel(r'$D(t)/D_{0}$')
		plt.xlabel(r'$(D_{0}\,t / a^{2})^{1/2}$')
		plt.show()

		# subplot msd and S&T
		rows = 1
		cols = 2
		fig, axs = plt.subplots(rows, cols, figsize=(16.2, 7.2)) 
		fig.suptitle(plot_title)
		axs[0].hlines(0.722, 0.5, 4.0, linestyle='dashed', color='black', label=r'$D_{e}/D_{p}=0.722$ ($\phi$ = 0.476)')
		axs[1].hlines(0.722, 0.5, 4.0, linestyle='dashed', color='black', label=r'$D_{e}/D_{p}=0.722$ ($\phi$ = 0.476)')
		for edge_idx in range(number_of_lengths):
			for div in range(number_of_divisions):
				idx = edge_idx * number_of_divisions + div
				new_label = r'$\phi_{img}$ = ' +  str(true_phi[edge_idx]) + ', step = ' + str(step_sizes[div]) + ' $\mu m$'
				axs[0].set_title("Mean Squared Displacement (msd)")
				axs[1].set_title("Stejskal and Tanner (s&t)")

				axs[0].plot(rw_wrap[idx]['t'], rw_wrap[idx]['D_msd'], 'x', label=new_label)
				axs[1].plot(rw_wrap[idx]['t'], rw_wrap[idx]['D_sat'], 'x', label=new_label)
		for ax in axs:	
			ax.legend(loc='upper right')
			ax.set_xlim([0.0, 1.2])
			ax.set_ylim([0.3, 1.2])	
			ax.set_ylabel(r'$D(t)/D_{0}$')
			ax.set_xlabel(r'$(D_{0}\,t / a^{2})^{1/2}$')
		
		plt.tight_layout()
		plt.show()


	return
			

if __name__ == '__main__':
	main()