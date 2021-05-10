import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_ReadFromFile import *
from LeastSquaresRegression import LeastSquaresRegression

def main():
	# simulation parameters
	Dp = 2.5
	inspection_length = 10.0
	sphere_radius = 5.0
	phi = 0.476 
	resolution = 1.0
	walker_pop = r'10M'
	walker_samples = r'100'
	walkers_per_sample = r'100k'
	rhos = [0.0, 1.0]
	tort_lims = [0.72, 0.68]
	shifts = [0, 1, 2] 
	step_size = [resolution/2**(shift) for shift in shifts] 
	
	colors = [['pink', 'magenta', 'red'],['gray', 'blue', 'navy']] 
	markers = ['.', '.'] # ['v', 'x', 'o', '^']
	
	SQRT_TIME = False
	ADIM_TIME = True
	PLOT_TORT_LIM = False

	# size variables
	nshifts = len(shifts)
	nrhos = len(rhos)

	# Database dir
	# db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/bergman_test/'
	# db_dir += r'Dt_rho=' + str(relaxation_strength) + r'/'
	# db_dir += r'phi=' + str(phi) + r'_res=' + str(resolution) + r'/'

	# db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/bergman_test/new_version_tests/tests_periodic_bcs/'

	db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/'
	db_dir += r'saved_data/bergman_test/Dt_rho=var/'
	db_dir += r'phi=' + str(phi) + r'_res=' + str(resolution) + r'/data/'


	# PFGSE_NMR_random_pack_phi=high_a=100.0_rho=0.0_res=1.0_shift=0_w=10k_cell

	# Plot title
	line1 = r'$\bf{Periodic \, Array}$' + r' ($\phi =$' + str(phi) + r')' 
	line2 = r'a = ' + str(inspection_length) + r' $\mu m$, '
	# line2 += r'$\epsilon =$ ' + str(step_size[0]) + r' $\mu m$, '
	line2 += 'resolution = ' + str(resolution) + r' $\mu$m/voxel'
	line3 =  r'$D_{0}$ = ' + str(Dp)  + r' $\mu$m²/s, walkers = ' + walkers_per_sample
	plot_title =  line1 + '\n' + line2 + '\n' + line3

	# find data files 
	complete_paths = []
	for rho in rhos:
		for shift in shifts:
			# sim_dir = r'a=' + str(edge_length[idx]) + r'um_r=' + str(radius[idx]) + r'um/data/'
			sim_dir = r'PFGSE_NMR_periodic'
			sim_dir += r'_a=' + str(inspection_length) 
			sim_dir += r'_r=' + str(sphere_radius) 
			sim_dir += r'_rho=' + str(1e3*rho*Dp/inspection_length) 
			sim_dir += r'_res=' + str(resolution) 
			sim_dir += r'_shift=' + str(shift) 
			sim_dir += r'_w='+ walker_pop
			sim_dir += r'_ws=' + walker_samples
			sim_dir += r'/'

			complete_paths.append(db_dir + sim_dir)

	# List all data files in directory
	dataset_files = []
	for complete_path in complete_paths:
		exp_dirs = os.listdir(complete_path)
		number_of_sim_files = len(exp_dirs)
		echoes_files = [complete_path + exp_dir + '/PFGSE_echoes.txt' for exp_dir in exp_dirs]
		params_files = [complete_path + exp_dir + '/PFGSE_parameters.txt' for exp_dir in exp_dirs]
		msd_files = [complete_path + exp_dir + '/PFGSE_msd.txt' for exp_dir in exp_dirs]

		data_files = []
		for i in range(number_of_sim_files):
			data_files.append([params_files[i], echoes_files[i], msd_files[i]])

		# check if path exists
		for file in data_files:
			if not (os.path.isfile(file[0]) and os.path.isfile(file[1]) and os.path.isfile(file[2])):
				# print('FILES ' + file[0] + ', ' + file[1] + ' AND ' + file[2] + ' FOUND.')
			# else:
				# print('FILE ' + file[0] + ', ' + file[1] + ' AND ' + file[2] + ' NOT FOUND. Removed from list')
				data_files.remove(file)

		number_of_sim_files = len(data_files)
		dataset_files.append(data_files)
	
	rw_wrap = []
	for i in range(nrhos):
		for j in range(nshifts):
			idx = i*nshifts + j 

			rw_times = []
			rw_D_sat = []
			rwDmsdX = []
			rwDmsdX_stdev = []
			rwDmsdY = []
			rwDmsdY_stdev = []
			rwDmsdZ = []
			rwDmsdZ_stdev = []
			rwDmsdMean = []
			rwDmsdStdDev = []

			for time_dir in dataset_files[idx]:
				rw_data = read_pfgse_data_from_rwnmr_file(time_dir)
				if(SQRT_TIME):
					rw_times.append(np.sqrt(rw_data['delta']))
				elif(ADIM_TIME):
					rw_times.append(np.sqrt(rw_data['delta'] * Dp / inspection_length**2))
				else:
					rw_times.append(rw_data['delta'])
				rw_D_sat.append(rw_data['D_sat'] / Dp)

				msd_data = read_msd_data_from_rwnmr_file(time_dir[2])
				rwDmsdX.append(msd_data['DmsdX'] / Dp)
				rwDmsdX_stdev.append(msd_data['DmsdX_stdev'] / Dp)
				rwDmsdY.append(msd_data['DmsdY'] / Dp)
				rwDmsdY_stdev.append(msd_data['DmsdY_stdev'] / Dp)
				rwDmsdZ.append(msd_data['DmsdZ'] / Dp)
				rwDmsdZ_stdev.append(msd_data['DmsdZ_stdev'] / Dp)

				rwDmsdMean.append((msd_data['DmsdX'] + msd_data['DmsdY'] + msd_data['DmsdZ']) / (3*Dp))
				rwDmsdStdDev.append(max([msd_data['DmsdX_stdev'], msd_data['DmsdY_stdev'], msd_data['DmsdZ_stdev']]) / (3*Dp))

			# convert to numpy arrays
			rw_times = np.array(rw_times)
			rw_D_sat = np.array(rw_D_sat)
			rwDmsdX = np.array(rwDmsdX)
			rwDmsdX_stdev = np.array(rwDmsdX_stdev)
			rwDmsdY = np.array(rwDmsdY)
			rwDmsdY_stdev = np.array(rwDmsdY_stdev)
			rwDmsdZ = np.array(rwDmsdZ)
			rwDmsdZ_stdev = np.array(rwDmsdZ_stdev)
			rwDmsdMean = np.array(rwDmsdMean)
			rwDmsdStdDev = np.array(rwDmsdStdDev)


			# sort arrays based on time measures
			indexes = rw_times.argsort()
			sorted_times = rw_times[indexes[::-1]]
			sorted_Dsat = rw_D_sat[indexes[::-1]]
			sorted_DmsdX = rwDmsdX[indexes[::-1]]
			sorted_DmsdX_stdev = rwDmsdX_stdev[indexes[::-1]]
			sorted_DmsdY = rwDmsdY[indexes[::-1]]
			sorted_DmsdY_stdev = rwDmsdY_stdev[indexes[::-1]]
			sorted_DmsdZ = rwDmsdZ[indexes[::-1]]
			sorted_DmsdZ_stdev = rwDmsdZ_stdev[indexes[::-1]]
			sorted_DmsdMean = rwDmsdMean[indexes[::-1]]
			sorted_DmsdStdDev = rwDmsdStdDev[indexes[::-1]]

			indexes = np.argsort(rw_times)
			new_wrap = {
				'time': sorted_times,
				'D_sat': sorted_Dsat,
				'DmsdX': sorted_DmsdX,
				'DmsdX_stdev': sorted_DmsdX_stdev,
				'DmsdY': sorted_DmsdY,
				'DmsdY_stdev': sorted_DmsdY_stdev,
				'DmsdZ': sorted_DmsdZ,
				'DmsdZ_stdev': sorted_DmsdZ_stdev,
				'DmsdMean': sorted_DmsdMean,
				'DmsdStdDev': sorted_DmsdStdDev
			}
			rw_wrap.append(new_wrap)


	# Plot 
	# plt.rcParams.update({'font.size': 14})
	plt.figure(figsize=[9.6, 7.2])
	for i in range(nrhos):
		for j in range(nshifts):
			idx = i*nshifts + j
			new_label = r'$\rho a / D_{0} = $' +  str(rhos[i]) + r', $\epsilon =$ ' + str(step_size[0]) + r' $\mu m$, '
			plt.errorbar(rw_wrap[idx]['time'], rw_wrap[idx]['DmsdMean'], yerr=rw_wrap[idx]['DmsdStdDev'], fmt=markers[i], color=colors[i][j], label=new_label)
			# plt.plot(rw_wrap[idx]['time'], rw_wrap[idx]['D_sat'], 'x', color=colors[i][j])
			if (PLOT_TORT_LIM):
				plt.hlines(tort_lims[i], 
					       0.95*max(rw_wrap[0]['time']), 
					       1.1*max(rw_wrap[0]['time']), 
					       linestyle='dashed', 
					       color=colors[i][j], 
					       label=r'$D_{e}/D_{0}=$'+str(tort_lims[i]))

	plt.legend(loc="best", ncol = nrhos)
	plt.title(plot_title)
	plt.xlim([0.0, 1.01*max(rw_wrap[0]['time'])])
	plt.ylim([0.0, 1.0])	
	plt.ylabel(r'$D(t)/D_{0}$')
	if(SQRT_TIME):
		plt.xlabel(r'$\sqrt{t}$' + '\t' + r'$\bf{[msec}^{1/2}\bf{]}$')
	elif(ADIM_TIME):
		plt.xlabel(r'$\sqrt{D_{0}t/a^{2}}$')
	else:
		plt.xlabel(r'$t$' + '\t' + r'$\bf{[msec]}$')	
	plt.tight_layout()
	plt.show()
				

			

if __name__ == '__main__':
	main()