import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_data import NMR_data
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import NMR_PlotProperties
from LeastSquaresRegression import LeastSquaresRegression

# def main():
# simulation parameters
Dp = 2.5
inspection_length = 100.0
walker_strings = [r'100k']
placement = 'cell'
rho = 0.0
resolution = 1.0
shifts = [3] 
colors =  ['black'] 
markers = ['^']
phi = ['low'] # ['high', 'low']
phi_value = [0.073] # [0.262, 0.073]
SQRT_TIME = False

# size variables
nshifts = len(shifts)
nphi = len(phi)

# Database dir
# db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/bergman_test/'
# db_dir += r'Dt_rho=' + str(relaxation_strength) + r'/'
# db_dir += r'phi=' + str(phi) + r'_res=' + str(resolution) + r'/'

# db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/bergman_test/new_version_tests/tests_periodic_bcs/'

db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/'
db_dir += r'saved_data/tortuosity_tests/'
db_dir += r'random_sphere_packs/pc_lab_short_times/'


# PFGSE_NMR_random_pack_phi=high_a=100.0_rho=0.0_res=1.0_shift=0_w=10k_cell

for walkers in walker_strings:
	# Plot title
	line1 = r'$\bf{Sphere\,Packing}$: resolution = ' + str(resolution) + r' $\mu$m/voxel'
	line2 = r'$\rho $= ' + str(rho) + r' $\mu$m/ms, D = ' + str(Dp)  + r' $\mu$m²/s, walkers=' + walkers
	plot_title =  line1 + '\n' + line2

	# find data files 
	complete_paths = []
	for porosity in phi:
		for shift in shifts:
			# sim_dir = r'a=' + str(edge_length[idx]) + r'um_r=' + str(radius[idx]) + r'um/data/'
			sim_dir = r'PFGSE_NMR_random_pack_phi=' + porosity
			sim_dir += r'_a=' + str(inspection_length) 
			sim_dir += r'_rho=' + str(rho) 
			sim_dir += r'_res=' + str(resolution) 
			sim_dir += r'_shift=' + str(shift) 
			sim_dir += r'_w='+ walkers
			sim_dir += r'_' + placement 
			sim_dir += r'/'

			complete_paths.append(db_dir + sim_dir)

	print(complete_paths)

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
			if(os.path.isfile(file[0]) and os.path.isfile(file[1])):
				print('FILES ' + file[0] + ' AND ' + file[1] + ' FOUND.')
			else:
				print('FILE ' + file[0] + ' AND ' + file[1] + ' NOT FOUND. Removed from list')
				data_files.remove(file)

		number_of_sim_files = len(data_files)
		dataset_files.append(data_files)

	rw_wrap = []
	for i in range(nphi):
		for j in range(nshifts):
			idx = i*nshifts + j 
			rw_times = []
			rw_D_sat = []
			rw_D_msd = []
			rw_DmsdX = []
			rw_DmsdY = []
			rw_DmsdZ = []

			for time_dir in dataset_files[idx]:
				rw_data = read_pfgse_data_from_rwnmr_file([time_dir[0], time_dir[1]])
				if(SQRT_TIME):
					rw_times.append(np.sqrt(rw_data['delta']))
				else:
					rw_times.append(rw_data['delta'])
				rw_D_sat.append(rw_data['D_sat'] / Dp)
				rw_D_msd.append(rw_data['D_msd'] / Dp)
				
				msd_data = read_msd_data_from_rwnmr_file(time_dir[2])
				rw_DmsdX.append(msd_data['DmsdX'] / Dp)
				rw_DmsdY.append(msd_data['DmsdY'] / Dp)
				rw_DmsdZ.append(msd_data['DmsdZ'] / Dp)

			new_wrap = {
				'time': rw_times,
				'D_sat': rw_D_sat,
				'D_msd': rw_D_msd,
				'DmsdX': rw_DmsdX,
				'DmsdY': rw_DmsdY,
				'DmsdZ': rw_DmsdZ
			}
			rw_wrap.append(new_wrap)

	plt.figure(figsize=[9.6, 7.2])
	for i in range(nphi):
		for j in range(nshifts):
			idx = i*nshifts + j
			new_label = r'$\phi = $' +  str(phi_value[i])
			plt.plot(rw_wrap[idx]['time'], rw_wrap[idx]['D_msd'], markers[j], color=colors[i][j], label=new_label)

	plt.legend(loc='best')
	plt.title(plot_title)
	plt.xlim([0.0, 1.01*max(rw_wrap[0]['time'])])
	plt.ylim([0.0, 1.0])	
	plt.ylabel(r'$D(t)/D_{0}$')
	if(SQRT_TIME):
		plt.xlabel(r'$\sqrt{t}$' + '\t' + r'$\bf{[msec}^{1/2}\bf{]}$')
	else:
		plt.xlabel(r'$t$' + '\t' + r'$\bf{[msec]}$')	
	plt.tight_layout()
	plt.show()

	Dt_points = int(input('\nRecover D(t) with how many points? (end of data)\n'))
	SV_points = int(input('Recover SVp with how many points? (begin of data)\n'))

	De = 0.0
	Dxx = 0.0
	Dyy = 0.0
	Dzz = 0.0
	te = 0.0
	txx = 0.0
	tyy = 0.0
	tzz = 0.0

	indexes = np.argsort(rw_wrap[0]['time'])
	last_idx = indexes.shape[0] - 1
	for i in range(last_idx, last_idx - Dt_points, -1):
		De += rw_wrap[0]['D_msd'][indexes[i]]
		Dxx += rw_wrap[0]['DmsdX'][indexes[i]]
		Dyy += rw_wrap[0]['DmsdY'][indexes[i]]
		Dzz += rw_wrap[0]['DmsdZ'][indexes[i]]

	# compute difusion coefficient
	De /= Dt_points
	Dxx /= Dt_points
	Dyy /= Dt_points
	Dzz /= Dt_points

	# compute tortuosity as t = sqrt(Dp/Dii)
	te = np.sqrt(1.0/De)
	txx = np.sqrt(1.0/Dxx)
	tyy = np.sqrt(1.0/Dyy)
	tzz = np.sqrt(1.0/Dzz)

	print('--- Results from last ', Dt_points, 'points:')
	print('- Diffusion effective coefficient (D)')
	print('De:\t', De)
	print('Dxx:\t', Dxx)
	print('Dyy:\t', Dyy)
	print('Dzz:\t', Dzz)
	print('- Tortuosity (t)')
	print('te:\t', te)
	print('txx:\t', txx)
	print('tyy:\t', tyy)
	print('tzz:\t', tzz)

	# Estimate SVp with least squares adjust
	SVp_factor = ( -2.25 * np.sqrt(np.pi) ) / (Dp)
	SVp = 0.0
	sqrt_time = None
	if(SQRT_TIME):
		sqrt_time = np.array([rw_wrap[0]['time'][indexes[i]] for i in range(SV_points)])
	else:
		sqrt_time = np.array([np.sqrt(rw_wrap[0]['time'][indexes[i]]) for i in range(SV_points)])
	Dt_data = np.array([np.sqrt(rw_wrap[0]['D_msd'][indexes[i]]) for i in range(SV_points)])
	lsa = LeastSquaresRegression()		
	lsa.config(sqrt_time, Dt_data, SV_points)
	lsa.solve()
	SVp = SVp_factor * lsa.results()["B"]

	print('- Pore surface-volume ratio')
	print('SVp:\t', SVp)
	print('Pore size estimation:\t', 3.0/SVp)

	# A = 
	# SVp_edge = np.array([1.0, ])
	plt.figure(figsize=[9.6, 7.2])
	plt.hlines(De, 0.0, 1.01*max(rw_wrap[0]['time']), linestyle='dashed', color='black', label=r'$D_{e}/D_{p}=$'+str(De))
	for i in range(nphi):
		for j in range(nshifts):
			idx = i*nshifts + j
			new_label = r'$\phi = $' +  str(phi_value[i])
			plt.plot(rw_wrap[idx]['time'], rw_wrap[idx]['D_msd'], markers[j], color=colors[i][j], label=new_label)

	plt.legend(loc='best')
	plt.title(plot_title)
	plt.xlim([0.0, 1.01*max(rw_wrap[0]['time'])])
	plt.ylim([0.0, 1.0])	
	plt.ylabel(r'$D(t)/D_{0}$')
	if(SQRT_TIME):
		plt.xlabel(r'$\sqrt{t}$' + '\t' + r'$\bf{[msec}^{1/2}\bf{]}$')
	else:
		plt.xlabel(r'$t$' + '\t' + r'$\bf{[msec]}$')	
	plt.tight_layout()
	plt.show()

# if __name__ == '__main__':
# 	main()