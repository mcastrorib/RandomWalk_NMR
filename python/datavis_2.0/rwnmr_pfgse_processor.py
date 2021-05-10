import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_ReadFromFile import *
from LeastSquaresRegression import LeastSquaresRegression

def find_optimal_SV_points(sqrt_time, Dt_data, min_points=2):
	nTimes = len(Dt_data) + 1
	points = min_points
	lsa = LeastSquaresRegression()		
	lsa.config(sqrt_time, Dt_data, points)
	lsa.solve()
	error = np.abs(lsa.results()["A"] - 1.0)
	print("points:", points, "A:", lsa.results()['A'], "error:", error)

	for idx in range(min_points+1, nTimes):
		lsa.config(sqrt_time, Dt_data, idx)
		lsa.solve()
		new_error = np.abs(lsa.results()["A"] - 1.0)
		print("points:", idx, "A:", lsa.results()['A'], "error:", new_error)
		if (new_error < error):
			error = new_error
			points = idx		

	return points

def main():
	# simulation parameters
	Dp = 2.5
	inspection_length = 10.0
	phi = 0.476
	resolution = 1.0
	walker_pop = r'100k'
	walker_samples = r'1'
	walkers_per_sample = r'100k'
	rho = 0.0
	shift = 0
	step_size = resolution/2**(shift)  
	min_SV_points = 3

	colors =  ['blue'] 
	markers = ['.']

	# set DB directory path
	db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/db/'
	# Plot title
	line1 = r'$\bf{Sphere\,Packing}$' + r' ($\phi =$' + str(phi) + r')' 
	line2 = 'resolution = ' + str(resolution) + r' $\mu$m/voxel'
	line2 += r', $\epsilon =$ ' + str(step_size) + r' $\mu m$, '
	line3 =  r'$D_{0}$ = ' + str(Dp)  + r' $\mu$m²/s, walkers = ' + walkers_per_sample
	plot_title =  line1 + '\n' + line2 + '\n' + line3

	# find data files 
	complete_paths = []
	# sim_dir = r'a=' + str(edge_length[idx]) + r'um_r=' + str(radius[idx]) + r'um/data/'
	sim_dir = r'PFGSE_NMR_periodic_a=20.0_r=10.0_rho=0.0_res=1.0_shift=0_w=100k_ws=1/'

	complete_path = db_dir + sim_dir

	# List all data files in directory
	dataset_files = []
	exp_dirs = os.listdir(complete_path)
	number_of_sim_files = len(exp_dirs)
	params_files = [complete_path + exp_dir + '/PFGSE_parameters.txt' for exp_dir in exp_dirs]
	echoes_files = [complete_path + exp_dir + '/PFGSE_echoes.txt' for exp_dir in exp_dirs]
	msd_files = [complete_path + exp_dir + '/PFGSE_msd.txt' for exp_dir in exp_dirs]

	data_files = []
	for i in range(number_of_sim_files):
		data_files.append([params_files[i], echoes_files[i], msd_files[i]])

	# check if path exists
	for file in data_files:
		if not (os.path.isfile(file[0]) and os.path.isfile(file[1]) and os.path.isfile(file[2])):
			data_files.remove(file)

	number_of_sim_files = len(data_files)

	rw_times = []
	rw_Dsat = []
	rw_Dmsd = []
	rw_DmsdX = []
	rw_DmsdY = []
	rw_DmsdZ = []

	for time_dir in data_files:
		rw_data = read_pfgse_data_from_rwnmr_file([time_dir[0], time_dir[1]])
		msd_data = read_msd_data_from_rwnmr_file(time_dir[2])
		if not (rw_data['delta'] in rw_times):	
			rw_times.append(rw_data['delta'])
			rw_Dsat.append(rw_data['Dsat'] / Dp)
			rw_Dmsd.append(rw_data['Dmsd'] / Dp)		
			rw_DmsdX.append(msd_data['DmsdX'] / Dp)
			rw_DmsdY.append(msd_data['DmsdY'] / Dp)
			rw_DmsdZ.append(msd_data['DmsdZ'] / Dp)

	rw_wrap = {
		'time': rw_times,
		'Dsat': rw_Dsat,
		'Dmsd': rw_Dmsd,
		'DmsdX': rw_DmsdX,
		'DmsdY': rw_DmsdY,
		'DmsdZ': rw_DmsdZ
	}

	plt.figure(figsize=[9.6, 7.2])
	new_label = r'$\phi = $' +  str(phi)
	plt.plot(rw_wrap['time'], rw_wrap['Dmsd'], markers[0], color=colors[0], label=new_label)

	plt.legend(loc='best')
	plt.title(plot_title)
	plt.xlim([0.0, 1.01*max(rw_wrap['time'])])
	plt.ylim([0.0, 1.0])	
	plt.ylabel(r'$D(t)/D_{0}$')
	plt.xlabel(r'$t$' + '\t' + r'$\bf{[msec]}$')	
	plt.tight_layout()
	plt.show()

	print('Unique time samples:', len(rw_wrap['time']))
	Dt_points = int(input('\nRecover D(t) with how many points? (end of data)\n'))
	SV_points = int(input('Recover SVp with how many points? (begin of data [-1: find best adjust])\n'))
	SV_findBest = False
	if(SV_points == -1):
		SV_findBest = True
		SV_points = len(rw_wrap['time'])

	De = 0.0
	Dxx = 0.0
	Dyy = 0.0
	Dzz = 0.0
	te = 0.0
	txx = 0.0
	tyy = 0.0
	tzz = 0.0

	indexes = np.argsort(rw_wrap['time'])
	last_idx = indexes.shape[0] - 1
	for i in range(last_idx, last_idx - Dt_points, -1):
		De += rw_wrap['Dmsd'][indexes[i]]
		Dxx += rw_wrap['DmsdX'][indexes[i]]
		Dyy += rw_wrap['DmsdY'][indexes[i]]
		Dzz += rw_wrap['DmsdZ'][indexes[i]]

	if(Dt_points > 0):
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
	SVp_factor = (-2.25 * np.sqrt(np.pi / Dp))
	SVp = 0.0
	sqrt_time = np.array([np.sqrt(rw_wrap['time'][indexes[i]]) for i in range(SV_points)])
	Dt_data = np.array([rw_wrap['Dmsd'][indexes[i]] for i in range(SV_points)])

	if(SV_findBest):
		SV_points = find_optimal_SV_points(sqrt_time, Dt_data, min_SV_points)

	lsa = LeastSquaresRegression()		
	lsa.config(sqrt_time, Dt_data, SV_points)
	lsa.solve()
	SVp = SVp_factor * lsa.results()["B"]

	print("B:", lsa.results()["B"])
	print("A:", lsa.results()["A"])
	print('- Pore surface-volume ratio')
	print('SVp:\t', SVp)
	print('points to adjust:\t', SV_points)
	print('D(0)/Dp estimation:', lsa.results()["A"])

	# Generate adjust data
	SV_x = np.array([0.0, (-lsa.results()["A"]/lsa.results()["B"])])
	SV_y = np.array([lsa.results()["A"], 0.0])


	plt.figure(figsize=[9.6, 7.2])
	# plot tortuosity line
	if(Dt_points > 0):
		plt.hlines(De, 0.0, 1.01*max(rw_wrap['time']), linestyle='dashed', color='black', label=r'$D_{e}/D_{p}=$'+'{:.6f}'.format(De))
	
	# plot SVp adjust line
	SV_label = r'$S_{V}=$' + '{:.6f}'.format(SVp) + r' $\mu m^{-1}$'
	if(SV_points > 0):
		plt.plot(SV_x, SV_y, '--', color='red', label=SV_label)

	# plot original data
	new_label = r'$\phi = $' +  str(phi)
	plt.plot(np.sqrt(rw_wrap['time']), rw_wrap['Dmsd'], markers[0], color=colors[0], label=new_label)

	

	plt.legend(loc='best')
	plt.title(plot_title)
	plt.xlim([0.0, np.sqrt(1.01*max(rw_wrap['time']))])
	plt.ylim([0.0, 1.0])	
	plt.ylabel(r'$D(t)/D_{0}$')
	plt.xlabel(r'$\sqrt{t}$' + '\t' + r'$\bf{[msec}^{1/2}\bf{]}$')
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()