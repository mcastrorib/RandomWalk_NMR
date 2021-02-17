import os.path
import numpy as np
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_data import NMR_data
from NMR_PlotProperties import NMR_PlotProperties
from NMR_walker import NMR_walker

def main():
	data_dir = r'/home/matheus/Documentos/Doutorado IC/tese/NMR/RWNMR/data'
	sim_dir = r'/PFGSE_NMR_cylinder_r=10.0_rho=0.0_res=1.0_shift=0_w=100k'
	exp_dir = r'/NMR_pfgse_20-0ms_0ms_42_sT.txt'
	walker_file = r'/NMR_collisions.txt'
	file = data_dir + sim_dir + exp_dir + walker_file
	n_walkers = read_number_of_walkers_from_file(file)
	print('number of walkers is {}'.format(n_walkers))

	# alloc resources
	n_walkers = 1000
	x0 = np.zeros((n_walkers, 3), dtype=np.int32)
	xF = np.zeros((n_walkers, 3), dtype=np.int32)

	# read data from file
	for idx in range(3):
		initial_data = read_walker_position_info_from_file_to_array(file, n_walkers, idx + 1)
		final_data = read_walker_position_info_from_file_to_array(file, n_walkers, idx + 4)
		x0[:,idx] = initial_data
		xF[:,idx] = final_data

	plot_walker_positions(x0, xF, True)

	# Random sampling of walkers
	# walkers_sample = 0.001
	# walkers_to_read = int(walkers_sample * n_walkers)
	# print('walkers sampled: {}'.format(walkers_to_read))
	# walkers_ids = np.random.randint(0,n_walkers,walkers_to_read)
	# print('sample = {}'.format(walkers_ids))

	# # alloc resources
	# initial_positions = np.zeros((walkers_to_read, 3), dtype=np.int32)
	# final_positions = np.zeros((walkers_to_read, 3), dtype=np.int32)
	
	# for id in range(walkers_to_read):
	# 	walker = read_walker_from_file(file, walkers_ids[id])
	# 	initial_positions[id, :] = [walker.get_x0(), walker.get_y0(), walker.get_z0()]
	# 	final_positions[id, :] = [walker.get_x(), walker.get_y(), walker.get_z()]		

	# plot_walker_positions(initial_positions, final_positions, True)

	return

if __name__ == '__main__':
	main()
	
