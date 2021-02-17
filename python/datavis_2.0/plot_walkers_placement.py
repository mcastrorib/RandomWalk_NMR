import os.path
import numpy as np
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_data import NMR_data
from NMR_PlotProperties import NMR_PlotProperties
from NMR_walker import NMR_walker

def main():
	# dirpaths
	data_dir = r'/home/matheus/Documentos/Doutorado IC/tese/NMR/RWNMR/data'
	sim_dir = r'/PFGSE_NMR_cylinder_r=10.0_rho=0.0_res=1.0_shift=0_w=100k'
	exp_dir = r'/NMR_pfgse_80-0ms_0ms_42_sT.txt'
	
	# txt file names
	walker_file = r'/NMR_collisions.txt'
	info_file = r'/NMR_imageInfo.txt'
	
	# complete paths
	file_img = data_dir + sim_dir + info_file 
	file_wlk = data_dir + sim_dir + exp_dir + walker_file

	# read image info
	image_info = read_image_info_from_file(file_img)

	# read walker info
	n_walkers = read_number_of_walkers_from_file(file_wlk)
	print('number of walkers is {}'.format(n_walkers))
	
	# alloc resources
	x0 = np.zeros((n_walkers, 3), dtype=np.int32)
	xF = np.zeros((n_walkers, 3), dtype=np.int32)

	# read data from file
	for idx in range(3):
		initial_data = read_walker_position_info_from_file_to_array(file_wlk, n_walkers, idx + 1)
		final_data = read_walker_position_info_from_file_to_array(file_wlk, n_walkers, idx + 4)
		x0[:,idx] = initial_data
		xF[:,idx] = final_data

	# plot histogram
	plot_walker_placement_histograms(x0, xF, image_info)

	return

if __name__ == '__main__':
	main()
	
