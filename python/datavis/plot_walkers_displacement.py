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
	sim_dir = r'/PFGSE_NMR_cylinder_r=10.0_rho=0.0_res=1.0_shift=0_w=1M'
	delta_times = [r"8-0", r"20-0", r"40-0", r"80-0"]
	exp_dir = []
	for time in delta_times:
		delta_dir = r"/NMR_pfgse_" + time +"ms_0ms_42_sT.txt/"
		exp_dir.append(delta_dir)

	# txt file names
	walker_file = r'/NMR_collisions.txt'
	info_file = r'/NMR_imageInfo.txt'
	
	# assemblying complete paths
	file_img = data_dir + sim_dir + info_file 
	file_wlk = []
	for time_dir in exp_dir:
		file_wlk.append(data_dir + sim_dir + time_dir + walker_file)

	# read image info
	image_info = read_image_info_from_file(file_img)

	# read walker info
	n_walkers = read_number_of_walkers_from_file(file_wlk[0])
	print('number of walkers is {}'.format(n_walkers))

	# read displacement histogram
	labels = ['t = 8 ms', 't = 20 ms', 't = 40 ms', 't = 80 ms']
	bpos = []
	fpos = []
	for file in file_wlk:
		# alloc resources
		x0 = np.zeros((3, n_walkers), dtype=np.int32)
		xF = np.zeros((3, n_walkers), dtype=np.int32)

		# read data from file
		for idx in range(3):
			initial_data = read_walker_position_info_from_file_to_array(file, n_walkers, idx + 1)
			final_data = read_walker_position_info_from_file_to_array(file, n_walkers, idx + 4)
			x0[idx, :] = initial_data
			xF[idx, :] = final_data

		bpos.append(x0)
		fpos.append(xF)

	plot_walker_displacement_histograms(bpos, fpos, labels, image_info)


	return

if __name__ == '__main__':
	main()
	
