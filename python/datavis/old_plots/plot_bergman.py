import numpy as np
from NMR_ReadFromFile import *
from NMR_Plots import *

def main():
	data_dir = r"../data/"#saved_data/bergman_test/"
	experiment_dir = r"PFGSE_NMR_rho0_phi20_res1/"
	# experiment_dir = r"PFGSE_NMR_rho0/"
	# experiment_dir = r"PFGSE_NMR_rho250/"

	datafiles = []
	datafiles.append(r"NMR_pfgse_0ms_0ms_42_sT.txt")
	datafiles.append(r"NMR_pfgse_3ms_0ms_42_sT.txt")
	datafiles.append(r"NMR_pfgse_6ms_0ms_42_sT.txt")
	datafiles.append(r"NMR_pfgse_13ms_0ms_42_sT.txt")
	datafiles.append(r"NMR_pfgse_26ms_0ms_42_sT.txt")
	datafiles.append(r"NMR_pfgse_40ms_0ms_42_sT.txt")

	
	filenames = []
	for datafile in datafiles:
		filenames.append(data_dir + experiment_dir + datafile)

	# read and store data from file list
	# read time, width and gamma values
	observation_time = read_exposure_time_from_file(filenames[0])
	pulse_width = read_pulse_width_from_file(filenames[0])
	gyromagnetic_ratio = read_gyromagnetic_ratio_from_file(filenames[0])
	gradient = read_gradient_from_file(filenames[0])
	lhs = read_lhs_from_file(filenames[0])


	title = r'$\rho a / D = 0.0$'		
	plot_pfgse_bergman(lhs, gradient, observation_time, pulse_width, gyromagnetic_ratio, title)
	return

if __name__ == '__main__':
	main()