import numpy as np
from NMR_ReadFromFile import *
from NMR_Plots import *

def main():
	data_dir = r"../data/"
	experiment_dir0 = r"PFGSE_NMR_rho0_phi20_res1/"
	experiment_dir250 = r"PFGSE_NMR_rho250_phi20_res1/"

	datafiles = []
	datafiles.append(r"NMR_pfgse_0ms_0ms_42_sT.txt")
	datafiles.append(r"NMR_pfgse_3ms_0ms_42_sT.txt")
	datafiles.append(r"NMR_pfgse_6ms_0ms_42_sT.txt")
	datafiles.append(r"NMR_pfgse_13ms_0ms_42_sT.txt")
	datafiles.append(r"NMR_pfgse_26ms_0ms_42_sT.txt")
	datafiles.append(r"NMR_pfgse_40ms_0ms_42_sT.txt")
	
	labels = []
	labels.append(r'$\rho = 0.0$')
	labels.append(r'$\rho a/D = 250.0$')	
	rho_samples = len(datafiles)
	filenames = []
	observation_time = []
	pulse_width = []
	gyromagnetic_ratio = []
	gradient = []
	lhs = []

	for datafile in datafiles:
		filenames.append(data_dir + experiment_dir0 + datafile)
		filenames.append(data_dir + experiment_dir250 + datafile)

	# read and store data from file list
		
	for file in filenames:
		observation_time.append(read_exposure_time_from_file(file))
		pulse_width.append(read_pulse_width_from_file(file))
		gyromagnetic_ratio.append(read_gyromagnetic_ratio_from_file(file))
		gradient.append(read_gradient_from_file(file))
		lhs.append(read_lhs_from_file(file))

	title = r'$\phi = 0.202$'
	colors = []
	colors.append("black")
	colors.append("blue")
	markers = []
	markers.append(r'-')
	markers.append(r'o')
	labels = []
	labels.append(r'$\rho a/D_{0} = 0.0$')
	labels.append(r'$\rho a/D_{0} = 1.0$')
	
			
	plot_pfgse_bergman_comparison(lhs, 
		                          gradient, 
		                          observation_time, 
		                          pulse_width, 
		                          gyromagnetic_ratio, 
		                          rho_samples, 
		                          colors, 
		                          markers, 
		                          labels, 
		                          title)
	return

if __name__ == '__main__':
	main()