import numpy as np
from NMR_ReadFromFile import *
from NMR_Plots import *

def main():
	data_dir = r"../data/"
	experiment_dir0 = r"PFGSE_NMR_Dtest_rho0_phi47/"
	experiment_dir250 = r"PFGSE_NMR_Dtest_rho0_phi20/"

	samples = 42
	D0 = 2.5
	edge_length = 10

	times = ["10.000000", "16.449315", "23.324159", "30.684675", 
	         "38.604738", "41.977854", "47.176470", "56.516806", 
	         "66.777282", "76.065620", "78.159198", "90.938120",
	         ]

	datafiles = []
	for idx in range(samples-1):
		datafiles.append(r"NMR_pfgse_" + str(idx) +"ms_0ms_42_sT.txt")	

	labels = []
	labels.append(r'$\rho = 0.0$')
	labels.append(r'$\rho a/D = 1.0$')	
	rho_samples = len(datafiles)
	observation_time = []
	observation_time.append(0.0)
	Dt1 = []
	Dt1.append(D0)
	
	# read and store data from file list
	filenames = []
	for datafile in datafiles:
		filenames.append(data_dir + experiment_dir0 + datafile)
	for file in filenames:
		Dt1.append(read_diffusion_coefficient_from_file(file))
		observation_time.append(read_exposure_time_from_file(file))

	

	title = r'$\rho a/D_{0} = 1.0$' 
	colors = []
	colors.append("black")
	colors.append("blue")
	markers = []
	markers.append(r'o')
	markers.append(r'o')
	labels = []
	labels.append(r'$\phi = 0.476$')
	
			
	plot_pfgse_bergman_D_comparison(observation_time, 
		                            Dt1,
		                            Dt2, 
		                            samples,
		                            D0, 
		                            edge_length,  
		                            labels, 
		                            title)
	return

if __name__ == '__main__':
	main()