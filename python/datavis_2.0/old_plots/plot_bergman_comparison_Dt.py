import numpy as np
from NMR_ReadFromFile import *
from NMR_Plots import *

def main():
	data_dir = r"../data/"
	experiment_dir0 = r"PFGSE_NMR_Dtest_rho250_phi47_res1/"
	experiment_dir250 = r"PFGSE_NMR_Dtest_rho250_phi20_res1/"

	samples = 40
	D0 = 2.5
	edge_length = 10

	timefiles = []
	for idx in range(samples - 1):
		timefiles.append(idx + 1)
	print(timefiles)


	datafiles = []
	for idx in range(samples - 1):
		datafiles.append(r"NMR_pfgse_" + str(timefiles[idx]) +"ms_0ms_42_sT.txt")	

	labels = []
	labels.append(r'$\rho = 0.0$')
	labels.append(r'$\rho a/D = 1.0$')	
	rho_samples = len(datafiles)
	observation_time = []
	Dt1 = []
	Dt1.append(D0)
	Dt2 = []
	Dt2.append(D0)	

	# read and store data from file list
	filenames = []
	for datafile in datafiles:
		filenames.append(data_dir + experiment_dir0 + datafile)
	for file in filenames:
		Dt1.append(read_diffusion_coefficient_from_file(file))

	filenames = []
	for datafile in datafiles:
		filenames.append(data_dir + experiment_dir250 + datafile)
	for file in filenames:
		Dt2.append(read_diffusion_coefficient_from_file(file))

	delta_max = 40.0
	delta_min = 40.0 / (samples)
	for idx in range(samples):
		observation_time.append(idx*delta_min)

	title = r'$\rho a/D_{0} = 0.0$' 
	colors = []
	colors.append("black")
	colors.append("blue")
	markers = []
	markers.append(r'o')
	markers.append(r'o')
	labels = []
	labels.append(r'$\phi = 0.476$')
	labels.append(r'$\phi = 0.202$')
	
			
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