from NMR_ReadFromFile import *
from NMR_Plots import *

def main():
	data_dir = r"../data/saved_data/resolution_tests/"
	dir_01 = r"PFGSE_NMR_01um_"
	dir_1 = r"PFGSE_NMR_1um_"
	dir_10 = r"PFGSE_NMR_10um_"
	walkers = r"10M_walkers"
	filename = r"/NMR_pfgse_200ms_2ms_42_sT.txt"

	file_01 = data_dir + dir_01 + walkers + filename
	file_1 = data_dir + dir_1 + walkers + filename
	file_10 = data_dir + dir_10 + walkers + filename

	files = []
	files.append(file_01)
	files.append(file_1)
	files.append(file_10)

	resolutions = []
	resolutions.append(0.1)
	resolutions.append(1.0)
	resolutions.append(10.0)

	D0s = []
	deltas = []
	thresholds = []
	gradients = []
	lhs = []
	rhs = []

	for file in files:
		D0s.append(read_diffusion_coefficient_from_file(file))
		deltas.append(read_exposure_time_from_file(file))
		thresholds.append(read_threshold_from_file(file))
		gradients.append(read_gradient_from_file(file))
		lhs.append(read_lhs_from_file(file))
		rhs.append(read_rhs_from_file(file))

	plot_lhs_vs_rhs_comparing_resolutions(lhs, rhs, D0s, deltas, thresholds, resolutions)

if __name__ == '__main__':
	main()