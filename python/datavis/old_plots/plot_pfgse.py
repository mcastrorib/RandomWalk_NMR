from NMR_ReadFromFile import *
from NMR_Plots import *

def plot_pfgse_results():
	data_dir = r"../data/saved_data/PFGSE_2_3_um2_s/PFGSE_NMR_10M_walkers/"

	# data_dir = r"../data/saved_data/PFGSE_NMR/"

	file_50 = data_dir + r"NMR_pfgse_50ms_2ms_42_sT.txt"
	file_100 = data_dir + r"NMR_pfgse_100ms_2ms_42_sT.txt"
	file_150 = data_dir + r"NMR_pfgse_150ms_2ms_42_sT.txt"
	file_200 = data_dir + r"NMR_pfgse_200ms_2ms_42_sT.txt"

	filename = []
	filename.append(file_50)
	filename.append(file_100)
	filename.append(file_150)
	filename.append(file_200)

	D0 = []
	delta = []
	limit = []
	gradient = []
	lhs = []
	rhs = []

	for file in filename:
		D0.append(read_diffusion_coefficient_from_file(file))
		delta.append(read_exposure_time_from_file(file))
		limit.append(read_threshold_from_file(file))
		gradient.append(read_gradient_from_file(file))
		lhs.append(read_lhs_from_file(file))
		rhs.append(read_rhs_from_file(file))
	
	plot_magnetization_vs_gradient(lhs, gradient, D0, delta)
	plot_lhs_vs_rhs(lhs, rhs, D0, delta, limit)
	return

if __name__ == '__main__':
	main()