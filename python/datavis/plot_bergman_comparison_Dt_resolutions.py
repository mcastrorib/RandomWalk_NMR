import os.path
import numpy as np
from NMR_data import *
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import *

def main():
	# simulation parameters
	D0 = 2.5
	edge_length = 5.0
	samples = 40
	max_integer_time = 10
	use_decimals = True

	# data file location
	data_dir = r"../data/dtest/r=2.5um"
	gt_dir = r"/gt/"
	res_dirs = [r"/res05/", r"/res1/"]
	ground_truth_dir = r"PFGSE_NMR_rho=0_phi=47_radius=2.5_res=0.05/"
	experiment_dir = r"PFGSE_NMR_rho=0_phi=47_radius=2.5_"
	resolutions = [r"res=0.5", r"res=1.0"]
	shifts = [r"shift=0", r"shift=1", r"shift=2", r"shift=3"]

	dirs = []
	dirs.append(data_dir + gt_dir + ground_truth_dir)

	# add files from
	for idx in range(len(res_dirs)):
		for shift in shifts:
			dirs.append(data_dir + res_dirs[idx] + experiment_dir + resolutions[idx] + "_" + shift + "/")


	# set plot labels for each dataset
	labels = []
	labels.append("resolution = 0.05 um/voxel")
	labels.append("resolution = 0.5 um/voxel, shift = 0")
	labels.append("resolution = 0.5 um/voxel, shift = 1")
	labels.append("resolution = 0.5 um/voxel, shift = 2")
	labels.append("resolution = 0.5 um/voxel, shift = 3")
	labels.append("resolution = 1.0 um/voxel, shift = 0")
	labels.append("resolution = 1.0 um/voxel, shift = 1")
	labels.append("resolution = 1.0 um/voxel, shift = 2")
	labels.append("resolution = 1.0 um/voxel, shift = 3")

	# set plot colors for each dataset
	colors = []
	colors.append('red')
	colors.append('violet')
	colors.append('dodgerblue')
	colors.append('mediumpurple')
	colors.append('navy')
	colors.append('violet')
	colors.append('dodgerblue')
	colors.append('mediumpurple')
	colors.append('navy')

	# set plot markers for each dataset
	markers = []
	markers.append('--')
	markers.append('o')
	markers.append('o')
	markers.append('o')
	markers.append('o')
	markers.append('x')
	markers.append('x')
	markers.append('x')
	markers.append('x')



	integers = []
	for idx in range(max_integer_time + 1):
		integers.append(str(idx))

	# decimals = ["0"]
	decimals = [r"0", r"25", r"50", r"75"]
	

	datafiles = []
	for integer in integers:
		if(use_decimals):
			for decimal in decimals:
				filepath = r"NMR_pfgse_" + integer + "-" + decimal +"ms_0ms_42_sT.txt"
				if(os.path.isfile(dirs[0] + filepath)):
					datafiles.append(filepath)
		else:
			filepath = r"NMR_pfgse_" + integer + "ms_0ms_42_sT.txt"
	
			# print("path: ", directory + filepath)
			if(os.path.isfile(dirs[0] + filepath)):
				datafiles.append(filepath)

	print(datafiles)


	# create list of data for ploting
	dataList = []
	for dataset in range(len(dirs)):
		observation_time = []
		observation_time.append(0.0)
		Dt = []
		Dt.append(D0)	

		# read and store data from file list
		filenames = []
		for datafile in datafiles:
			filenames.append(dirs[dataset] + datafile)
		for file in filenames:
			Dt.append(read_diffusion_coefficient_from_file(file))
			observation_time.append(read_exposure_time_from_file(file))

		x_data = []
		for idx in range(samples):
			value = np.sqrt(D0 * observation_time[idx] / (edge_length * edge_length)) 
			x_data.append(value)

		y_data = []
		for idx in range(samples):
			value = Dt[idx] / D0
			y_data.append(value)

		print("D(t) = \n", Dt)
		print("y_data = \n", y_data)

		# set NMR data object
		nmr_data = NMR_data()
		nmr_data.setXData(x_data)
		nmr_data.setYData(y_data)
		nmr_data.setLabel(labels[dataset])
		nmr_data.setMarker(markers[dataset])
		nmr_data.setColor(colors[dataset])

		# add to data list
		dataList.append(nmr_data)

	# set plot properties
	plot_props = PlotProperties()
	plot_props.setTitle(r'$\rho a/D_{0} = 0.0$, $\phi = 0.476$')	
	plot_props.setXLabel(r'$ [D_{0} t / a^{2}]^{1/2} $')
	plot_props.setYLabel(r'$ D(t) / D_{0} $')
	plot_props.setYLim([-0.05, 1.05])	
	plot_NMR_data(dataList, plot_props)
	return

if __name__ == '__main__':
	main()