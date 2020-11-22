import os.path
import numpy as np
from NMR_data import *
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import *

def main():
	# simulation parameters
	edge_length = 5.0

	# plot title
	plot_title = r'$\bf{Cylinder:}$ a = 5.0 $\mu$m, D = 2.5 $\mu$m²/s w=100k'

	# data file location
	data_dir = r"/home/matheus/Documentos/Doutorado IC/tese/saved_data/callaghan_test/a=5um/isolated_cylinder/using_q/res=1.0/"
	# data_dir = r"/home/matheus/Documentos/Doutorado IC/tese/NMR/RWNMR/data/"
	experiment_dir = r"PFGSE_NMR_cylinder_r=5.0_rho=0.0_res=1.0_shift=0_w=100k/"
	delta_times = [r"2-0", r"5-0", r"10-0", r"20-0"]	


	dirs = []
	dirs.append(data_dir + experiment_dir)

	datafiles = []
	echoes_file = r"PFGSE_echoes.txt"
	for time in delta_times:
		delta_dir = r"NMR_pfgse_" + time +"ms_0ms_42_sT/"
		filepath = delta_dir + echoes_file
		if(os.path.isfile(dirs[0] + filepath)):
			datafiles.append(filepath)

	print(datafiles)
	
	# set plot labels for each dataset
	labels = []
	labels.append(r"$\Delta = 0.2 a^{2}/D$")
	labels.append(r"$\Delta = 0.5 a^{2}/D$")
	labels.append(r"$\Delta = 1.0 a^{2}/D$")
	labels.append(r"$\Delta = 2.0 a^{2}/D$")

	# set plot colors for each dataset
	colors = []
	colors.append('violet')
	colors.append('dodgerblue')
	colors.append('mediumpurple')
	colors.append('navy')


	# set plot markers for each dataset
	markers = []
	# markers.append('s')
	# markers.append('p')
	# markers.append('D')
	# markers.append('*')
	markers.append('-s')
	markers.append('-p')
	markers.append('-D')
	markers.append('-*')

	# create list of data for ploting
	dataList = []
	for dataset in range(len(dirs)):	
		# create file list for dataset
		filenames = []
		for datafile in datafiles:
			filenames.append(dirs[dataset] + datafile)

		# read and store data from file list
		for file in range(len(filenames)):
			observation_time = read_exposure_time_from_file(filenames[file])
			pulse_width = read_pulse_width_from_file(filenames[file])
			gyromagnetic_ratio = read_gyromagnetic_ratio_from_file(filenames[file])
			gradient = read_gradient_from_file(filenames[file])
			lhs = read_lhs_from_file(filenames[file])

			x_data = []
			for idx in range(len(gradient)):
				value = (1.0/(2*np.pi)) * gradient[idx] * gyromagnetic_ratio * pulse_width * 1.0e-5
				x_data.append(value * edge_length)

			y_data = []
			for idx in range(len(lhs)):
				y_data.append(np.exp(lhs[idx]))

			# set NMR data object
			nmr_data = NMR_data()
			nmr_data.setXData(x_data)
			nmr_data.setYData(y_data)
			nmr_data.setLabel(labels[file])
			nmr_data.setMarker(markers[file])
			nmr_data.setColor(colors[file])

			# add to data list
			dataList.append(nmr_data)

	# set plot properties
	plot_props = NMR_PlotProperties()
	plot_props.setTitle(plot_title)	
	plot_props.setXLabel(r'$ qa $')
	plot_props.setYLabel(r'$ E(q,\Delta) $')
	plot_props.setFigureSize([10,7])
	plot_props.setDPI(100)	

	# set plot without threshold
	semilogy_NMR_data(dataList, plot_props)

	# set plot with threshold
	plot_props.setXLim([0.0, 2.0])	
	plot_props.setYLim([0.0001, 1.0])
	semilogy_NMR_data(dataList, plot_props)

	# set plot threshold
	plot_props.setXLim([0.0, 1.5])	
	plot_props.setYLim([0.01, 1.0])
	semilogy_NMR_data(dataList, plot_props)
	return

if __name__ == '__main__':
	main()