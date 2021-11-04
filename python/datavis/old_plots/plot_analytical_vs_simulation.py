import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_data import *
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import *

def main():
	# simulation parameters
	res = 1.0
	Dfree = 2.5
	
	walker_strings = [r'10k', r'100k', r'1M', r'10M']
	edge_length = 20.0
	edge_str = '20'
	
	relaxation_strength = 0.2
	rho_analytic = 0.05
	rho_sim = (1000.0 * relaxation_strength * Dfree)/edge_length


	'''
	list of time samples
	a = 1um ==> [r"0-08", r"0-20", r"0-40", r"0-80"]
	a = 2.5um ==> [r"0-50", r"1-25", r"2-50", r"5-0"]
	a = 5um ==> [r"2-0", r"5-0", r"10-0", r"20-0"]
	a = 10um ==> [r"8-0", r"20-0", r"40-0", r"80-0"]
	a = 20um ==> [r"32-0", r"80-0", r"160-0", r"320-0"]
	'''
	exp_times = [r"32-0", r"80-0", r"160-0", r"320-0"]

	for walker_str in walker_strings:
		# Plot title
		line1 = r'$\bf{Spherical Pore}$: resolution = ' + str(res) + r' $\mu$m/voxel'
		line2 = r'$\rho a/D$= ' + str(relaxation_strength) + r', a = ' + str(edge_length) + r'$\mu$m, D = ' + str(Dfree)  + r' $\mu$m²/s, walkers=' + walker_str
		plot_title =  line1 + '\n' + line2


		# Set simulation data files
		sim_data_dir = r"/home/matheus/Documentos/doutorado_ic/tese/saved_data/callaghan_test/relaxation_strength="+ str(relaxation_strength) +"/a=" + edge_str + r"um/isolated_sphere/using_q/res=1.0/"
		# sim_data_dir = r"/home/matheus/Documentos/Doutorado IC/tese/NMR/RWNMR/data/"
		sim_experiment_dir = r"PFGSE_NMR_sphere_r=" + str(edge_length) + r"_rho=" + str(rho_sim) + r"_res=1.0_shift=0_w=" + walker_str + r"/"
		sim_delta_times = exp_times	
		sim_dirs = []
		sim_dirs.append(sim_data_dir + sim_experiment_dir)
		sim_datafiles = []
		sim_echoes_file = r"PFGSE_echoes.txt"
		for time in sim_delta_times:
			sim_delta_dir = r"NMR_pfgse_" + time +"ms_0ms_42_sT/"
			filepath = sim_delta_dir + sim_echoes_file
			if(os.path.isfile(sim_dirs[0] + filepath)):
				sim_datafiles.append(filepath)
		
		# Set analytical data files (JSON format)
		analytic_data_dir = r"/home/matheus/Documentos/doutorado_ic/tese/NMR/callaghan_analytical_pores/data/"
		analytic_experiment_dir = r"SphericalPore_a=10.0_rho=0.05_D=2.5/"
		analytic_delta_times = [8.0, 20.0, 40.0, 80.0]
		analytic_dirs = []
		analytic_dirs.append(analytic_data_dir + analytic_experiment_dir)

		analytic_datafiles = []
		for time in analytic_delta_times:
			filepath = 'echoes_t=' + str(time) + 'ms_a=' + str(10.0) + '_rho=' + str(rho_analytic) + '.json'
			if(os.path.isfile(analytic_dirs[0] + filepath)):
				analytic_datafiles.append(filepath)

		# Set subplot titles 
		titles = []
		titles.append(r"$\Delta$ = 0.2 a²/D")
		titles.append(r"$\Delta$ = 0.5 a²/D")
		titles.append(r"$\Delta$ = 1.0 a²/D")
		titles.append(r"$\Delta$ = 2.0 a²/D")

		# set plot labels for each dataset
		labels = []
		labels.append(r"rw simulation")
		labels.append(r"analytical solution")

		# set plot colors for each dataset
		colors = []
		colors.append('violet')
		colors.append('black')

		# set plot markers for each dataset
		markers = []
		markers.append('*')
		markers.append('-')

		# rw data collection
		# create list of data for ploting
		rw_dataList = []
		for dataset in range(len(sim_dirs)):	
			# create file list for dataset
			filenames = []
			for datafile in sim_datafiles:
				filenames.append(sim_dirs[dataset] + datafile)
			print('\nrw data = \n', filenames)

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
				nmr_data.setLabel(labels[0])
				nmr_data.setMarker(markers[0])
				nmr_data.setColor(colors[0])

				# add to data list
				rw_dataList.append(nmr_data)
		# -- end of rw data collection

		# analytical data collection
		# create list of data for ploting
		analytic_dataList = []
		for dataset in range(len(analytic_dirs)):	
			# create file list for dataset
			filenames = []
			for datafile in analytic_datafiles:
				filenames.append(analytic_dirs[dataset] + datafile)
			print('\nanalytical data = \n', filenames)

			# read and store data from file list
			for file in range(len(filenames)):
				qa = read_analytic_qa_data_from_json(filenames[file])
				echoes = read_analytic_echoes_from_json(filenames[file])

				x_data = []
				for idx in range(len(qa)):
					x_data.append(qa[idx])

				y_data = []
				for idx in range(len(echoes)):
					y_data.append(echoes[idx])

				# set NMR data object
				nmr_data = NMR_data()
				nmr_data.setXData(x_data)
				nmr_data.setYData(y_data)
				nmr_data.setLabel(labels[1])
				nmr_data.setMarker(markers[1])
				nmr_data.setColor(colors[1])

				# add to data list
				analytic_dataList.append(nmr_data)
		# -- end of analytical data collection
		
		# Create list of organized data for plots
		figs_data = []
		figs_props = []
		for idx in range(len(rw_dataList)):
			# set plot properties
			plot_props = NMR_PlotProperties()	
			plot_props.setXLabel(r'$ qa $')
			plot_props.setYLabel(r'$ E(q,\Delta) $')
			plot_props.setFigureSize([10,10])
			plot_props.setDPI(100)	
			plot_props.setTitle(titles[idx])
			# set plot with threshold
			plot_props.setXLim([0.0, 2.0])	
			plot_props.setYLim([0.0001, 1.0])

			# set data
			plot_data = []
			plot_data.append(rw_dataList[idx])
			plot_data.append(analytic_dataList[idx])
			
			figs_data.append(plot_data)
			figs_props.append(plot_props)

		# Plot data
		subplot_semilogy(figs_data, figs_props, plot_title)
	return

if __name__ == '__main__':
	main()