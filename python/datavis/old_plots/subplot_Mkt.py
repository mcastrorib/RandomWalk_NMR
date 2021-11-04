import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_data import *
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import *
from LeastSquaresRegression import LeastSquaresRegression

def read_periodic_data_from_json_file(filepath):
	D0 = 0.0
	length = 0.0
	rho = 0.0
	time = 0.0
	k_samples = 0.0
	ka_max = 0.0
	k = []
	Mkt = []

	with open(filepath) as json_file:
		# load json object
		json_data = json.load(json_file)
		
		# read scalars
		D0 = float(json_data["D0"])
		length = float(json_data["length"])
		rho = float(json_data["rho"])
		time = float(json_data["time"])
		k_samples = int(json_data['k_samples'])
		ka_max = float(json_data['ka_max'])

		# unwrap k array
		k = np.zeros([k_samples, 3])
		for idx in range(k_samples):
			k[idx, 0] = float(json_data['k'][idx][0])
			k[idx, 1] = float(json_data['k'][idx][1])
			k[idx, 2] = float(json_data['k'][idx][2])

		# unwrap Mkt array
		Mkt = np.zeros([k_samples])
		for idx in range(k_samples):
			Mkt[idx] = float(json_data['Mkt'][idx])

	data = {
		'D0': D0,
		'length': length,
		'rho': rho,
		'time': time,
		'k_samples': k_samples,
		'ka_max': ka_max,
		'k': k,
		'Mkt': Mkt
	}

	return data


def get_ka_vec(gradient_vec, giromagnetic, width, length_a, samples):
	vecKA = np.zeros(samples)
	for idx in range(samples):
		vecKA[idx] = (1.e-5 * gradient_vec[idx] * giromagnetic * width) * length_a
	return vecKA

def main():
	# simulation parameters
	Dp = 2.5
	edge_length = [5.0, 10.0, 20.0, 40.0, 100.0]
	radius = [2.5, 5.0, 10.0, 20.0, 50.0]
	number_of_lengths = len(edge_length)
	walker_strings = [r'1M'] #[r'10k', r'100k', r'1M', r'10M']
	time_scales = [edge**2 / Dp for edge in edge_length] 
	phi = 0.47
	relaxation_strength = 0.0
	rho = [relaxation_strength*Dp/edge for edge in edge_length]
	resolution = 1.0
	shift = 0
	
	# Database dir
	db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/bergman_test/'
	db_dir += r'Mkt_rho=' + str(relaxation_strength) + r'/'
	db_dir += r'phi=' + str(phi) + r'_res=' + str(resolution) + r'/' 

	for walkers in walker_strings:
		# Plot title
		line1 = r'$\bf{Periodic Array}$: resolution = ' + str(resolution) + r' $\mu$m/voxel'
		line2 = r'$\rho a/D$= ' + str(relaxation_strength) + r', $D_{0}$ = ' + str(Dp)  + r' $\mu$m²/s, walkers=' + walkers
		plot_title =  line1 + '\n' + line2


		# find data files 
		complete_paths = []
		for idx in range(number_of_lengths):
			sim_dir = r'a=' + str(edge_length[idx]) + r'um_r=' + str(radius[idx]) + r'um/data/'
			sim_dir += r'PFGSE_NMR_periodic_a=' + str(edge_length[idx]) + r'_r=' + str(radius[idx]) 
			sim_dir += r'_rho=' + str(rho[idx]) + r'_res=' + str(resolution) + r'_shift=' + str(shift) + r'_w='+ walkers + r'/'

			# assembly complete path to data directory
			complete_paths.append(db_dir + sim_dir)

		# List all data files in directory
		dataset_files = []
		for complete_path in complete_paths:
			exp_dirs = os.listdir(complete_path)
			number_of_sim_files = len(exp_dirs)
			echoes_files = [complete_path + exp_dir + '/PFGSE_echoess.txt' for exp_dir in exp_dirs]
			params_files = [complete_path + exp_dir + '/PFGSE_parameters.txt' for exp_dir in exp_dirs]

			data_files = []
			for i in range(number_of_sim_files):
				data_files.append([params_files[i], echoes_files[i]])

			# check if path exists
			for file in data_files:
				if(os.path.isfile(file[0]) and os.path.isfile(file[1])):
					print('FILES ' + file[0] + ' AND ' + file[1] + ' FOUND.')
				else:
					print('FILE ' + file[0] + ' AND ' + file[1] + ' NOT FOUND. Removed from list')
					data_files.remove(file)
	
			number_of_sim_files = len(data_files)
			dataset_files.append(data_files)

		# obs: dataset_files[edge][time][file]
		rw_wrap = []
		for edge_idx in range(number_of_lengths):
			rw_data_x = []
			rw_data_y = []
			rw_times = []
			rw_labels = []
			rw_indexes = []
			for time_dir in dataset_files[edge_idx]:
				rw_data = read_pfgse_data_from_rwnmr_file(time_dir)
				rw_times.append(rw_data['delta'])
				rw_data_y.append(np.array(rw_data['Mkt']))
				rw_data_x.append(get_ka_vec(rw_data['gradient'], rw_data['giromagnet'], rw_data['width'], edge_length[edge_idx], rw_data['points']))
				rw_labels.append('rw: ' + str(rw_data['delta'] / time_scales[edge_idx])  + ' x a²/D0')

			rw_indexes = np.array(rw_times).argsort()
			wrapped_data = {
				'times': rw_times,
				'data_x': rw_data_x,
				'data_y': rw_data_y,
				'labels': rw_labels,
				'indexes': rw_indexes
			}

			rw_wrap.append(wrapped_data)

		# Set analytical data files (JSON format)
		# find analytic solution data files
		# analytical solution parameters
		ansol_Dp = 2.5
		ansol_a = 10.0
		ansol_r = 5.0
		ansol_time_scale = ansol_a**2 / ansol_Dp
		ansol_db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_1.0/db'
		ansol_dir = r'/Analytical_Periodic_Media_phi=0.47_rho=0.0_N=5'
		ansol_files = [ansol_db_dir + ansol_dir + '/' + file for file in os.listdir(ansol_db_dir + ansol_dir)]
		number_of_ansol_files = len(ansol_files)
		sim_xlim = rw_wrap[0]['data_x'][0][-1]

		# read and process data from analytic solution files
		ansol_datasets = []
		for file in ansol_files:
			ansol_datasets.append(read_periodic_data_from_json_file(file))

		ansol_data_x = []
		ansol_data_y = []
		ansol_times = []
		ansol_labels = []
		data_idx = 0
		for dataset in ansol_datasets:
			ansol_times.append(dataset['time'])
			ansol_labels.append('analytic: ' + str(dataset['time'] / ansol_time_scale) + ' x a²/D0')

			# read x data
			ka_vec = []
			a = dataset['length']
			count = 0
			for i in range(dataset['k_samples']):
				kval = a * np.linalg.norm(dataset['k'][i])
				if(kval <= sim_xlim):
					ka_vec.append(kval)
					count += 1
			ansol_data_x.append(np.array(ka_vec))

			Mkt = np.zeros(count)
			M0 = dataset['Mkt'][0]
			for i in range(count):
				Mkt[i] = dataset['Mkt'][i] / M0
			ansol_data_y.append(Mkt)
			data_idx += 1	
		ansol_indexes = np.array(ansol_times).argsort()	

		# Set subplot titles 
		titles = []
		titles.append(r"$\Delta$ = 0.2 a²/D")
		titles.append(r"$\Delta$ = 0.5 a²/D")
		titles.append(r"$\Delta$ = 1.0 a²/D")
		titles.append(r"$\Delta$ = 2.0 a²/D")

		# set plot labels for each dataset
		labels = []
		for edge in edge_length:
			labels.append(r"rw: a = " + str(edge))
		labels.append(r"analytical solution")

		# set plot colors for each dataset
		colors = []
		colors.append('magenta')
		colors.append('darkorchid')
		colors.append('grey')
		colors.append('dodgerblue')
		colors.append('navy')
		colors.append('red')

		# set plot markers for each dataset
		markers = []
		markers.append('d')
		markers.append('x')
		markers.append('d')
		markers.append('x')
		markers.append('d')
		markers.append('-')	

		# print(rw_wrap)
		# rw data collection
		rw_dataList = []
		for edge_idx in range(number_of_lengths):
			dataset = []
			for idx in rw_wrap[edge_idx]['indexes']:
				nmr_data = NMR_data()
				nmr_data.setXData(rw_wrap[edge_idx]['data_x'][idx])
				nmr_data.setYData(rw_wrap[edge_idx]['data_y'][idx])
				nmr_data.setLabel(labels[edge_idx])
				nmr_data.setMarker(markers[edge_idx])
				nmr_data.setColor(colors[edge_idx])

				# add to data list
				dataset.append(nmr_data)
			rw_dataList.append(dataset)


		# analytical data collection
		# create list of data for ploting
		analytic_dataList = []
		for idx in ansol_indexes:	
			# set NMR data object
			nmr_data = NMR_data()
			nmr_data.setXData(ansol_data_x[idx])
			nmr_data.setYData(ansol_data_y[idx])
			nmr_data.setLabel(labels[-1])
			nmr_data.setMarker(markers[-1])
			nmr_data.setColor(colors[-1])

			# add to data list
			analytic_dataList.append(nmr_data)
		# -- end of analytical data collection

		# Create list of organized data for plots
		figs_data = []
		figs_props = []
		for time in range(len(titles)):
			# set plot properties
			plot_props = NMR_PlotProperties()	
			plot_props.setXLabel(r'$ ka $')
			plot_props.setYLabel(r'$ M(k,\Delta) $')
			plot_props.setFigureSize([10, 10])
			plot_props.setDPI(100)	
			plot_props.setTitle(titles[time])
			# set plot with threshold
			# plot_props.setXLim([0.0, 1.35*2*np.pi])	
			# plot_props.setYLim([0.000001, 1.0])

			# set data
			plot_data = []
			for dataset in range(len(rw_dataList)):
				plot_data.append(rw_dataList[dataset][time])
			plot_data.append(analytic_dataList[time])
			
			figs_data.append(plot_data)
			figs_props.append(plot_props)

		# Plot data
		subplot_semilogy(figs_data, figs_props, plot_title)

		# Debug do jeff
		plt.rcParams.update({'font.size': 16})
		plt.figure(figsize=[10, 7.8])
		for idx in range(number_of_lengths):
			new_label = r'a = ' + str(edge_length[idx]) + r' $\mu m$'
			plt.semilogy(rw_dataList[idx][2].x_data, rw_dataList[idx][2].y_data, 'x', markersize=4.0, color=colors[idx], label=new_label)
		plt.semilogy(analytic_dataList[2].x_data, analytic_dataList[2].y_data, '-', linewidth=3.0, color='red', label='analytical solution')
		plt.legend(loc='upper right')
		plt.title(plot_title)
		# plt.xlim([0.0, 1.0])
		# plt.ylim([0.3, 1.0])	
		plt.ylabel(r'$ M(k,\Delta) $')
		plt.xlabel(r'$ ka $')
		# Show the major grid lines with dark grey lines
		plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

		# Show the minor grid lines with very faint and almost transparent grey lines
		plt.minorticks_on()
		plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)
		plt.show()
	return

if __name__ == '__main__':
	main()