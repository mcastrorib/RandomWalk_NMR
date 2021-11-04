import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_data import NMR_data
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import NMR_PlotProperties
from LeastSquaresRegression import LeastSquaresRegression

def get_folder_for_visualization(db_dir):
	folders = sorted(os.listdir(db_dir))
	n_folders = len(folders)
	print("current saved NMR simulations:")
	folderID = 0
	for folder in folders:
	    print(folderID, '>', folder)
	    folderID += 1

	# Read and process user input
	user_inputs = input('choose the folder: ').split(' ')

	selected_folders = []
	valid_inputs = []	
	invalid_inputs = []
	for user_input in user_inputs:
		try:
			fId = int(user_input)
			if not (fId < 0 or fId > n_folders):
				if not (fId in valid_inputs):
					print(fId, '>', folders[fId])
					valid_inputs.append(fId)
					selected_folders.append(folders[fId])
		except:
			invalid_inputs.append(user_input)

	return selected_folders

def get_folders_labels(folders):
	labels = []
	print('')
	for folder in folders:
		new_label = input('>> Set label to folder ' + folder + ': ')
		labels.append(new_label)
	print('')
	return labels

def get_data_folder_files_and_labels(db_dir):
	sim_dirs = get_folder_for_visualization(db_dir)
	rw_labels = get_folders_labels(sim_dirs)
	
	complete_paths = []
	for sim_dir in sim_dirs:
		complete_paths.append(os.path.join(db_dir, sim_dir))

	# List all data files in directory
	rw_files = []
	for complete_path in complete_paths:
		dataset_files = []
		exp_dirs = os.listdir(complete_path)
		number_of_sim_files = len(exp_dirs)
		echoes_files = sorted( [os.path.join(complete_path, exp_dir, 'PFGSE_echoes.txt') for exp_dir in exp_dirs] )
		params_files = sorted( [os.path.join(complete_path, exp_dir, 'PFGSE_parameters.txt') for exp_dir in exp_dirs] )
		msd_files = sorted( [os.path.join(complete_path, exp_dir, 'PFGSE_msd.txt') for exp_dir in exp_dirs] )


		# wrap files in one list
		data_files = []
		for i in range(number_of_sim_files):
			data_files.append([params_files[i], echoes_files[i], msd_files[i]])


		# check if path exists, remove if necessary
		files_to_remove = []
		for file in data_files:
			if not (os.path.isfile(file[0]) and os.path.isfile(file[1]) and os.path.isfile(file[2])):
				files_to_remove.append(file)
		for file in files_to_remove:
			data_files.remove(file)
		rw_files.append(data_files)

	return rw_files, rw_labels

def read_data_from_output_files(data_files, data_labels):
	rw_wraps = []

	dir_count = 0
	for sim_dir in data_files:
		# read data from output files
		number_of_sim_files = len(sim_dir)
		rw_times = [] 
		rw_D_sat = [] 
		rw_D_msd = [] 
		rw_DmsdX = [] 
		rw_DmsdY = [] 
		rw_DmsdZ = [] 

		Dp = 0.0
		for dirIdx in range(number_of_sim_files):
			rw_data = read_pfgse_data_from_rwnmr_file([sim_dir[dirIdx][0], sim_dir[dirIdx][1]])
			msd_data = read_msd_data_from_rwnmr_file(sim_dir[dirIdx][2])
			Dp = rw_data['D0']
			
			if not (rw_data['delta'] in rw_times):	
				rw_times.append(rw_data['delta'])
				rw_D_sat.append(rw_data['Dsat'] / rw_data['D0'])
				rw_D_msd.append(rw_data['Dmsd'] / rw_data['D0'])	
				rw_DmsdX.append(msd_data['DmsdX'] / rw_data['D0'])
				rw_DmsdY.append(msd_data['DmsdY'] / rw_data['D0'])
				rw_DmsdZ.append(msd_data['DmsdZ'] / rw_data['D0'])

		rw_wrap = {
			'time': np.array(rw_times),
			'Dsat': np.array(rw_D_sat),
			'Dmsd': np.array(rw_D_msd),
			'DmsdX': np.array(rw_DmsdX),
			'DmsdY': np.array(rw_DmsdY),
			'DmsdZ': np.array(rw_DmsdZ),
			'D0': Dp,
			'label': data_labels[dir_count]
		}

		dir_count += 1

		rw_wraps.append(rw_wrap)

	return rw_wraps

def choose_data_to_plot():
	data_str = 'Dsat'
	plot_option = 0
	print('\n>> Select data to plot (default: Dsat):')
	print('0 > Dsat \n1 > Dmsd \n2 > DmsdX \n3 > DmsdY \n4 > DmsdZ')
	try: 
		plot_option = int(input('>> Option: '))
	except:
		plot_option = 0

	if(plot_option == 1):
		data_str = 'Dmsd'
	elif(plot_option == 2):
		data_str = 'DmsdX'
	elif(plot_option == 3):
		data_str = 'DmsdY'
	elif(plot_option == 4):
		data_str = 'DmsdZ'

	return data_str

def plot_rw_diffusion_data(rw_wraps, SQRT_TIME=False, marker='o', marker_size=3.0):

	data_str = choose_data_to_plot()
	print(data_str)	
	plt.figure(figsize=[9.6, 7.2])
	max_times = []
	for rw_wrap in rw_wraps:
		if(SQRT_TIME):
			plt.plot(np.sqrt(rw_wrap['time']), rw_wrap[data_str], marker, ms=marker_size, label=rw_wrap['label'])
			plt.xlabel(r'$\sqrt{t}$' + '\t' + r'$\bf{[msec}^{1/2}\bf{]}$')
			max_times.append(np.sqrt(np.max(rw_wrap['time'])))
		else:
			plt.plot(rw_wrap['time'], rw_wrap[data_str], marker, ms=marker_size, label=rw_wrap['label'])
			plt.xlabel(r'$t$' + '\t' + r'$\bf{[msec]}$')
			max_times.append(np.max(rw_wrap['time']))
	
	
	if(SQRT_TIME):
		plt.xlabel(r'$\sqrt{t}$' + '\t' + r'$\bf{[msec}^{1/2}\bf{]}$')
	else:
		plt.xlabel(r'$t$' + '\t' + r'$\bf{[msec]}$')
	plt.xlim([0.05, 1.01*max(max_times)])
	plt.title(data_str)
	plt.legend(loc='best')	
	plt.ylim([0.0, 1.0])	
	plt.ylabel(r'$D(t)/D_{0}$')
	plt.tight_layout()
	plt.show()
	return


def main():
	# Plot setup
	markers = ['o']
	markers_size = [3.0]
	SQRT_TIME = True

	# Database dir
	db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/db/'
	data_files, data_labels = get_data_folder_files_and_labels(db_dir)
	rw_wrap = read_data_from_output_files(data_files, data_labels)
	quit = False
	while(quit != True):
		plot_rw_diffusion_data(rw_wrap, SQRT_TIME, markers[0], markers_size[0])	
		quit_input = input('\n>> Plot again using same data? (y/n): ')
		if not (quit_input == 'y' or quit_input == 'Y'):
			quit = True

if __name__ == '__main__':
	main()
