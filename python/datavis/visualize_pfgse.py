import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_ReadFromFile import *

def get_rwnmr_data_folder(db_dir):
	folders = sorted(os.listdir(db_dir))
	valid_folders = []
	for folder in folders:
		if(os.path.isdir(os.path.join(db_dir,folder))):
			valid_folders.append(folder)

	print("current saved NMR simulations:")
	folderID = 0
	for folder in valid_folders:
	    print(folderID, '>', folder)
	    folderID += 1

	# Read and process user input
	user_input = input('choose the folder: ')
	print('\n')
	n_folders = len(valid_folders)
	fId = int(user_input)
	if not (fId < 0 or fId > n_folders):
		return os.path.join(db_dir, valid_folders[fId])
	else:
		return r''

def get_rwnmr_pfgse_folder(rwnmr_dir):
	folders = sorted(os.listdir(rwnmr_dir))
	valid_folders = []
	for folder in folders:
		if(os.path.isdir(os.path.join(rwnmr_dir,folder))):
			valid_folders.append(folder)

	print("RWNMR folders:")
	folderID = 0
	for folder in valid_folders:
	    print(folderID, '>', folder)
	    folderID += 1

	# Read and process user input
	user_input = input('choose the PFGSE folder: ')
	print('\n')
	n_folders = len(valid_folders)
	fId = int(user_input)
	if not (fId < 0 or fId > n_folders):
		return os.path.join(rwnmr_dir, valid_folders[fId])
	else:
		return r''

def read_data_from_pfgse_folder(cpmg_folder):
	params_file = os.path.join(cpmg_folder, r'PFGSE_parameters.txt')
	results_file = os.path.join(cpmg_folder, r'PFGSE_results.csv')

	params = read_pfgse_parameters_from_rwnmr_file(params_file)
	results = read_data_from_rwnmr_csvfile(results_file)

	rw_wrap = {
		'params': params,
		'results': results,
	}

	return rw_wrap

def plot_rw_diffusion_data(rw_wrap):
	rows = 3
	cols = 1
	fig, axs = plt.subplots(rows, cols, figsize=[6.0*cols, 3.0*rows]) #, sharex=True, sharey=True)

	axs[0].scatter(rw_wrap['results']['Time'], rw_wrap['results']['D_sat']/rw_wrap['params']['D_0'], marker='^', color='blue', label='S&T')
	if(rw_wrap['params']['Gradient direction'][0] != 0):
		axs[0].scatter(rw_wrap['results']['Time'], rw_wrap['results']['D_msdX']/rw_wrap['params']['D_0'], marker='v', color='red', label='msd_x')
	elif(rw_wrap['params']['Gradient direction'][1] != 0):
		axs[0].scatter(rw_wrap['results']['Time'], rw_wrap['results']['D_msdY']/rw_wrap['params']['D_0'], marker='v', color='red', label='msd_y')
	elif(rw_wrap['params']['Gradient direction'][2] != 0):
		axs[0].scatter(rw_wrap['results']['Time'], rw_wrap['results']['D_msdZ']/rw_wrap['params']['D_0'], marker='v', color='red', label='msd_z')
	axs[0].legend(loc='upper right', frameon=False)

	axs[1].scatter(rw_wrap['results']['Time'], rw_wrap['results']['D_sat']/rw_wrap['params']['D_0'], marker='^', color='blue', label='S&T')
	
	if(rw_wrap['params']['Gradient direction'][0] != 0):
		axs[2].scatter(rw_wrap['results']['Time'], rw_wrap['results']['D_msdX']/rw_wrap['params']['D_0'], marker='v', color='red', label='msd_x')
	if(rw_wrap['params']['Gradient direction'][1] != 0):
		axs[2].scatter(rw_wrap['results']['Time'], rw_wrap['results']['D_msdY']/rw_wrap['params']['D_0'], marker='v', color='red', label='msd_y')
	if(rw_wrap['params']['Gradient direction'][2] != 0):
		axs[2].scatter(rw_wrap['results']['Time'], rw_wrap['results']['D_msdZ']/rw_wrap['params']['D_0'], marker='v', color='red', label='msd_z')
	
	for ax in axs:
		ax.set_xlabel(r'Time' + '\t' + r'$[msec]$')
		ax.set_xlim([0.0, 1.01*rw_wrap['results']['Time'][-1]])
		ax.set_ylabel(r'$D(t)/D_{0}$')
		ax.set_ylim([0.0, 1.0])

	for ax in axs:
		ax.label_outer()

	plt.tight_layout()
	plt.show()
	return

def main():
	# Database dir
	db_dir = r'./db/'
	
	quit = False
	while(quit != True):
		rwnmr_folder = get_rwnmr_data_folder(db_dir)
		pfgse_folder = get_rwnmr_pfgse_folder(rwnmr_folder)
		rw_wrap = read_data_from_pfgse_folder(pfgse_folder)
		plot_rw_diffusion_data(rw_wrap)	

		quit_input = input('\n>> Want to plot again? (y/n): ')
		if not (quit_input == 'y' or quit_input == 'Y'):
			quit = True
	return

if __name__ == '__main__':
	main()

