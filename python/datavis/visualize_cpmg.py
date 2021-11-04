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

def get_rwnmr_cpmg_folder(rwnmr_dir):
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
	user_input = input('choose the CPMG folder: ')
	print('\n')
	n_folders = len(valid_folders)
	fId = int(user_input)
	if not (fId < 0 or fId > n_folders):
		return os.path.join(rwnmr_dir, valid_folders[fId])
	else:
		return r''

def read_data_from_cpmg_folder(cpmg_folder):
	decay_file = os.path.join(cpmg_folder, r'NMR_decay.txt')
	dist_file = os.path.join(cpmg_folder, r'cpmg_T2.txt')

	T2_decay = read_T2_decay_from_rwnmr_file(decay_file)
	T2_dist = read_T2_distribution_from_rwnmr_file(dist_file)

	rw_wrap = {
		'T2_decay': T2_decay,
		'T2_dist': T2_dist,
	}

	return rw_wrap

def plot_rw_T2_data(rw_wrap):
	fig, axs = plt.subplots(2, 1, figsize=[9.6, 7.2])
	axs[0].plot(rw_wrap['T2_decay']['times'], rw_wrap['T2_decay']['amps'])
	axs[0].set_xlabel(r'exposure time' + '\t' + r'$[msec]$')
	axs[0].set_xlim([0.0, 1.01*rw_wrap['T2_decay']['times'][-1]])
	axs[0].set_ylabel(r'normalized magnetization')
	axs[0].set_ylim([0.0, 1.01*rw_wrap['T2_decay']['amps'].max()])

	axs[1].semilogx(rw_wrap['T2_dist']['bins'], rw_wrap['T2_dist']['amps'])
	axs[1].set_xlabel(r'relaxation times $(T_{2})$')
	axs[1].set_xlim([0.9*rw_wrap['T2_dist']['bins'][0], 1.1*rw_wrap['T2_dist']['bins'][-1]])
	axs[1].set_ylabel(r'amplitudes')
	axs[1].set_ylim([0.0, 1.1*rw_wrap['T2_dist']['amps'].max()])
	
	plt.tight_layout()
	plt.show()
	return

def main():
	# Database dir
	db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/db/'
	
	quit = False
	while(quit != True):
		rwnmr_folder = get_rwnmr_data_folder(db_dir)
		cpmg_folder = get_rwnmr_cpmg_folder(rwnmr_folder)
		rw_wrap = read_data_from_cpmg_folder(cpmg_folder)
		plot_rw_T2_data(rw_wrap)	

		quit_input = input('\n>> Want to plot again? (y/n): ')
		if not (quit_input == 'y' or quit_input == 'Y'):
			quit = True
	return

if __name__ == '__main__':
	main()

