import os.path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_data import NMR_data
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import NMR_PlotProperties
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

# simulation parameters
Dp = 2.5
length_a = 10.0
radius = 5.0
phi = 0.47
rho = 0.0
resolution = 0.1
walkers = '100M'
shift = 0
sim_time_scale = length_a**2 / Dp
relaxation_strength = rho*length_a / Dp

# find data files 
db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/bergman_test/'
db_dir += r'Mkt_rho=' + str(rho) + r'/'
db_dir += r'phi=' + str(phi) + '_res=' + str(resolution) + '/' 
db_dir += r'a=' + str(length_a) + r'um_r=' + str(radius) + r'um/data/'
sim_dir = r'PFGSE_NMR_periodic_a=' + str(length_a) + '_r=' + str(radius) + '_rho=' + str(rho) + '_res=' + str(resolution) + '_shift=' + str(shift) + '_w='+ walkers + '/'

# assembly complete path to data directory
complete_path = db_dir + sim_dir

# List all data files in directory
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

# read and process data from simulation files
sim_datasets = []
for file in data_files:
	sim_datasets.append(read_pfgse_data_from_rwnmr_file(file))

sim_data_x = []
sim_data_y = []
sim_times = []
sim_labels = []
for idx in range(number_of_sim_files):
	sim_times.append(sim_datasets[idx]['delta'])
	sim_data_y.append(np.array(sim_datasets[idx]['lhs']))
	sim_data_x.append(get_ka_vec(sim_datasets[idx]['gradient'], sim_datasets[idx]['giromagnet'], sim_datasets[idx]['width'], length_a, sim_datasets[idx]['points']))
	sim_labels.append('rw: ' + str(sim_datasets[idx]['delta'] / sim_time_scale)  + ' x a²/D0')

sim_xlim = []
for idx in range(number_of_sim_files):
	sim_xlim.append(sim_data_x[idx][-1])

# analytical solution parameters
ansol_Dp = 2.5
ansol_a = 10.0
ansol_r = 5.0
ansol_time_scale = ansol_a**2 / ansol_Dp

# find analytic solution data files
db_dir = r'/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_1.0/db'
ansol_dir = r'/Analytical_Periodic_Media_phi=0.47_rho=0.0_N=5'
ansol_files = [db_dir + ansol_dir + '/' + file for file in os.listdir(db_dir + ansol_dir)]
number_of_ansol_files = len(ansol_files)

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
		if(kval <= sim_xlim[data_idx]):
			ka_vec.append(kval)
			count += 1
	ansol_data_x.append(np.array(ka_vec))

	Mkt = np.zeros(count)
	M0 = dataset['Mkt'][0]
	for i in range(count):
		Mkt[i] = dataset['Mkt'][i] / M0
	ansol_data_y.append(Mkt)
	data_idx += 1

# plot data files
line1 = r'$\bf{Periodic\,Array}$: resolution = ' + str(resolution) + r' $\mu$m/voxel'
line2 = r'$\rho a/D$= ' + str(relaxation_strength) + r', a = ' + str(length_a) + r'$\mu$m, $D_0$ = ' + str(Dp)  + r' $\mu$m²/s, Walkers=' + walkers
plot_title =  line1 + '\n' + line2
plt.figure(figsize=([10,9]))
for idx in range(number_of_sim_files):
	plt.semilogy(sim_data_x[idx], np.exp(sim_data_y[idx]), 'o', markersize=2.5, label=sim_labels[idx])

# plot data files
for idx in range(number_of_sim_files):
	plt.semilogy(ansol_data_x[idx], ansol_data_y[idx], '-', label=ansol_labels[idx])

plt.title(plot_title)
plt.legend(loc='upper right')
plt.axvline(x=1*np.pi, color="black", linewidth=1.0)
plt.axvline(x=2*np.pi, color="black", linewidth=1.25)
plt.axvline(x=3*np.pi, color="black", linewidth=1.0)
plt.axvline(x=4*np.pi, color="black", linewidth=1.25)
plt.xlim([0,1.35*sim_xlim[0]])
plt.ylim([1.e-5,1.e0])
# Show the major grid lines with dark grey lines
plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)
plt.tight_layout()
plt.show()