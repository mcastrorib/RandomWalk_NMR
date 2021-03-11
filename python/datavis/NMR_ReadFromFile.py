import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from NMR_walker import NMR_walker

def read_diffusion_coefficient_from_file(filename):
	with open(filename, "r") as txt_file:
		# ignore first 2 lines
		size = 2
		for idx in range(size):
			next(txt_file)

		# read file lines
		read_line = True
		for line in txt_file:
			if(read_line):
				current_line = line.split(",")
				D0 = float(current_line[4])
				read_line = False

	return D0

def read_exposure_time_from_file(filename):
	with open(filename, "r") as txt_file:
		# ignore first 2 lines
		size = 2
		for idx in range(size):
			next(txt_file)

		# read file lines
		read_line = True
		for line in txt_file:
			if(read_line):
				current_line = line.split(",")
				Delta = float(current_line[1])
				read_line = False

	return Delta

def read_gyromagnetic_ratio_from_file(filename):
	with open(filename, "r") as txt_file:
		# ignore first 2 lines
		size = 2
		for idx in range(size):
			next(txt_file)

		# read file lines
		read_line = True
		for line in txt_file:
			if(read_line):
				current_line = line.split(",")
				gamma = float(current_line[3])
				read_line = False

	return gamma

def read_pulse_width_from_file(filename):
	with open(filename, "r") as txt_file:
		# ignore first 2 lines
		size = 2
		for idx in range(size):
			next(txt_file)

		# read file lines
		read_line = True
		for line in txt_file:
			if(read_line):
				current_line = line.split(",")
				width = float(current_line[2])
				read_line = False

	return width

def read_threshold_from_file(filename):
	with open(filename, "r") as txt_file:
		# ignore first 2 lines
		size = 2
		for idx in range(size):
			next(txt_file)

		# read file lines
		read_line = True
		for line in txt_file:
			if(read_line):
				current_line = line.split(",")
				Threshold = float(current_line[5])
				read_line = False

	return Threshold

def read_rhs_from_file(filename):
	rhs = []
	with open(filename, "r") as txt_file:
		# ignore first 6 lines
		size = 6
		for idx in range(size):
			next(txt_file)

		# read file lines
		for line in txt_file:
			current_line = line.split(",")
			rhs.append((-1) * float(current_line[3]))

	return rhs

def read_lhs_from_file(filename):
	lhs = []
	with open(filename, "r") as txt_file:
		# ignore first 6 lines
		size = 6
		for idx in range(size):
			next(txt_file)

		# read file lines
		for line in txt_file:
			current_line = line.split(",")
			lhs.append(float(current_line[2]))
	return lhs

def read_gradient_from_file(filename):
	gradient = []
	with open(filename, "r") as txt_file:
		# ignore first 6 lines
		size = 6
		for idx in range(size):
			next(txt_file)

		# read file lines
		for line in txt_file:
			current_line = line.split(",")
			gradient.append(float(current_line[1]))
	return gradient


def read_T2_from_file(file):
	# Read Input Data From File
	with open(file) as textFile:
		lines = [line.strip().split() for line in textFile]

	# Amplitude and T2 Arrays Construction
	bins = []
	amps = []
	for i in range(len(lines)):
		bins.append(float(lines[i][0]))
		amps.append(float(lines[i][1]))

	return bins, amps

def read_T2_from_csv(_filename, _skiplines = 0):
	bins = []
	amps = []

	with open(_filename, "r") as txt_file:
		# ignore first skip lines
		for idx in range(_skiplines):
			next(txt_file)

		# read file lines
		for line in txt_file:
			current_line = line.split(",")
			bins.append(float(current_line[0]))
			amps.append(float(current_line[1]))

	return bins, amps

def read_nmr_relaxation_from_file(file):
	# Read Data From File
	with open(file) as textFile:
		lines = [line.strip().split() for line in textFile]

	# Amplitude and T2 Arrays Construction
	amps = []
	echoes = []
	for i in range(len(lines)):
		echoes.append(float(lines[i][0]))
		amps.append(float(lines[i][1]))

	# Normalize amplitudes
	maxvalue = amps[0]

	for i in range(len(amps)):
		amps[i] /= maxvalue

	return echoes, amps

def read_analytic_qa_data_from_json(file):
	qa_data = []
	with open(file) as json_file:
		# load json object
		data = json.load(json_file)
		
		# read arrays
		qa_size = int(data['qa_samples'])
		qa_data = np.zeros(qa_size)
		for idx in range(qa_size):
			qa_data[idx] = float(data['qa'][idx])
	return qa_data

def read_analytic_echoes_from_json(file):
	echoes = []
	with open(file) as json_file:
		# load json object
		data = json.load(json_file)
		
		# read arrays
		qa_size = int(data['qa_samples'])
		echoes = np.zeros(qa_size)
		for idx in range(qa_size):
			echoes[idx] = float(data['echoes'][idx])
	return echoes

def read_analytic_data_from_json(file):
	# init json variables
	D0 = 0
	length = 0.0
	rho = 0.0
	time = 0.0
	qa_samples = 0.0
	qa_max = 0.0
	qa = []
	echoes = []

	# read json data
	with open(file) as json_file:
		# load json object
		json_data = json.load(json_file)
		
		# read scalars
		D0 = float(json_data["D0"])
		length = float(json_data["length"])
		rho = float(json_data["rho"])
		time = float(json_data["time"])
		qa_samples = int(json_data['qa_samples'])
		qa_max = float(json_data['qa_max'])

		# read arrays
		qa = np.zeros(qa_samples)
		for idx in range(qa_samples):
			qa[idx] = float(json_data['qa'][idx])

		echoes = np.zeros(qa_samples)
		for idx in range(qa_samples):
			echoes[idx] = float(json_data['echoes'][idx])
	
	# wrap json data into dictionary
	data = {}
	data["D0"] = D0
	data["length"] = length
	data["rho"] = rho
	data["delta"] = time
	data["qa_samples"] = qa_samples
	data["qa_max"] = qa_max
	data["qa"] = qa
	data["echoes"] = echoes
	
	return data

def read_number_of_walkers_from_file(_filename):
	'''
	this function count the number of lines from a NMR txt file @_filename with a NMR walker info
	in this format, the number of lines (except the first line) corresponds to the number of walkers in the simulation

	ARGS:
		_filename (str): txt file with NMR walker data

	RETURNS:
		n_walkers (int): Number of walkers data in file
	'''
	n_walkers = 0
	with open(_filename, "r") as txt_file:
		# ignore first 1 lines
		size = 1
		for idx in range(size):
			next(txt_file)

		# read file lines
		for line in txt_file:
			n_walkers += 1
	return n_walkers

def read_walker_from_file(_filename, _walker_id):
	'''
	this function read NMR walker data from a NMR txt file @_filename 
	the walker is identified by the parameter @_walker_id 

	ARGS:
		_filename (str): txt file with NMR walker data
		_walker_id (int): NMR walker identification

	RETURNS:
		walker (NMR_walker): The NMR_walker object with read data
	'''
	walker = NMR_walker()
	with open(_filename, "r") as txt_file:
		# ignore first (1 + w_id) lines
		size = 1 + _walker_id
		for idx in range(size):
			next(txt_file)

		# read file lines
		line = txt_file.readline()
		current_line = line.split(", ")

		# set walker data
		walker.set_x0(int(current_line[1]))
		walker.set_y0(int(current_line[2]))
		walker.set_z0(int(current_line[3]))
		walker.set_x(int(current_line[4]))
		walker.set_y(int(current_line[5]))
		walker.set_z(int(current_line[6]))
		walker.set_collisions(int(current_line[7]))
		walker.set_xirate(float(current_line[8]))

		# TODO: In future files, this entry will be an int64
		walker.set_rng_initial_seed(np.float64(current_line[9]))

		walker.reset_rng_seed()			
	return walker

def read_walker_position_info_from_file_to_array(_filename, _number_of_walkers, _col=-1):
	'''
	this function read NMR walker data from a NMR txt file @_filename 
	and returns an array with the collected info 

	ARGS:
		_filename (str): txt file with NMR walker data
		_number_of_walkers (int): the number of walkers in the sample 
		_col (int): identifies the info
		_col = 1 => initial position x
		_col = 2 => initial position y
		_col = 3 => initial position z
		_col = 4 => current/final position x
		_col = 5 => current/final position y
		_col = 6 => current/final position z
		_col = 7 => number of individual collisions

	RETURNS:
		data (np.array): An np.array with data
	'''

	if(_col > 0 and _col < 8):
		data = np.zeros(_number_of_walkers, dtype=np.int32)
		with open(_filename, "r") as txt_file:
			# ignore first line
			next(txt_file)	

			# read file lines
			idx = 0
			for line in txt_file:
				if(idx < _number_of_walkers):
					current_line = line.split(",")
					data[idx] = np.int32(current_line[_col])
				idx += 1

		return data
	else:
		print('Invalid collumn index. It must be [1,7].')
		data = np.zeros(1, dtype=np.int32)
		return data

def read_image_info_from_file(_filename):
	info = {}
	with open(_filename, 'r') as txt_file:
		# read 1st line
		line = txt_file.readline().split(': ')
		info['path'] = str(line[1])

		# read 2nd line
		line = txt_file.readline().split(': ')
		info['dim_x'] = int(line[1])

		# read 3rd line
		line = txt_file.readline().split(': ')
		info['dim_y'] = int(line[1])

		# read 4th line
		line = txt_file.readline().split(': ')
		info['dim_z'] = int(line[1])

		# read 5th line
		line = txt_file.readline().split(': ')
		info['voxel_size'] = float(line[1])

	return info

def read_console_log_data(_filename):
	D_times = []
	D_sat = []
	D_msd = []
	S_volume = []

	with open(_filename, 'r') as txt_file:
		line = txt_file.readline()

		# read while EOF msg 
		while(line != 'NMR_simulation object destroyed.\n'):
			# read next line
			line = txt_file.readline()

			# reading pfgse log
			if(line == 'running PFGSE simulation:\n'):
				# read times
				line = txt_file.readline().split()
				D_times.append(float(line[3]))

				# trash - ignore 4 lines 
				next(txt_file)
				next(txt_file)
				next(txt_file)
				next(txt_file)

				# read s&t data
				line = txt_file.readline().split()
				D_sat.append(float(line[3]))

				# read msd data
				line = txt_file.readline().split()
				D_msd.append(float(line[3]))

				# read superficial-volume relation
				line = txt_file.readline().split()
				S_volume.append(float(line[2]))

				# trash - ignore next line 
				next(txt_file)

	# data assembly
	consoleData = {}
	consoleData["D_times"] = D_times
	consoleData["D_sat"] = D_sat
	consoleData["D_msd"] = D_msd
	consoleData["S_volume"] = S_volume

	return consoleData

def read_data_from_pfgse_echoes_file(filename):
	# scallar data
	points = 0.0
	delta = 0.0
	width = 0.0
	giromagnet = 0.0
	diffusion_coefficient = 0.0
	rhs_threshold = 0.0
	
	# vector data
	gradient = []
	lhs = []
	rhs = []

	# data collection
	with open(filename, 'r') as txt_file:
		# ignore header 
		next(txt_file)

		# ignore variable naming line
		next(txt_file)

		# read and split next line
		line = txt_file.readline().split(',')

		points = int(line[0])
		delta = float(line[1])
		width = float(line[2])
		giromagnet = float(line[3])
		diffusion_coefficient = float(line[4])
		rhs_threshold = float(line[5])

		while(line != 'Stejskal-Tanner Equation\n'):
			line = txt_file.readline()

		# ignore variables naming line
		next(txt_file)

		# reading pfgse data
		for idx in range(points):
			# read and split next line
			line = txt_file.readline().split(',')

			gradient.append(float(line[1]))
			lhs.append(float(line[2]))
			rhs.append(float(line[3]))

	# data assembly to dictionary object
	pfgse_data = {}
	pfgse_data["points"] = points
	pfgse_data["delta"] = delta
	pfgse_data["width"] = width
	pfgse_data["giromagnet"] = giromagnet
	pfgse_data["diffusion_coefficient"] = diffusion_coefficient
	pfgse_data["rhs_threshold"] = rhs_threshold
	pfgse_data["gradient"] = gradient
	pfgse_data["lhs"] = lhs
	pfgse_data["rhs"] = rhs

	return pfgse_data

def read_pfgse_data_from_rwnmr_file(file):
	# scallar data
	points = 0.0
	delta = 0.0
	width = 0.0
	giromagnet = 0.0
	D0 = 0.0
	D_sat = 0.0
	D_msd = 0.0
	Msd = 0.0
	SVp = 0.0
	rhs_threshold = 0.0
	
	# vector data
	gradient = []
	Mkt = []
	lhs = []
	rhs = []

	# read params file
	with open(file[0], 'r') as txt_file:
		# ignore header 
		next(txt_file)

		# ignore variable naming line
		next(txt_file)

		# read and split next line
		line = txt_file.readline().split(',')

		points = int(line[0])
		delta = float(line[1])
		width = float(line[2])
		giromagnet = float(line[3])
		D0 = float(line[4])
		D_sat = float(line[5])
		D_msd = float(line[6])
		Msd = float(line[7])
		SVp = float(line[8])
		rhs_threshold = float(line[9])

	# read echoes file
	with open(file[1], 'r') as txt_file:
		# ignore header 
		next(txt_file)

		# ignore variable naming line
		next(txt_file)

		# reading pfgse data
		gradient = np.zeros(points)
		Mkt = np.zeros(points)
		lhs = np.zeros(points)
		rhs = np.zeros(points)
		for idx in range(points):
			# read and split next line
			line = txt_file.readline().split(',')

			gradient[idx] = float(line[1])
			Mkt[idx] = float(line[2])
			lhs[idx] = float(line[3])
			rhs[idx] = float(line[4])

	# data assembly to dictionary object
	pfgse_data = {
		"points": points,
		"delta": delta,
		"width": width,
		"giromagnet": giromagnet,
		"D0": D0,
		"D_sat": D_sat,
		"D_msd": D_msd,
		"Msd": Msd,
		"SVp": SVp,
		"rhs_threshold": rhs_threshold,
		"gradient": gradient,
		"Mkt": Mkt,
		"lhs": lhs,
		"rhs": rhs
	}	

	return pfgse_data

def read_msd_data_from_rwnmr_file(file):
	
	# scallar data
	msdX = 0.0
	msdY = 0.0
	msdZ = 0.0
	DmsdX = 0.0
	DmsdY = 0.0
	DmsdZ = 0.0

	# read params file
	with open(file, 'r') as txt_file:
		# ignore header 
		next(txt_file)

		# ignore variable naming line
		next(txt_file)

		# read and split next line
		line = txt_file.readline().split(',')
		msdX = float(line[0])
		msdY = float(line[1])
		msdZ = float(line[2])
		DmsdX = float(line[3])
		DmsdY = float(line[4])
		DmsdZ = float(line[5])

	# data assembly to dictionary object
	msd_data = {
		"msdX": msdX,
		"msdY": msdY,
		"msdZ": msdZ,
		"DmsdX": DmsdX,
		"DmsdY": DmsdY,
		"DmsdZ": DmsdZ
	}	

	return msd_data


def count_points_to_apply_lhs_threshold(pfgse_lhs, threshold):
	log_threshold = np.log(threshold)
	lhs_size = len(pfgse_lhs)
	idx = 0
	threshold_reached = False

	while(idx < lhs_size and threshold_reached == False):
		if(np.fabs(pfgse_lhs[idx]) < np.fabs(log_threshold)):
			idx += 1
		else:
			threshold_reached = True
	if(idx == lhs_size):
		return idx-1
	else:
		return idx

