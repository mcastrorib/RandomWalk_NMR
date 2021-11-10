import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

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


def read_T2_decay_from_rwnmr_file(file):
	# Read Data From File
	with open(file, 'r') as txt_file:
		lines = [line.strip().split(', ') for line in txt_file]

	# Amplitude and T2 Arrays Construction
	offset = 0
	header = True
	while (header):
		try:
			float(lines[offset][0])
			header = False
		except:
			offset += 1

	columns = lines[offset - 1]
	ncols = len(columns)
	size = len(lines) - offset
	column_data = []
	for column in columns:
		column_data.append(np.zeros(size))
	
	for i in range(size):
		for col in range(ncols):
			column_data[col][i] = float(lines[i + offset][col])

	T2_decay = {}
	for col in range(ncols):
		T2_decay[columns[col]] = column_data[col]

	return T2_decay

def read_data_from_rwnmr_csvfile(file):
	df = pd.read_csv(file)
	columns = df.columns
	
	data = {}
	for col in range(columns.size):
		data[columns[col]] = df[columns[col]].to_numpy()

	return data

def read_T2_distribution_from_rwnmr_file(file):
	# Read Data From File
	with open(file, 'r') as txt_file:
		lines = [line.strip().split(',') for line in txt_file]

	# Amplitude and T2 Arrays Construction
	offset = 0
	if(lines[0][0][-4:] == 'bins' or lines[0][1][-4:] == 'amps'):
		offset = 1
		
	size = len(lines) - offset
	bins = np.zeros(size)
	amps = np.zeros(size)
	for i in range(size):
		bins[i] = float(lines[i + offset][0])
		amps[i] = float(lines[i + offset][1])

	T2_dist = {
		'bins': bins,
		'amps': amps
	}
	
	return T2_dist