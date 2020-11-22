import numpy as np
import matplotlib.pyplot as plt
from NMR_ReadFromFile import *
from NMR_Plots import *	


if __name__ == '__main__':
	image_ref = "synthrock_rough"
	image_ga = "synthrock"	
	ga_dir = image_ref + "-" + image_ga 

	# construct directories
	ref_data_dir = "../../../REPORTS/CSV_FILES/" + image_ref
	ga_data_dir = "../../../REPORTS/GENETIC/" + ga_dir 

	# add ga cases
	ga_data_dir += "/GA3_mixed/" 

	# files
	ref_filename = "/" + image_ref + "_LAPLACE.csv"
	ga_filename = "/Creature_1_Laplace.csv"

	# assembly filepaths
	ref_filepath = ref_data_dir + ref_filename
	ga_filepath = ga_data_dir + ga_filename

	# read data from csv file
	ref_T2, ref_Amps = read_T2_from_csv(ref_filepath, 1)
	ga_T2, ga_Amps = read_T2_from_csv(ga_filepath, 2)

	save_dir = "../../../REPORTS/plots/T2/"
	save_file = "T2_adjust_" + ga_dir + "3.png"
	save_path = save_dir + save_file
	plot_T2_adjust(ga_T2, ga_Amps, ref_T2, ref_Amps, save_path)


	