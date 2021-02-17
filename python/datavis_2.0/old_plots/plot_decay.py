import numpy as np
import matplotlib.pyplot as plt
from NMR_ReadFromFile import *
from NMR_Plots import *


if __name__ == '__main__':
    # Read Data From File
    decay_file_a = "../data/tiny_3D_test_0/NMR_decay.txt"
    decay_file_b = "../data/tiny_3D_test_1/NMR_decay.txt"
    echoes_a, amps_a = read_nmr_relaxation_from_file(decay_file_a)
    echoes_b, amps_b = read_nmr_relaxation_from_file(decay_file_b)

    plot_nmr_relaxation(echoes_a, amps_a)
    plot_nmr_relaxation(echoes_b, amps_b) 
