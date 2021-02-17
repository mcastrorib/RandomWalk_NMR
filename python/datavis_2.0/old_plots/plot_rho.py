import numpy as np
import matplotlib.pyplot as plt
from NMR_ReadFromFile import *
from NMR_Plots import *


def relaxativity(params, eps):
    K1 = params[0]
    A1 = params[1]
    e1 = params[2]
    B1 = params[3]
    K2 = params[4]
    A2 = params[5]
    e2 = params[6]
    B2 = params[7]

    shape1 = A1 + ((K1 - A1) / (1 + np.exp((-1) * B1 * (eps - e1))))

    shape2 = A2 + ((K2 - A2) / (1 + np.exp((-1) * B2 * (eps - e2))))

    return (shape1 + shape2)

def find_dev(mean, values):
    size = len(values)
    sum = 0
    for value in values:
        sum += (mean - value) ** 2

    return np.sqrt(sum/size)


if __name__ == '__main__':
    begin = 0
    end = 1
    size = 100
    # XI rate domain
    eps = np.linspace(begin, end, size)

    # & GA_reference 
    # synth_synth
    # params_ga = [14.574,10.2444,0.5819,460.982,12.7893,14.787,0.5937,329.912]

    # synth_filter
    # params_ga = [2.2885,6.3326,0.2721,551.672,3.1183,18.6083,0.1821,287.55]
    # params_ga = [3.9326,21.426,0.1818,248.512,3.5664,3.4899,0.3001,98.035]

    # rough_rough
    # params_ga = [28.3168,11.934,0.5473,360.05,25.907,13.0238,0.7356,444.695]

    # rhough_synth
    # params_ga = [1.6389,20.1603,0.1972,552.66,19.1686,22.3469,0.6045,496.16]
    # params_ga = [14.4932,12.3714,0.6385,355.294,20.2174,12.8669,0.6721,405.772]
    # params_ga = [7.2002,25.43,0.1979,218.025,14.2183,16.8129,0.4937,164.292]
    params_ga = [1.7546,14.2722,0.0401,151.736,21.5326,39.653,0.2279,262.749]

    # rho function - sigmoid construction
    # & GA_reference
    rho_ga = np.zeros(size)
    for i in range(size):
        rho_ga[i] = relaxativity(params_ga, eps[i])

    # & GA_solution  
    params_ref = [12.5, 12.5, 0.25, 5.0, 12.5, 12.5, 0.75, 5.0]
    rho_ref = np.zeros(size)
    for i in range(size):
        rho_ref[i] = relaxativity(params_ref, eps[i])

    plot_rhos(rho_ga, rho_ref)
