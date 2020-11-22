import numpy as np
import matplotlib.pyplot as plt
from NMR_ReadFromFile import *
from NMR_Plots import *

def compute_pfgse_k_value(gradient, tiny_delta, giromagnetic_ratio):
	return (tiny_delta * 1.0e-03) * (2 * np.pi * giromagnetic_ratio * 1.0e+06) * (gradient * 1.0e-08)

def compute_analytical_magnetization(k, rho, a, D, delta):
	expoent1 = (-1) * (3 * rho / D) * delta
	expoent1 = 1
	expoent2 = (-0.2) * k*k * a*a
	return (np.exp(expoent1) * np.exp(expoent2))

def get_k2a2_evolution(k, a):
	k2a2 = []
	for idx in range(len(k)):
		value = k[idx]*k[idx] * a*a
		k2a2.append(value)

	return k2a2

def get_analytical_decay(k, rho, a, D, delta):
	M = []
	M0 = compute_analytical_magnetization(0.0, rho, a, D, delta)
	for idx in range(len(k)):
		M.append(compute_analytical_magnetization(k[idx], rho, a, D, delta))

	ln_M_M0 = []
	for idx in range(len(M)):
		value = np.log(M[idx]/M0)
		ln_M_M0.append(value)

	return ln_M_M0

def plot_decay(k2a2, ln_M_M0):
	
	# Plot Results
	# plt.style.use('seaborn-darkgrid') 
	plt.plot(k2a2, ln_M_M0, '-')

	# Set plot texts
	plt.xlabel(r'$\bf{(ka)}^{2}$')
	plt.ylabel(r'ln $M(2\tau,G) / M(2\tau,0)$')

	# Show the major grid lines with dark grey lines
	plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

	# Show image
	plt.show()

	return

def main():
	g0 = 0.0
	gF = 20.0
	samples = 128
	gradient = np.linspace(g0, gF, samples)
	gamma = 42.576
	tiny_delta = 4.0
	k = []
	for idx in range(len(gradient)):
		value = compute_pfgse_k_value(gradient[idx], tiny_delta, gamma)
		k.append(value)

	print("g = \n", gradient)
	print("k = \n", k)

	# analytical values
	rho = 1.0           # um/s
	rho *= 1.0e-06	    # convert to m/s	
	a = 5               # um
	a *= 1.0e-06        # convert to m 
	D = 2 
	D *= 1.0e-09	
	delta = 200 
	delta *= 1.0e-03
	
	k2a2 = get_k2a2_evolution(k, a)
	decay = get_analytical_decay(k, rho, a, D, delta)

	# print("k2a2 = \n", k2a2)
	# print("ln(M/M0) = \n", decay)
	plot_decay(k2a2, decay)

	return

if __name__ == '__main__':
	main()

