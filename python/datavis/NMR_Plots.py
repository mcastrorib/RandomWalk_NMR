import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import math
from NMR_data import NMR_data
from NMR_PlotProperties import NMR_PlotProperties

def find_mean(_data):
    size = len(_data)
    data_sum = 0

    for index in range(size):
        data_sum += _data[index]

    return (data_sum/size)

def find_deviation(_data, _mean=0):
    size = len(_data)
    mu = _mean
    if(mu==0):
        mu = find_mean(_data)
    
    distance_sum = 0
    for index in range(size):
        distance_sum += (_data[index] - mu)*(_data[index] - mu)

    return np.sqrt(distance_sum/size)

def plot_histogram(_data, _y_axis_label = ' ', _numberOfBins=1024):
    # find data mean and standard deviation
    mu = find_mean(_data)
    sigma = find_deviation(_data)

    # create data histogram
    n, bins, patches = plt.hist(_data, _numberOfBins, density=True, facecolor='blue', alpha=0.5)

    # plot properties
    plt.xlabel(_y_axis_label)
    plt.ylabel('occurrence')
    plt.title('Histogram')

    # tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    
    # Show the major grid lines with dark grey lines
    plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)
    
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

    plt.show()
    return



def plot_magnetization_vs_gradient(LHS, gradient, D0, delta):
	for idx in range(len(D0)):
		M = []
		for point in range(len(LHS[idx])):
			M.append(np.exp(LHS[idx][point]))

		G2 = []
		for point in range(len(gradient[idx])):
			G2.append(gradient[idx][point]**2)

		# Create label
		plot_label = r'$\Delta=$' + str(delta[idx]) + r' $ms$, $D=$ ' + str(D0[idx]) + r' $um^{2}ms^{-1}$'

		# Plot Results
		# plt.style.use('seaborn-darkgrid') 
		plt.plot(G2, M, '-', label=plot_label)

		# Set plot texts
		plt.xlabel(r'$\bf{G}^{2},$ $\bf{gauss}^{2}\bf{cm}^{-2}$')
		plt.ylabel(r'$M(2\tau,G) / M(2\tau,0)$')
		plt.title(r'$\bf{Magnetization}$ vs. $\bf{Gradient}^{2}$')

		# Plot curve legend
		plt.legend(loc = "upper right")

		# Show the major grid lines with dark grey lines
		plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

		# Show the minor grid lines with very faint and almost transparent grey lines
		plt.minorticks_on()
		plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

	# Show image
	plt.show()

	return

def plot_lhs_vs_rhs(LHS, RHS, D0, delta, limit):
	columns = 2
	if(len(D0) == 1):
		columns = 1

	rows = math.ceil(len(D0)/columns)
	fig, axs = plt.subplots(rows, columns)
	fig.suptitle('Stejskal-Tanner Equation')

	for idx in range(len(D0)):
		M = []
		for point in range(len(LHS[idx])):
			M.append(np.exp(LHS[idx][point]))

		Adjust = []
		Adjust_RHS = []
		for point in range(len(RHS[idx])):
			if(RHS[idx][point] < limit[idx]):
				Adjust.append(np.exp(-D0[idx] * RHS[idx][point]))
				Adjust_RHS.append(RHS[idx][point])

		# set plot indexes
		plot_idx0 = idx // columns
		plot_idx1 = idx % columns

		# Create label
		plot_title = r"$\Delta = $" + str(delta[idx]) + r" $ms$"
		plot_label = r"$D = $" + str(D0[idx]) + r" $um²/ms$"
		plot_limit = r"Threshold = " + str(limit[idx]) + r" $gauss^{2} cm^{-2} s^{3}$"

		# Plot Results 
		axs[plot_idx0, plot_idx1].semilogy(RHS[idx], M, '+', color="red",label=plot_limit)
		axs[plot_idx0, plot_idx1].semilogy(Adjust_RHS, Adjust, '-', color="black",label=plot_label)
		# Set plot texts
		axs[plot_idx0, plot_idx1].set_title(plot_title)
		axs[plot_idx0, plot_idx1].legend(loc = "upper right")

		# Show the major grid lines with dark grey lines
		axs[plot_idx0, plot_idx1].grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

		# Show the minor grid lines with very faint and almost transparent grey lines
		axs[plot_idx0, plot_idx1].minorticks_on()
		axs[plot_idx0, plot_idx1].grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

		# add plot axis labels
		xaxis_label = r'$G^{2}(\gamma \delta)^{2}(\Delta - \delta /3)$'+ ', ' + r'$gauss^{2} cm^{-2} s^{3}$'
		yaxis_label = r'$M(2\tau,G)/M(2\tau,0)$'

	for ax in range(rows):
		axs[ax, 0].set(ylabel=yaxis_label)

	for ax in range(columns):
		axs[rows - 1, ax].set(xlabel=xaxis_label)

	# for ax in axs.flat:
	# ax.set(xlabel=xaxis_label, ylabel=r'$M(2\tau,G)/M(2\tau,0)$')

	# Hide x labels and tick labels for top plots and y ticks for right plots.
	# for ax in axs.flat:
	# 	ax.label_outer()

	# Show image
	plt.show()
	return

def plot_pfgse_bergman(_LHS, _gradient, _delta, _width, _gamma, title=""):
	k = []
	for idx in range(len(_gradient[0])):
		value = _gradient[0][idx] * 2*np.pi * _gamma[0] * _width[0] * 1.0e-5
		k.append(value)

	for idx in range(len(_delta)):

		# Create label
		plot_label = r'$\Delta = $' + str(_delta[idx]) + r'ms'

		# Plot Results
		# plt.style.use('seaborn-darkgrid') 
		plt.plot(k, _LHS[idx], '-', label=plot_label)

	# Set plot texts
	plt.title(title)
	plt.xlabel(r'wave vector k')
	plt.ylabel(r'ln\[$M(2\tau,G) / M(2\tau,0)$\]')

	# Plot curve legend
	plt.legend(loc = "best")

	# Show the major grid lines with dark grey lines
	plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

	# Show image
	plt.show()

def plot_pfgse_bergman_comparison(_LHS, _gradient, _delta, _width, _gamma, _samples, _colors, _markers, _labels, _title):
	rhos = len(_labels)
	
	k = []
	for idx in range(len(_gradient[0])):
		value = _gradient[0][idx] * 2*np.pi * _gamma[0] * _width[0] * 1.0e-5
		k.append(value)

	for idx in range(_samples):

		# Create label
		plot_label = r'$\Delta = $' + str(_delta[idx]) + r'ms'

		# Plot Results
		# plt.style.use('seaborn-darkgrid') 
		if(idx == 0):
			for rho in range(rhos):
				plt.plot(k, _LHS[rhos*idx + rho], _markers[rho], color=_colors[rho], label=_labels[rho])
		else:
			for rho in range(rhos):
				plt.plot(k, _LHS[rhos*idx + rho], _markers[rho], color=_colors[rho])


	# Set plot texts
	plt.title(_title)
	plt.xlabel(r'wave vector k')
	plt.ylabel(r'ln\[$M(2\tau,G) / M(2\tau,0)$\]')

	# Plot curve legend
	plt.legend(loc = "best")

	# Show the major grid lines with dark grey lines
	plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

	# Show image
	plt.show()
	return

def plot_pfgse_bergman_D_comparison(_delta, _Dt1, _Dt2, _samples, _D0, _edge, _labels, _title):
	rhos = len(_labels)
	
	x = []
	for idx in range(_samples):
		value = np.sqrt(_D0 * _delta[idx] / (_edge * _edge)) 
		x.append(value)

	data1 = []
	for idx in range(_samples):
		value = _Dt1[idx] / _D0
		data1.append(value)

	data2 = []
	for idx in range(_samples):
		value = _Dt2[idx] / _D0
		data2.append(value)

	plt.plot(x, data1, 'o', color='blue', label=_labels[0])
	plt.plot(x, data2, 'o', color='black', label=_labels[1])

	# Set plot texts
	plt.title(_title)
	plt.xlabel(r'$ [D_{0} t / a^{2}]^{1/2} $')
	plt.ylabel(r'$ D(t) / D_{0} $')

	plt.ylim(0.2, 1.05)

	# Plot curve legend
	plt.legend(loc = "best")

	# Show the major grid lines with dark grey lines
	plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

	# Show image
	plt.show()
	return

def plot_lhs_vs_rhs_comparing_resolutions(LHS, RHS, D0, delta, limit, resolutions):
	rows = math.ceil(len(D0))
	fig, axs = plt.subplots(rows)
	fig.suptitle('Stejskal-Tanner Equation')

	for idx in range(len(D0)):
		M = []
		for point in range(len(LHS[idx])):
			M.append(np.exp(LHS[idx][point]))

		Adjust = []
		Adjust_RHS = []
		for point in range(len(RHS[idx])):
			if(RHS[idx][point] < limit[idx]):
				Adjust.append(np.exp(-D0[idx] * RHS[idx][point]))
				Adjust_RHS.append(RHS[idx][point])

		
		# Create label
		plot_title = r"$Resolution = $" + str(resolutions[idx]) + r" $\mu m$"
		plot_label = r"$D = $" + str(D0[idx]) + r" $um²/ms$"
		plot_limit = r"Threshold = " + str(limit[idx]) + r" $gauss^{2} cm^{-2} s^{3}$"

		# Plot Results 
		axs[idx].semilogy(RHS[idx], M, '+', color="red",label=plot_limit)
		axs[idx].semilogy(Adjust_RHS, Adjust, '-', color="black",label=plot_label)
		# Set plot texts
		axs[idx].set_title(plot_title)
		axs[idx].legend(loc = "upper right")

		# Show the major grid lines with dark grey lines
		axs[idx].grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

		# Show the minor grid lines with very faint and almost transparent grey lines
		axs[idx].minorticks_on()
		axs[idx].grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

		# add plot axis labels
		xaxis_label = r'$G^{2}(\gamma \delta)^{2}(\Delta - \delta /3)$'+ ', ' + r'$gauss^{2} cm^{-2} s^{3}$'
		yaxis_label = r'$M(2\tau,G)/M(2\tau,0)$'

	for ax in range(rows):
		axs[ax].set(ylabel=yaxis_label)

	axs[rows - 1].set(xlabel=xaxis_label)

	# for ax in axs.flat:
	# ax.set(xlabel=xaxis_label, ylabel=r'$M(2\tau,G)/M(2\tau,0)$')

	# Hide x labels and tick labels for top plots and y ticks for right plots.
	# for ax in axs.flat:
	# 	ax.label_outer()

	# Show image
	plt.show()
	return

def autolabel(rects):
	"""Attach a text label above each bar in *rects*, displaying its height."""
	for rect in rects:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
					xy=(rect.get_x() + rect.get_width() / 2, height),
					xytext=(0, 3),  # 3 points vertical offset
					textcoords="offset points",
					ha='center', va='bottom')

	return

def plot_time_results():
	instances = ["small", "medium", "big"]
	mpi_ga_times = [1424.55, 2190.96, 20485.15]
	mpi_ga_dev = [593.84, 1426.46, 0]
	ga_small_times = [535.48, 3098.26, 6231.08]
	ga_small_dev = [86.17, 9.05, 0]
	ga_big_times = [1333.62, 11816.05, 23865.06]
	ga_big_dev = [478.76, 206.48, 0]

	x = np.arange(len(instances))  # the label locations
	print(x)
	width = 0.25  # the width of the bars

	plt.style.use('seaborn-darkgrid')
	fig, ax = plt.subplots()

	rects1 = ax.bar(x - width, mpi_ga_times, width, color='dodgerblue',label='with MPI')
	rects2 = ax.bar(x, ga_small_times, width, color='c',label='withou MPI (island size)')
	rects3 = ax.bar(x + width, ga_big_times, width, color='darkslategray', label='without MPI (total size)')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Time (s)')
	ax.set_xlabel('Instance')
	ax.set_title('GA runtimes')
	ax.set_xticks(x)
	ax.set_xticklabels(instances)
	ax.legend()


	# autolabel(rects1)
	# autolabel(rects2)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()


	# plt.savefig('results_time.png', transparent=True)
	plt.show()
	return

def barplot(bar_data, bar_labels,  x_ticklabels):
	print(bar_data)
	barsets = len(bar_data)
	bars_per_xtick = len(bar_labels)
	x = np.arange(len(x_ticklabels))  # the label locations
	print(x)
	width = 0.8 / bars_per_xtick  # the width of the bars
	x_dev = np.linspace(-0.4+(width*0.5), 0.4-(width*0.5), bars_per_xtick)

	# plt.style.use('seaborn-darkgrid')
	fig, ax = plt.subplots(figsize=[10,8])

	# plot y line
	# y_target = np.ones(4)
	# x_target = np.linspace(-2,10,4)
	# ax.plot(x_target, y_target, '--', color='red', zorder=3)

	rects = []
	for i in range(barsets):
		rect1 = ax.bar(x + x_dev[i], bar_data[i], width, label=bar_labels[i], zorder=2)
		rects.append(rect1)

	# plot normalized estimation
	# rects1 = ax.bar(x - 1.5*width, bar_data[0], width, color='c', label=bar_labels[0], zorder=2)
	# rects2 = ax.bar(x - 0.5*width, bar_data[1], width, color='b', label=bar_labels[1], zorder=2)
	# rects3 = ax.bar(x + 0.5*width, bar_data[2], width, color='y', label=bar_labels[2], zorder=2)
	# rects4 = ax.bar(x + 1.5*width, bar_data[3], width, color='g', label=bar_labels[3], zorder=2)



	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Estimated radii')
	
	ax.set_xlabel('Actual radii')
	ax.set_title('Pore radii estimation')
	ax.set_xticks(x)
	ax.set_xticklabels(x_ticklabels)
	plt.xlim([-1,5])
	# plt.ylim([0,1.7])
	ax.legend()


	# autolabel(rects1)
	# autolabel(rects2)

	# Show the major grid lines with dark grey lines
	plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

	plt.show()
	return

def scatterplot(scatter_dataX, scatter_dataY,  scatter_labels, markers, title):
	# plt.style.use('seaborn-darkgrid')
	fig, ax = plt.subplots(figsize=[10,8])

	# plot normalized estimation
	points = []
	for i in range(len(scatter_dataY)):
		pts = ax.scatter(scatter_dataX, scatter_dataY[i], s=100.0, marker=markers[i], label=scatter_labels[i], zorder=2)
		points.append(pts)



	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel(r'D(t), [$\mu$$m^{2}$/ms]')
	
	ax.set_xlabel(r'$(D_{0}\,t)^{(1/2)}$, [$\mu$m]')
	ax.set_title(title)
	# ax.set_xticks(x)
	# ax.set_xticklabels(x_ticklabels)
	plt.xlim([0.0, 1.2*max(scatter_dataX)])
	plt.ylim([0.0, 1.2*max(max(scatter_dataY[0]),max(scatter_dataY[1]))])
	ax.legend()


	# autolabel(rects1)
	# autolabel(rects2)

	# Show the major grid lines with dark grey lines
	plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

	plt.show()
	return

def plot_gen_results():
	instances = ["small", "medium", "big"]
	mpi_ga_gens = [69.8, 100, 263]
	mpi_ga_dev = [29.97, 63.16, 0]
	ga_small_gens = [61.4, 300, 300]
	ga_small_dev = [10.21, 0, 0]
	ga_big_gens = [38.33, 300, 300]
	ga_big_dev = [14.47, 0, 0]

	x = np.arange(len(instances))  # the label locations
	print(x)
	width = 0.25  # the width of the bars

	plt.style.use('seaborn-darkgrid')
	fig, ax = plt.subplots()

	rects1 = ax.bar(x - width, mpi_ga_gens, width, color='dodgerblue',label='with MPI')
	rects2 = ax.bar(x, ga_small_gens, width, color='c',label='withou MPI (island size)')
	rects3 = ax.bar(x + width, ga_big_gens, width, color='darkslategray', label='without MPI (total size)')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Generations')
	ax.set_xlabel('Instance')
	ax.set_title('GA Generations')
	ax.set_xticks(x)
	ax.set_xticklabels(instances)
	ax.legend()


	# autolabel(rects1)
	# autolabel(rects2)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.ylim(0,400)


	plt.savefig('results_gens.png', transparent=True)
	plt.show()

	return

def plot_T2_adjust(GA_bins, GA_amps, ref_bins, ref_amps, savefile="plotT2.png"):
	# Plot Results
	# plt.style.use('seaborn-darkgrid') 
	plt.semilogx(GA_bins, GA_amps,color='red', label = "GA")
	plt.semilogx(ref_bins, ref_amps, '--',color='blue', label = "Reference")

	# Plot curve legend
	plt.legend(loc = "upper left")

	# Set plot texts
	plt.xlabel('T2')
	plt.ylabel('Amplitudes')
	plt.title('T2 Distribution')

	# Show the major grid lines with dark grey lines
	plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

	# Show image
	plt.savefig(savefile, transparent=True)
	plt.show()
	return

def plot_rho(rho):
	begin = 0
	end = 1
	size = 100
	eps = np.linspace(begin, end, size)

	# Plot results
	plt.style.use('seaborn-darkgrid')    

	# plot mean and dev MPI rhos
	plt.plot(eps, rho, color='red')    


	# Set plot texts
	plt.xlabel('collisions rate')
	plt.ylabel('rho')
	plt.title('Superficial Relaxativity')

	# Define plot axis limits
	plt.xlim(0, 1)
	plt.ylim(0, 60)

	# Show the major grid lines with dark grey lines
	plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

	plt.show()
	return

def plot_rhos(rho_ga, rho_ref):
	begin = 0
	end = 1
	size = 100
	eps = np.linspace(begin, end, size)

	# Plot results
	# plt.style.use('seaborn-darkgrid')    

	# plot mean and dev MPI rhos
	plt.plot(eps, rho_ga, color='red', label="GA")
	plt.plot(eps, rho_ref, '--', color='blue', label="Reference")    


	# Set plot texts
	plt.legend(loc="upper right")
	plt.xlabel('XI rate')
	plt.ylabel('Rho')
	plt.title('Superficial Relaxativity')

	# Define plot axis limits
	plt.xlim(0, 1)
	plt.ylim(0, 60)

	# Show the major grid lines with dark grey lines
	plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

	# plt.savefig(filename, transparent=True)
	plt.show()
	return

def plot_nmr_relaxation(echoes, amps):
	plt.plot(echoes, amps)

	# Plot curve legend
	plt.legend(loc="upper right")

	# Set plot texts
	plt.xlabel('Time (ms)')
	plt.ylabel('Magnetization')
	plt.title('NMR Relaxation')

	# Show the major grid lines with dark grey lines
	plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

	# Show image
	plt.show()


def plot_NMR_data(_dataList, _plotProps):
	# set plot size
	plt.figure(figsize=(_plotProps.fig_size[0], _plotProps.fig_size[1]), dpi=_plotProps.dpi)

	for dataset in _dataList:
		plt.plot(dataset.x_data, dataset.y_data, dataset.marker, color=dataset.color, label=dataset.label)

	# Plot curve legend
	plt.legend(loc="best")

	# Set plot texts
	plt.xlabel(_plotProps.xlabel)
	plt.ylabel(_plotProps.ylabel)
	plt.title(_plotProps.title)

	# Define plot axis limits
	if(len(_plotProps.xlim) == 2):
		plt.xlim(_plotProps.xlim[0], _plotProps.xlim[1])

	if(len(_plotProps.ylim) == 2):
		plt.ylim(_plotProps.ylim[0], _plotProps.ylim[1])

	# Show the major grid lines with dark grey lines
	plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

	# Show image
	if(_plotProps.show):
		plt.show()

	return plt

def semilogy_NMR_data(_dataList, _plotProps):
	# set plot size
	plt.figure(figsize=(_plotProps.fig_size[0], _plotProps.fig_size[1]), dpi=_plotProps.dpi)

	for dataset in _dataList:
		plt.semilogy(dataset.x_data, dataset.y_data, dataset.marker, color=dataset.color, label=dataset.label)

	# Plot curve legend
	plt.legend(loc="best")

	# Set plot texts
	plt.xlabel(_plotProps.xlabel)
	plt.ylabel(_plotProps.ylabel)
	plt.title(_plotProps.title)

	# Define plot axis limits
	if(len(_plotProps.xlim) == 2):
		plt.xlim(_plotProps.xlim[0], _plotProps.xlim[1])

	if(len(_plotProps.ylim) == 2):
		plt.ylim(_plotProps.ylim[0], _plotProps.ylim[1])

	# Show the major grid lines with dark grey lines
	plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

	# Show image
	if(_plotProps.show):
		plt.show()

	return plt

def semilogx_NMR_data(_dataList, _plotProps):
	# set plot size
	plt.figure(figsize=(_plotProps.fig_size[0], _plotProps.fig_size[1]), dpi=_plotProps.dpi)

	for dataset in _dataList:
		plt.semilogx(dataset.x_data, dataset.y_data, dataset.marker, color=dataset.color, label=dataset.label)

	# Plot curve legend
	plt.legend(loc="best")

	# Set plot texts
	plt.xlabel(_plotProps.xlabel)
	plt.ylabel(_plotProps.ylabel)
	plt.title(_plotProps.title)

	# Define plot axis limits
	if(len(_plotProps.xlim) == 2):
		plt.xlim(_plotProps.xlim[0], _plotProps.xlim[1])

	if(len(_plotProps.ylim) == 2):
		plt.ylim(_plotProps.ylim[0], _plotProps.ylim[1])

	# Show the major grid lines with dark grey lines
	plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

	# Show image
	if(_plotProps.show):
		plt.show()

	return plt

def logplot_NMR_data(_dataList, _plotProps):
	for dataset in _dataList:
		plt.log(dataset.x_data, dataset.y_data, dataset.marker, color=dataset.color, label=dataset.label)

	# Plot curve legend
	plt.legend(loc="best")

	# Set plot texts
	plt.xlabel(_plotProps.xlabel)
	plt.ylabel(_plotProps.ylabel)
	plt.title(_plotProps.title)

	# Define plot axis limits
	if(len(_plotProps.xlim) == 2):
		plt.xlim(_plotProps.xlim[0], _plotProps.xlim[1])

	if(len(_plotProps.ylim) == 2):
		plt.ylim(_plotProps.ylim[0], _plotProps.ylim[1])

	# Show the major grid lines with dark grey lines
	plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

	# Show image
	if(_plotProps.show):
		plt.show()

	return plt

def subplot_semilogy(data, props, title):
	columns = 1
	rows = len(data)
	
	fig, axs = plt.subplots(rows, columns,figsize=(props[0].fig_size[0], props[0].fig_size[1]), dpi=props[0].dpi)
	fig.suptitle(title)

	for row in range(rows):
		# Plot data
		for idx in range(len(data[row])):
			if(data[row][idx].marker == '-'):
				axs[row].semilogy(data[row][idx].x_data, data[row][idx].y_data, data[row][idx].marker, color=data[row][idx].color, label=data[row][idx].label)
			else:
				axs[row].semilogy(data[row][idx].x_data, data[row][idx].y_data, data[row][idx].marker, markersize=2.5, color=data[row][idx].color, label=data[row][idx].label)

		# Set plot texts
		axs[row].set_title(props[row].title, loc='right')
		
		if(row==0):
			axs[row].legend(loc = "lower left", prop={'size': 8})

		axs[row].set(ylabel=props[row].ylabel)
		axs[row].set(xlabel=props[row].xlabel)
		if(len(props[row].xlim) == 2):
			axs[row].set_xlim(props[row].xlim)
		if(len(props[row].ylim) == 2):
			axs[row].set_ylim(props[row].ylim)

		# Show the major grid lines with dark grey lines
		axs[row].grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

		# Show the minor grid lines with very faint and almost transparent grey lines
		axs[row].minorticks_on()
		axs[row].grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)


	

	# Hide x labels and tick labels for top plots and y ticks for right plots.
	for ax in axs.flat:
		ax.label_outer()
	
	plt.show()
	return

def plot_walker_positions(i_data, f_data, plot_edge=True):
	# initialize 3D plot fig
	fig = plt.figure(figsize=plt.figaspect(0.25), dpi=100)
	fig.set_facecolor('brown')
	viewport = [[-10,30],[-10,30],[0,900]]

	# -------------
	# first subplot
	ax = fig.add_subplot(131, projection='3d')

	# Points (s=point_size, c=color, cmap=colormap)
	ax.scatter(i_data[:,0], i_data[:,1], i_data[:,2], zdir='y', s=10.0, c='red', marker='o', alpha=0.01) 
	# ax.axis('off')
	ax.set_title(r'$\bf{initial \, positions}$', color='white')
	ax.set_xlim(viewport[0])
	# x and z are swicthed so that z direction enters page
	ax.set_ylim(viewport[2])
	ax.set_zlim(viewport[1])
	ax.set_facecolor('grey')
	ax.grid(False) 
	ax.w_xaxis.pane.fill = False
	ax.w_yaxis.pane.fill = False
	ax.w_zaxis.pane.fill = False
	# ax.xaxis.set_ticklabels([])
	# ax.yaxis.set_ticklabels([])
	# ax.zaxis.set_ticklabels([])

	# --------------
	# second subplot
	ax = fig.add_subplot(132, projection='3d')

	# Points (s=point_size, c=color, cmap=colormap)
	ax.scatter(f_data[:,0], f_data[:,1], f_data[:,2], zdir='y', s=10.0, c='red', marker='o', alpha=0.01) 
	ax.set_title(r'$\bf{final \, positions}$', color='white')
	ax.set_xlim(viewport[0])
	# x and z are swicthed so that z direction enters page
	ax.set_ylim(viewport[2])
	ax.set_zlim(viewport[1])
	ax.set_facecolor('grey')
	ax.grid(False) 
	ax.w_xaxis.pane.fill = False
	ax.w_yaxis.pane.fill = False
	ax.w_zaxis.pane.fill = False
	# ax.xaxis.set_ticklabels([])
	# ax.yaxis.set_ticklabels([])
	# ax.zaxis.set_ticklabels([])

	# -------------
	# third subplot
	ax = fig.add_subplot(133, projection='3d')

	# Points (s=point_size, c=color, cmap=colormap)
	ax.scatter(i_data[:,0], i_data[:,1], i_data[:,2], zdir='y', s=1.0, c='blue', marker='o', alpha=0.5)
	ax.scatter(f_data[:,0], f_data[:,1], f_data[:,2], zdir='y', s=1.0, c='red', marker='*', alpha=0.5)  


	# Edges
	if(plot_edge and len(i_data) == len(f_data)):
		for i in range(len(i_data)):
			xe = [i_data[i,0], f_data[i,0]]
			ye = [i_data[i,1], f_data[i,1]]
			ze = [i_data[i,2], f_data[i,2]]
			# x and z are swicthed so that z direction enters page
			ax.plot(xe, ze, ye, c='red', alpha=0.1)

	ax.set_title(r'$\bf{displacement}$', color='white')
	ax.set_xlim(viewport[0])
	# x and z are swicthed so that z direction enters page
	ax.set_ylim(viewport[2])
	ax.set_zlim(viewport[1])
	ax.set_facecolor('grey')
	ax.grid(False) 
	ax.w_xaxis.pane.fill = False
	ax.w_yaxis.pane.fill = False
	ax.w_zaxis.pane.fill = False
	# ax.xaxis.set_ticklabels([])
	# ax.yaxis.set_ticklabels([])
	# ax.zaxis.set_ticklabels([])


	plt.show()
	return

def plot_walker_displacement_histogram(_dx, _dy, _dz, _numberOfBins=128):
	# find data mean and standard deviation
	mu = find_mean(_dx)
	sigma = find_deviation(_dx)

	# create data histogram
	n, bins, patches = plt.hist(_dx, _numberOfBins, density=True, facecolor='blue')#, alpha=0.5)

	# plot properties
	plt.xlabel('Voxels')
	plt.ylabel('Occurrence')
	plt.title('Displacement Histogram')

	# tweak spacing to prevent clipping of ylabel
	plt.subplots_adjust(left=0.15)

	# Show the major grid lines with dark grey lines
	plt.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	plt.minorticks_on()
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

	plt.show()
	return

def plot_walker_placement_histograms(_x0, _xF, _img_info):
	# read img data
	image_dim = [_img_info['dim_x'], _img_info['dim_y'], _img_info['dim_z']]
	voxel_size = _img_info['voxel_size']

	#  init matplotlib figure
	walkers, rows = np.shape(_x0)
	columns = 1
	fig, axs = plt.subplots(rows, columns, figsize=[10,10], dpi=100)
	# fig.set_facecolor('lightblue')
	fig.suptitle('Walker Position Histogram')
	subtitle = ['direction x', 'direction y', 'direction z']

	# red_amps = np.array((3,_numberOfBins))
	# red_bins = np.array((3,_numberOfBins))
	# blue_amps = np.array((3,_numberOfBins))
	# blue_amps = np.array((3,_numberOfBins))

	for row in range(rows):
		# set histogram range
		range_min = 0.0
		range_max = image_dim[row] * voxel_size
		hist_range = [range_min, range_max]
		number_of_bins = int(image_dim[row])

		# get histogram final (red) and initial (blue) state data
		red_amps, red_bins = np.histogram(_xF[:, row], number_of_bins, hist_range)
		blue_amps, blue_bins = np.histogram(_x0[:, row], number_of_bins, hist_range)
		
		# set the width and center of histogram bars
		width = red_bins[1] - red_bins[0]  
		center = width * 0.5
		cbins = np.zeros(number_of_bins)
		for i in range(number_of_bins):
			cbins[i] = center + red_bins[i]
			red_amps[i] = -red_amps[i]		
		
		# draw histograms
		rects1 = axs[row].bar(cbins, blue_amps, width, color='navy', label='initial state', zorder=3)
		rects2 = axs[row].bar(cbins, red_amps, width, color='maroon', label='final state', zorder=3)

		# plot properties
		axs[row].set_facecolor('ghostwhite')
		axs[row].set_ylabel('Occurrence')
		axs[row].set_title(subtitle[row], loc='right')
		axs[row].legend(loc='upper right')

		axs[row].axhline(0, color='black', lw=1.5)
		axs[row].set_xlim([range_min, range_max])
		axs[row].set_ylim([-1.25*max(blue_amps), 1.25*max(blue_amps)])
		axs[row].yaxis.set_ticklabels([])
		axs[row].xaxis.set_major_locator(MaxNLocator(integer=True))

		# Show the major grid lines with dark grey lines
		axs[row].grid(b = True, which = 'major', color = 'lightsteelblue', linestyle = '-', alpha = 0.75, zorder=0)

		# Show the minor grid lines with very faint and almost transparent grey lines
		axs[row].minorticks_on()
		axs[row].grid(b = True, which = 'minor', color = 'white', linestyle = '--', alpha = 0.5, zorder=0)

	# tweak spacing to prevent clipping of ylabel
	# plt.subplots_adjust(left=0.15)
	
	plt.show()
	return

def plot_walker_displacement_histograms(_x0, _xF, _labels, _img_info, _numberOfBins=256):
	# read img data
	image_dim = [_img_info['dim_x'], _img_info['dim_y'], _img_info['dim_z']]
	voxel_size = _img_info['voxel_size']

	#  init matplotlib figure
	size, rows, walkers = np.shape(_x0)
	columns = 1
	fig, axs = plt.subplots(rows, columns, figsize=[10,10], dpi=100)
	# fig.set_facecolor('lightblue')
	fig.suptitle('Walker Displacement Histogram')
	subtitle = ['direction x', 'direction y', 'direction z']

	# get displacements for each dataset
	datasets = []
	for time in range(size):
		displacement = np.zeros((rows, walkers))
		for row in range(rows):
			for walker in range(walkers):
				displacement[row, walker] = _xF[time][row, walker] - _x0[time][row, walker]
		datasets.append(displacement)

	for row in range(rows):
		# set histogram range
		range_max = image_dim[row] * voxel_size
		range_min = -range_max
		hist_range = [range_min, range_max]
		number_of_bins = int(2*image_dim[row])

		# get histogram final state data
		true_amps = []
		true_bins = []
		for time in range(size):
			amps, bins = np.histogram(datasets[time][row,:], number_of_bins, hist_range)
			true_amps.append(amps)
			true_bins.append(bins)

		# set the width and center of histogram bars
		width = true_bins[0][1] - true_bins[0][0]  
		center = width * 0.5
		cbins = np.zeros(number_of_bins)
		for i in range(number_of_bins):
			cbins[i] = center + true_bins[0][i]
		
		# draw histograms
		for time in range(size):
			axs[row].bar(cbins, true_amps[time], width, alpha=0.7, label=_labels[time], zorder=3) #, color='navy')
			

		# plot properties
		axs[row].set_facecolor('ghostwhite')
		axs[row].set_ylabel('Occurrence')
		axs[row].set_title(subtitle[row], loc='right')
		axs[row].legend(loc='upper right')

		axs[row].axhline(0, color='darkred', lw=1.25)
		# axs[row].xaxis.set_ticklabels([])
		# axs[row].yaxis.set_ticklabels([])
		axs[row].set_xlim([-200,200])
		axs[row].set_ylim([0, max(true_amps[0])])

		# Show the major grid lines with dark grey lines
		axs[row].grid(b = True, which = 'major', color = 'lightsteelblue', linestyle = '-', alpha = 0.75, zorder=0)

		# Show the minor grid lines with very faint and almost transparent grey lines
		axs[row].minorticks_on()
		axs[row].grid(b = True, which = 'minor', color = 'white', linestyle = '--', alpha = 0.5, zorder=0)

	# tweak spacing to prevent clipping of ylabel
	# plt.subplots_adjust(left=0.15)
	
	plt.show()
	return

def plot_least_squares_adjust(x_data, y_data, lsa_B, delta, lsa_points, lsa_threshold=0, title=''):
	# plt.style.use('seaborn-darkgrid')
	fig, ax = plt.subplots(figsize=[10,8])

	Mx = np.zeros(len(y_data))
	My = np.zeros(len(y_data))
	for idx in range(len(y_data)):
		My[idx] = np.exp(y_data[idx])
		Mx[idx] = (-1)*x_data[idx]

	y_adjust = np.zeros(lsa_points)
	x_adjust = np.zeros(lsa_points)
	for idx in range(lsa_points):
		y_adjust[idx] = np.exp(lsa_B * x_data[idx])
		x_adjust[idx] = (-1)*x_data[idx] 

	# Create label
	plot_title = r"$\Delta = $" + str(delta) + r" $ms$"
	plot_label = r"$D = $" + "{:.4f}".format(lsa_B) + r" $um²/ms$"
	
	# Plot Results
	ax.semilogy(Mx, My, 'x', color="blue", alpha=0.5)
	ax.semilogy(x_adjust, y_adjust, '-', color="red", zorder=3, label=plot_label)
	# Set plot texts
	ax.set_title(title + "\n" + plot_title)
	ax.legend(loc = "upper right")

	# Show the major grid lines with dark grey lines
	ax.grid(b = True, which = 'major', color = '#666666', linestyle = '-', alpha = 0.5)

	# Show the minor grid lines with very faint and almost transparent grey lines
	ax.minorticks_on()
	ax.grid(b = True, which = 'minor', color = '#999999', linestyle = '--', alpha = 0.2)

	# add plot axis labels
	xaxis_label = r'$G^{2}(\gamma \delta)^{2}(\Delta - \delta /3)$' #+ ', ' + r'$gauss^{2} cm^{-2} s^{3}$'
	yaxis_label = r'$M(\Delta,G)/M(\Delta,0)$'

	ax.set(ylabel=yaxis_label)
	ax.set(xlabel=xaxis_label)

	if(lsa_threshold > 0):
		ax.set(ylim=[0.95*lsa_threshold, 1.0])
		ax.set(xlim=[0.0, 1.05*Mx[lsa_points]])

	# for ax in axs.flat:
	# ax.set(xlabel=xaxis_label, ylabel=r'$M(2\tau,G)/M(2\tau,0)$')

	# Hide x labels and tick labels for top plots and y ticks for right plots.
	# for ax in axs.flat:
	# 	ax.label_outer()

	# Show image
	plt.show()
	return