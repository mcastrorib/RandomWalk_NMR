import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_data import NMR_data
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import NMR_PlotProperties
from LeastSquaresRegression import LeastSquaresRegression

def main():
    # simulation parameters
    walkers_str = r'100k'
    edge_length = [10.0] #, 2.5, 5.0, 10.0, 20.0]
    edge_length_str = ['10um'] #, '2.5um', '5um', '10um', '20um']
    pore_type = 'isolated_sphere'
    rho = 0.0
    relaxation_strength = 0
    Dfree = 2.50
    res = 1.0
    use_raw_data = True


    # Root dir
    root_dir = r"/home/matheus/Documentos/doutorado_ic/tese/saved_data/callaghan_test"
    # root_dir = r'/home/matheus/Documentos/doutorado_ic/tese/saved_data/free_diffusion')

    # Exp dir
    exp_dir = r"/analytical/data/40_logspace_times"
    sim_dir = r"/Dts_SphericalPore_a=10.0_rho=0.0_D=2.5"

    # Plot title
    line1 = r'$\bf{Spherical\,Pore}$: resolution = ' + str(res) + r' $\mu$m/voxel'
    line2 = r'$\rho a/D$= ' + str(relaxation_strength) + r', $D_{0}$ = ' + str(Dfree)  + r' $\mu$m²/s, walkers=' + walkers_str
    plot_title =  line1 + '\n' + line2

    # Set simulation data files

    # Callaghan test
    sim_data_dir = root_dir + exp_dir + sim_dir


    # collect via os list
    pfgse_echoes_files = [f for f in os.listdir(sim_data_dir) if os.path.isfile(os.path.join(sim_data_dir, f))]
    
    # read data from PFGSE echoes files
    pfgse_data = []
    for file in pfgse_echoes_files:
        pfgse_data.append(read_analytic_data_from_json(sim_data_dir + '/' + file))

    pfgse_data_size = len(pfgse_data)

    # Get 'delta' times
    D_delta = []
    for dataset in range(pfgse_data_size):
        D_delta.append(pfgse_data[dataset]["delta"])

    # Get D_sat
    D_callaghan = []
    for dataset in range(pfgse_data_size):
        dataY = []
        for y in pfgse_data[dataset]["echoes"]:
            dataY.append(np.log(y))
        
        dataX = []
        alength = pfgse_data[dataset]["length"]
        delta = pfgse_data[dataset]["delta"]
        for x in pfgse_data[dataset]["qa"]:
            dataX.append( (-1) * delta * ((2 * np.pi * (x / alength))**2) )

        # apply threshold for least squares adjust
        lsa_threshold = 0.5
        lhs_min = min(dataY)

        while(np.log(lsa_threshold) < lhs_min):
            lsa_threshold += 0.1 

        lsa_points = count_points_to_apply_lhs_threshold(dataY, lsa_threshold)
        print("threshold = ", lsa_threshold)
        print("points = ", lsa_points)

        # create and solve least square regression
        lsa = LeastSquaresRegression()		
        lsa.config(dataX, dataY, lsa_points)
        lsa.solve()
        D_callaghan.append(lsa.results()["B"])
        # lsa_results = lsa.results()
        # D_callaghan = lsa_results["B"]
        print("t= {:.2f}, D(t)= {:.4f}".format(delta, D_callaghan[dataset]))

        # plot adjust
        # lsa_title = plot_title
        # plot_least_squares_adjust(
        #     dataX, 
        #     dataY, 
        #     D_callaghan[dataset], 
        #     pfgse_data[dataset]["delta"], 
        #     lsa_points, 
        #     lsa_threshold, 
        #     lsa_title)

    # plot data
    t_adim = []
    for time in D_delta:
        t_adim.append(np.sqrt(time * Dfree))

    print(len(t_adim))
    print(len(D_callaghan))

    labels = ["S&T - Callaghan"]
    x_data = t_adim
    y_data = [D_callaghan]
    markers = 'o'
    title = plot_title + '\n' + r'a = ' + str(10) + r'$\mu$m'
    scatterplot(x_data, y_data, labels, markers, title) 

    return

if __name__ == '__main__':
    main()

    # dirpath =  r"/home/matheus/Documentos/Doutorado IC/tese/saved_data/callaghan_test/relaxation_strength=0/a=2.5um/isolated_sphere/using_q/res=1.0/"
    # datadir = r"PFGSE_NMR_sphere_r=2.5_rho=0.0_res=1.0_shift=0_w=10M/"
    # filename = r"consoleLog"
    # filepath = dirpath + datadir + filename
    # data = read_console_log_data(filepath)
    # print(data)