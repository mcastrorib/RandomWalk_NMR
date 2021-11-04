import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NMR_data import NMR_data
from NMR_ReadFromFile import *
from NMR_Plots import *
from NMR_PlotProperties import NMR_PlotProperties
from LeastSquaresRegression import LeastSquaresRegression

def get_Dt_analytical(dir_path, Dfree=2.5, a=1.0):
    # collect via os list
    pfgse_echoes_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    
    # read data from PFGSE echoes files
    pfgse_data = []
    for file in pfgse_echoes_files:
        pfgse_data.append(read_analytic_data_from_json(dir_path + '/' + file))

    # set data size
    Dfree = pfgse_data[0]["D0"]
    pfgse_data_size = len(pfgse_data)

    # Get 'delta' times
    D_delta = []
    for dataset in range(pfgse_data_size):
        D_delta.append(pfgse_data[dataset]["delta"])

    # Get D_sat
    D_callaghan = []
    # D_callaghan.append(pfgse_data[0]["D0"])
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
        lsa_threshold = 0.9
        lhs_min = min(dataY)

        while(np.log(lsa_threshold) < lhs_min):
            lsa_threshold += 0.1 

        lsa_points = count_points_to_apply_lhs_threshold(dataY, lsa_threshold)
        # print("threshold = ", lsa_threshold)
        # print("points = ", lsa_points)

        # create and solve least square regression
        lsa = LeastSquaresRegression()		
        lsa.config(dataX, dataY, 10)
        lsa.solve()
        D_callaghan.append(lsa.results()["B"]/Dfree)
        # lsa_results = lsa.results()
        # D_callaghan = lsa_results["B"]
        # print("t= {:.2f}, D(t)= {:.4f}".format(delta, D_callaghan[dataset]))

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
        t_adim.append(np.sqrt(time * Dfree)/a)

    data = {}
    data["label"] = "S&T - Callaghan"
    data["x_data"] = t_adim
    data["y_data"] = D_callaghan
    data["marker"] = "-"
    data["color"] = 'red'

    # scatterplot(x_data, y_data, labels, markers, title) 
    return data

def get_Dt_from_simulation(dir_path, Dfree=2.5, a=1.0):
    
    # collect via os list
    pfgse_echoes_filename = r"/PFGSE_echoes.txt"
    dirlist = [ item for item in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, item)) ]
    pfgse_echoes_files = []
    for item in dirlist:
        filepath = dir_path + r'/' + item + pfgse_echoes_filename
        if(os.path.isfile(filepath)):
            pfgse_echoes_files.append(filepath)
    
    # read data from PFGSE echoes files
    pfgse_data = []
    for file in pfgse_echoes_files:
        pfgse_data.append(read_data_from_pfgse_echoes_file(file))

    pfgse_data_size = len(pfgse_data)
    t_min = (10.0 * pfgse_data[0]["width"])  # only works if all data uses same width

    # Get D_msd
    D_msd = []
    for dataset in range(pfgse_data_size):
        if(pfgse_data[dataset]["delta"] > t_min):
            D_msd.append(pfgse_data[dataset]["diffusion_coefficient"]/Dfree)
    
    # Get D_sat
    D_sat = []
    t_adim = []
    for dataset in range(pfgse_data_size):
        # accept only when time >> width condition is satisfied
        if(pfgse_data[dataset]["delta"] > t_min):
            # apply threshold for least squares adjust
            lsa_threshold = 0.9
            lhs_min = min(pfgse_data[dataset]["lhs"])
            
            while(np.log(lsa_threshold) < lhs_min):
                lsa_threshold += 0.1 

            lsa_points = count_points_to_apply_lhs_threshold(pfgse_data[dataset]["lhs"], lsa_threshold)
            # print("threshold = ", lsa_threshold)
            # print("points = ", lsa_points)

            # create and solve least square regression
            lsa = LeastSquaresRegression()		
            lsa.config(pfgse_data[dataset]["rhs"], pfgse_data[dataset]["lhs"], lsa_points)
            lsa.solve()
            D_sat.append(lsa.results()["B"]/Dfree)
            # lsa_results = lsa.results()
            # D_sat = lsa_results["B"]
            # print("t= {:.2f}, D(t)= {:.4f}".format(pfgse_data[dataset]["delta"], lsa.results()["B"]))

            # # plot adjust
            # lsa_title = plot_title
            # plot_least_squares_adjust(
            # 	pfgse_data[dataset]["rhs"], 
            # 	pfgse_data[dataset]["lhs"], 
            # 	D_sat[dataset], 
            # 	pfgse_data[dataset]["delta"], 
            # 	lsa_points, 
            # 	lsa_threshold, 
            # 	lsa_title)
        else:
            print("cannot use simulation data because pfgse time (t = {:.2f}) is not greater than the pulse width.".format(pfgse_data[dataset]["delta"]))

    # get adimensional time
    for dataset in range(pfgse_data_size):
        if(pfgse_data[dataset]["delta"] > t_min):
            t_adim.append(np.sqrt(pfgse_data[dataset]["delta"] * Dfree)/a)
    
    # sat data
    sat_data = {}
    sat_data["label"] = "S&T - Simulation"
    sat_data["marker"] = "o"
    sat_data["color"] = "blue"
    sat_data["x_data"] = t_adim.copy()
    sat_data["y_data"] = D_sat

    # msd data
    msd_data = {}
    msd_data["label"] = "msd - Simulation"
    msd_data["marker"] = "x"
    msd_data["color"] = "green"
    msd_data["x_data"] = t_adim.copy()
    msd_data["y_data"] = D_msd

    return sat_data, msd_data

def get_pore_sv_relation(sqrt_D0t_data, Dt_data, D0):
    data_size = len(Dt_data)

    print("Sqrt(D0t) = \n", sqrt_D0t_data)
    print("D(t) = \n", Dt_data)

    # apply threshold for least squares adjust
    Sv_factor = ( -2.25 * np.sqrt(np.pi) ) / (D0)
    Sv = []
    idx = 0
    min_points = 3
    Bold = 0.0
    Bnew = 0.0
    while(idx < (data_size - min_points)):
        
        # create and solve least square regression
        lsa_points = data_size - idx    
        lsa = LeastSquaresRegression()		
        lsa.config(sqrt_D0t_data, Dt_data, lsa_points)
        lsa.solve()
        Bnew = lsa.results()["B"]
        Sv.append(Bnew * Sv_factor)
        
        # print results
        # convergence = 0.0
        convergence = ((Bnew - Bold) / Bold)
        # convergence = np.abs((Bnew - Bold))

        A = lsa.results()["A"]

        print("{} points: A = {:.4f}, B = {:.4f},  Sv = {:.4f}, radius ~= {:.4f}, convergence = {:.4f}".format(lsa_points, 
                                                                                                                A,
                                                                                                                Bnew,
                                                                                                                Sv[idx],
                                                                                                                (3.0/Sv[idx]), 
                                                                                                                convergence ))
        
        # increment index
        idx += 1
        Bold = Bnew
    
    
    return

def plot_Dt_data(data_for_plot, title=''):  
    # wrap data
    labels = []
    markers = []
    colors = []
    x_data = []
    y_data = []
    for dataset in data_for_plot:
        labels.append(dataset["label"])
        markers.append(dataset["marker"])
        colors.append(dataset["color"])

        time = np.array(dataset["x_data"])
        Dts = np.array(dataset["y_data"])
        inds = time.argsort()
        sortedTime = time[inds]
        sortedDts = Dts[inds]
        x_data.append(sortedTime)
        y_data.append(sortedDts)


    print("\nCallaghan Sv via D(t):")
    get_pore_sv_relation(x_data[0], y_data[0], 2.5)

    print("\nRW-S&T Sv via D(t):")
    get_pore_sv_relation(x_data[1], y_data[1], 2.5)
    
    print("\nRW-msd via D(t):")
    get_pore_sv_relation(x_data[2], y_data[2], 2.5)

    dataplot(x_data, y_data, labels, markers, colors, title)
    
    return

def insert_t0(data_x, data_y, _D0):
    data_x.insert(0, 0.0)
    data_y.insert(0, _D0)
    return

def main():
    # simulation parameters
    walkers_str = r'100k'
    edge_length = [1.0, 2.5, 5.0, 10.0, 20.0]
    edge_length_str = ['1um', '2.5um', '5um', '10um', '20um']
    edge_idx = 1
    pore_type = 'isolated_sphere'
    rho = 0.0
    relaxation_strength = 0
    Dfree = 2.5
    res = 1.0
    insert_t0_D0 = False 

    # Root dir
    root_dir = r"/home/matheus/Documentos/doutorado_ic/tese/saved_data/callaghan_test"

    # -- Analytical data
    # Exp dir
    exp_dir = r"/analytical/data/50_logspace_times"
    sim_dir = r"/Dts_SphericalPore_a=" + str(edge_length[edge_idx])+ r"_rho=" + str(rho) + r"_D=" + str(Dfree)

    # Set simulation data files
    sim_data_dir = root_dir + exp_dir + sim_dir
    callaghan_data = get_Dt_analytical(sim_data_dir, Dfree, edge_length[edge_idx])

    # -- Simulation data
    # Exp dir
    exp_dir = r"/Dt_recover/r="+ str(edge_length[edge_idx]) 
    sim_dir = r"/PFGSE_NMR_sphere_r=" + str(edge_length[edge_idx]) + r"_rho=" + str(rho) + r"_res=1.0_shift=0_w=" + walkers_str
		
    # Set simulation data files
    sim_data_dir = root_dir + exp_dir + sim_dir
    rw_data_sat, rw_data_msd = get_Dt_from_simulation(sim_data_dir, Dfree, edge_length[edge_idx])

    # -- Plot results
    # set title
    line1 = r'$\bf{Spherical\,Pore}$'
    line2 = r'a = ' + str(edge_length[edge_idx]) + r' $\mu$m, ' + r'$\rho a/D$= ' + str(relaxation_strength) + r', D = ' + str(Dfree)  + r' $\mu$m²/s'
    plot_title =  line1 + '\n' + line2

    # -- Wrap data
    datasets = [callaghan_data, rw_data_sat, rw_data_msd]
    
    # -- Insert artificial t=0.0, D(t=0.0)=D0
    if(insert_t0_D0):
       for idx in range(len(datasets)):
           D0 = Dfree
           insert_t0(datasets[idx]["x_data"], datasets[idx]["y_data"], Dfree)
    
    # plot data
    plot_Dt_data(datasets, plot_title)

    return
if __name__ == '__main__':
    main()