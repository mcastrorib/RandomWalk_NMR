// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

// include OpenMP for multicore implementation
#include <omp.h>

//include
#include "NMR_defs.h"
#include "NMR_Simulation.h"
#include "NMR_cpmg.h"
#include "../Laplace/tikhonov.h"
#include "../Laplace/include/nmrinv_core.h"
#include "../Walker/walker.h"
#include "../FileHandler/fileHandler.h"

using namespace cv;
using namespace std;

NMR_cpmg::NMR_cpmg( NMR_Simulation &_NMR,  
                    cpmg_config _cpmgConfig,
                    int _mpi_rank,
                    int _mpi_processes) : NMR(_NMR),
                                        CPMG_config(_cpmgConfig),
                                        mpi_rank(_mpi_rank),
                                        mpi_processes(_mpi_processes)
{
	// vectors object init
    vector<double> T2_bins();
    vector<double> T2_amps();

    (*this).setExposureTime(this->CPMG_config.getObservationTime());
    (*this).setMethod(this->CPMG_config.getMethod());
    (*this).set();
}

void NMR_cpmg::set()
{
	(*this).setName();
	(*this).createDirectoryForData();
	(*this).setNMRTimeFramework();
}

void NMR_cpmg::run()
{
    (*this).run_simulation();
    (*this).applyLaplace();
    (*this).save();
}


void NMR_cpmg::setName()
{
	string big_delta = std::to_string((int) this->exposureTime) + "-" 
					   + std::to_string(((int) (this->exposureTime * 100)) % 100);
	this->name = "/NMR_cpmg_" + big_delta + "ms";
}

void NMR_cpmg::createDirectoryForData()
{
	string path = this->NMR.getDBPath();
    createDirectory(path, this->NMR.simulationName + "/" + this->name);
    this->dir = (path + this->NMR.simulationName + "/" + this->name);
}


void NMR_cpmg::setNMRTimeFramework()
{
	cout << endl << "running CPMG simulation:" << endl;
	this->NMR.setTimeFramework((*this).getExposureTime());
	cout << "CPMG exposure time: " << (*this).getExposureTime() << " ms";
	cout << " (" << this->NMR.simulationSteps << " RW-steps)" << endl;
	this->NMR.mapSimulation();
	// this->NMR.updateRelaxativity(rho); but what rho to adopt?
}

// -- Simulations
void NMR_cpmg::run_simulation()
{
    if((*this).getMethod() == "image-based")
    {
        // Choose method considering GPU usage
        if(this->NMR.getGPU()) (*this).simulation_img_cuda();
        else (*this).simulation_img_omp();

    } else
    if((*this).getMethod() == "histogram")
    {
        vector<double> rho;
        rho = this->NMR.rwNMR_config.getRho();
        if(this->NMR.rwNMR_config.getRhoType() == "uniform") this->NMR.createPenaltiesVector(rho[0]);
        else if(this->NMR.rwNMR_config.getRhoType() == "sigmoid") this->NMR.createPenaltiesVector(rho);
        (*this).simulation_histogram();
    }

    // Normalize global energy decay 
    this->NMR.normalizeEnergyDecay();
}

void NMR_cpmg::simulation_img_omp()
{
    double begin_time = omp_get_wtime();

    cout << "initializing CPMG-NMR simulation... ";

    // reset walker's initial state with omp parallel for
// #pragma if(NMR_OPENMP) omp parallel for private(id) shared(walkers)
    for (uint id = 0; id < this->NMR.walkers.size(); id++)
    {
        this->NMR.walkers[id].resetPosition();
        this->NMR.walkers[id].resetSeed();
        this->NMR.walkers[id].resetEnergy();
    }

    // reset vector to store energy decay
    this->NMR.resetGlobalEnergy();
    this->NMR.globalEnergy.reserve(this->NMR.numberOfEchoes);

    // get initial energy state
    double energySum = 0.0;
    for (uint id = 0; id < this->NMR.walkers.size(); id++)
    {
        energySum += this->NMR.walkers[id].energy;
    }
    this->NMR.globalEnergy.push_back(energySum);


    energySum = 0.0;
    uint id, step;
    for (uint echo = 0; echo < this->NMR.numberOfEchoes; echo++)
    {

        // walkers walk some steps with omp parallel
        // #pragma omp parallel for if(NMR_OPENMP) private(id, step) shared(walkers, bitBlock, simulationSteps)
        for (id = 0; id < this->NMR.numberOfWalkers; id++)
        {
            for (step = 0; step < this->NMR.stepsPerEcho; step++)
            {
                this->NMR.walkers[id].walk(this->NMR.bitBlock);
            }
        }

        // collect energy from all walkers with omp reduce
        energySum = 0.0; // reset energy summation
        // #pragma omp parallel for if(NMR_OPENMP) reduction(+:energySum) private(id) shared(walkers)
        for (id = 0; id < this->NMR.numberOfWalkers; id++)
        {
            energySum += this->NMR.walkers[id].energy;
        }

        //energySum = energySum / (double)numberOfWalkers;
        this->NMR.globalEnergy.push_back(energySum);
    }

    cout << "Completed.";
    double finish_time = omp_get_wtime();
    printElapsedTime(begin_time, finish_time);
}


void NMR_cpmg::simulation_histogram()
{
    double begin_time = omp_get_wtime();
    cout << "initializing RW-NMR hist simulation... ";

    if(this->NMR.histogramList.size() == 0)  
    {
        cout << "could not start simulation without histogram list" << endl;
        return;
    }
    if(this->NMR.penalties == NULL)
    {
        cout << "could not start simulation without penalties vector" << endl;
        return;
    }

    // initialize energyDistribution array
    double *energyDistribution = NULL;
    energyDistribution = new double[this->NMR.histogram.size];


    // reset vector to store energy decay
    this->NMR.resetGlobalEnergy();
    this->NMR.globalEnergy.push_back(1.0);

    // histogram simulation main loop    
    for(int hst_ID = 0; hst_ID < this->NMR.histogramList.size(); hst_ID++)
    {
        for(uint id = 0; id < this->NMR.histogram.size; id++)
        {
            energyDistribution[id] = this->NMR.globalEnergy.back() * this->NMR.histogramList[hst_ID].amps[id];
        }

        double energyLvl;
        int eBegin = this->NMR.histogramList[hst_ID].firstEcho;
        int eEnd = this->NMR.histogramList[hst_ID].lastEcho;
        for(uint echo = eBegin; echo < eEnd; echo++)
        {
            // apply penalties
            for(uint id = 0; id < this->NMR.histogram.size; id++)
            {
                energyDistribution[id] *= this->NMR.penalties[id];
            }

            // get global energy
            energyLvl = 0.0;
            for(uint id = 0; id < this->NMR.histogram.size; id++)
            {
                energyLvl += energyDistribution[id];
            }

            // add to global energy vector
            this->NMR.globalEnergy.push_back(energyLvl);
        }
    }

    // cut out unnecessary computations 
    // this->globalEnergy.resize(this->numberOfEchoes);

    delete[] energyDistribution;
    energyDistribution = NULL;

    cout << "Completed.";
    double finish_time = omp_get_wtime();
    printElapsedTime(begin_time, finish_time);
}

// apply laplace inversion explicitly
void NMR_cpmg::applyLaplace()
{   
    // check if energy decay was done
    if(this->NMR.globalEnergy.size() == 0) 
    {
        cout << "no data available, could not apply inversion." << endl;
        return; 
    }

    // reset T2 distribution from previous simulation
    if(this->T2_bins.size() > 0) this->T2_bins.clear();
    if(this->T2_amps.size() > 0) this->T2_amps.clear();

    // get copy of decay info and remove first elements
    vector<double> decay = this->NMR.getGlobalEnergy();
    vector<double> times = this->NMR.getDecayTimes();
    times.erase(times.begin());
    decay.erase(decay.begin());     

    NMRInverterConfig nmr_inv_config(this->CPMG_config.getMinT2(), 
                                     this->CPMG_config.getMaxT2(),
                                     this->CPMG_config.getUseT2Logspace(),
                                     this->CPMG_config.getNumT2Bins(),
                                     this->CPMG_config.getMinLambda(),
                                     this->CPMG_config.getMaxLambda(),
                                     this->CPMG_config.getNumLambdas(),
                                     this->CPMG_config.getPruneNum(),
                                     this->CPMG_config.getNoiseAmp());

    NMRInverter nmr_inverter;
    nmr_inverter.set_config(nmr_inv_config, times);
    nmr_inverter.find_best_lambda(decay.size(), decay.data());
    nmr_inverter.invert(decay.size(), decay.data());
    for(uint i = 0; i < nmr_inverter.used_t2_bins.size(); i++)
    {
        this->T2_bins.push_back(nmr_inverter.used_t2_bins[i]);
        this->T2_amps.push_back(nmr_inverter.used_t2_amps[i]);
    }
}

// -- Savings
void NMR_cpmg::save()
{
	double time = omp_get_wtime();
    cout << "saving results...";
    
    if(this->CPMG_config.getSaveDecay()) 
    {
        this->NMR.saveEnergyDecay(this->dir);
	}
    
    if(this->CPMG_config.getSaveCollisions())
    {
        this->NMR.saveWalkerCollisions(this->dir);
    }

    if(this->CPMG_config.getSaveHistogram())
    {
        this->NMR.saveHistogram(this->dir);
    }    

    if(this->CPMG_config.getSaveHistogramList())
    {
        this->NMR.saveHistogramList(this->dir);
    }

    // write cpmg data
	(*this).writeResults();

	time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}

void NMR_cpmg::writeResults()
{
	// string big_delta = std::to_string((int) this->exposureTime) + "-" 
	// 				   + std::to_string(((int) (this->exposureTime * 100)) % 100);
	// string tiny_delta = std::to_string((int) this->pulseWidth);
	// string gamma = std::to_string((int) this->giromagneticRatio);
	string filename = this->dir + "/cpmg_T2.txt";

	ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    const size_t num_points = T2_bins.size();
    file << "NMRT2_bins, NMRT2_amps" << endl;
    for (int idx = 0; idx < num_points; idx++)
    {
        file << this->T2_bins[idx] << ", " << this->T2_amps[idx] << endl;
    }
    
    file.close();
}
