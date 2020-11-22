// include OpenCV core functions
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <random>
#include <vector>
#include <string>
#include <cmath>

// include OpenMP for multicore implementation
#include <omp.h>

//include
#include "../Walker/walker.h"
#include "../BitBlock/bitBlock.h"
#include "../RNG/xorshift.h"
#include "../FileHandler/fileHandler.h"
#include "NMR_Simulation.h"

using namespace std;
using namespace cv;

// GPU methods
// simulations in CUDA language for GPU application

// mapping simulation using bitblock data structure
void NMR_Simulation::mapSimulation_OMP()
{
    double begin_time = omp_get_wtime();
    cout << "initializing mapping simulation... ";
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        this->walkers[id].resetPosition();
        this->walkers[id].resetSeed();
        this->walkers[id].resetCollisions();
        this->walkers[id].resetTCollisions();
    }

    // initialize list of collision histograms
    (*this).initHistogramList(); 

    // loop throughout list
    for(int hst_ID = 0; hst_ID < this->histogramList.size(); hst_ID++)
    {
        int eBegin = this->histogramList[hst_ID].firstEcho;
        int eEnd = this->histogramList[hst_ID].lastEcho;
        for (uint id = 0; id < this->numberOfWalkers; id++)
        {
            for(uint echo = eBegin; echo < eEnd; echo++)
            {
                for (uint step = 0; step < this->stepsPerEcho; step++)
                {
                    this->walkers[id].map(bitBlock);
                }
            }
        }

        int steps = this->stepsPerEcho * (eEnd - eBegin);
        (*this).createHistogram(hst_ID, steps);

        for (uint id = 0; id < this->numberOfWalkers; id++)
        {
            this->walkers[id].tCollisions += this->walkers[id].collisions;
            this->walkers[id].resetCollisions();
        }
    }

    // recover walkers collisions from total sum and create a global histogram
    for (uint id = 0; id < this->numberOfWalkers; id++)
    {
        this->walkers[id].collisions = this->walkers[id].tCollisions;   
    }
    (*this).createHistogram();

    cout << "Completed.";
    double finish_time = omp_get_wtime();
    printElapsedTime(begin_time, finish_time);
}

// NMR_RW simulation using bitblock data structure
void NMR_Simulation::walkSimulation_OMP()
{
    double begin_time = omp_get_wtime();

    cout << "initializing RW-NMR simulation... ";

    // reset walker's initial state with omp parallel for
// #pragma if(NMR_OPENMP) omp parallel for private(id) shared(walkers)
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        this->walkers[id].resetPosition();
        this->walkers[id].resetSeed();
        this->walkers[id].resetEnergy();
    }

    // reset vector to store energy decay
    (*this).resetGlobalEnergy();
    this->globalEnergy.reserve(this->numberOfEchoes);

    // get initial energy state
    double energySum = 0.0;
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        energySum += this->walkers[id].energy;
    }
    this->globalEnergy.push_back(energySum);


    energySum = 0.0;
    uint id, step;
    for (uint echo = 0; echo < this->numberOfEchoes; echo++)
    {

        // walkers walk some steps with omp parallel
        // #pragma omp parallel for if(NMR_OPENMP) private(id, step) shared(walkers, bitBlock, simulationSteps)
        for (id = 0; id < numberOfWalkers; id++)
        {
            for (step = 0; step < this->stepsPerEcho; step++)
            {
                this->walkers[id].walk(this->bitBlock);
            }
        }

        // collect energy from all walkers with omp reduce
        energySum = 0.0; // reset energy summation
        // #pragma omp parallel for if(NMR_OPENMP) reduction(+:energySum) private(id) shared(walkers)
        for (id = 0; id < this->numberOfWalkers; id++)
        {
            energySum += this->walkers[id].energy;
        }

        //energySum = energySum / (double)numberOfWalkers;
        this->globalEnergy.push_back(energySum);
    }

    cout << "Completed.";
    double finish_time = omp_get_wtime();
    printElapsedTime(begin_time, finish_time);
}

void NMR_Simulation::fastSimulation()
{
    double begin_time = omp_get_wtime();
    cout << "initializing RW-NMR fast simulation... ";

    if(this->histogram.size == 0 || this->penalties == NULL) 
    {
        cout << "could not start simulation without histogram or penalties vector" << endl;
        return;
        // (*this).createHistogram();
        // (*this).createPenaltiesVector();
    }

    // initialize energyDistribution array
    double *energyDistribution = NULL;
    energyDistribution = new double[this->histogram.size];
    for(int idx = 0; idx < this->histogram.size; idx++)
    {
        energyDistribution[idx] = this->histogram.amps[idx];
    }

    // reset vector to store energy decay
    (*this).resetGlobalEnergy();
    this->globalEnergy.push_back(1.0);

    // fast simulation main loop
    for(uint echo = 0; echo < this->numberOfEchoes; echo++)
    {
        
        // apply penalties
        for(uint id = 0; id < this->histogram.size; id++)
        {
            energyDistribution[id] *= this->penalties[id];
        }

        // get global energy
        double energySum = 0.0;
        for(uint id = 0; id < this->histogram.size; id++)
        {
            energySum += energyDistribution[id];
        }

        // add to global energy vector
        this->globalEnergy.push_back(energySum);
    }

    // delete energy histogram distribution from memory
    delete[] energyDistribution;
    energyDistribution = NULL;

    cout << "Completed.";
    double finish_time = omp_get_wtime();
    printElapsedTime(begin_time, finish_time);
}

void NMR_Simulation::histSimulation()
{
    double begin_time = omp_get_wtime();
    cout << "initializing RW-NMR hist simulation... ";

    if(this->histogramList.size() == 0)  
    {
        cout << "could not start simulation without histogram list" << endl;
        return;
    }
    if(this->penalties == NULL)
    {
        cout << "could not start simulation without penalies vector" << endl;
        return;
    }

    // initialize energyDistribution array
    double *energyDistribution = NULL;
    energyDistribution = new double[this->histogram.size];


    // reset vector to store energy decay
    (*this).resetGlobalEnergy();
    this->globalEnergy.push_back(1.0);

    // histogram simulation main loop    
    for(int hst_ID = 0; hst_ID < this->histogramList.size(); hst_ID++)
    {
        for(uint id = 0; id < this->histogram.size; id++)
        {
            energyDistribution[id] = this->globalEnergy.back() * this->histogramList[hst_ID].amps[id];
        }

        double energyLvl;
        int eBegin = this->histogramList[hst_ID].firstEcho;
        int eEnd = this->histogramList[hst_ID].lastEcho;
        for(uint echo = eBegin; echo < eEnd; echo++)
        {
            // apply penalties
            for(uint id = 0; id < this->histogram.size; id++)
            {
                energyDistribution[id] *= this->penalties[id];
            }

            // get global energy
            energyLvl = 0.0;
            for(uint id = 0; id < this->histogram.size; id++)
            {
                energyLvl += energyDistribution[id];
            }

            // add to global energy vector
            this->globalEnergy.push_back(energyLvl);
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

void NMR_Simulation::createPenaltiesVector(vector<double> &_sigmoid)
{
    // initialize penalties array
    if(this->penalties == NULL)
        this->penalties = new double[this->histogram.size];
    
    Walker toy;
    double artificial_xirate;
    double artificial_steps = (double) this->stepsPerEcho;
    for(int idx = 0; idx < this->histogram.size; idx++)
    {   
        artificial_xirate = this->histogram.bins[idx];
        toy.setXIrate(artificial_xirate);
        toy.setSurfaceRelaxivity(_sigmoid);
        toy.computeDecreaseFactor(this->imageVoxelResolution, this->diffusionCoefficient);
        this->penalties[idx] = pow(toy.getDecreaseFactor(), (artificial_xirate * artificial_steps));

        // debug
        // cout << "delta[" << artificial_xirate << "] = " << penalties[idx] << endl;
    }
}

void NMR_Simulation::createPenaltiesVector(double rho)
{
    // initialize penalties array
    penalties = new double[this->histogram.size];
    Walker toy;
    double artificial_xirate;
    double artificial_steps = (double) this->stepsPerEcho;
    for(int idx = 0; idx < this->histogram.size; idx++)
    {   
        artificial_xirate = this->histogram.bins[idx];
        toy.setXIrate(artificial_xirate);
        toy.setSurfaceRelaxivity(rho);
        toy.computeDecreaseFactor(this->imageVoxelResolution, this->diffusionCoefficient);
        penalties[idx] = pow(toy.getDecreaseFactor(), (artificial_xirate * artificial_steps));

        // debug
        // cout << "delta[" << artificial_xirate << "] = " << penalties[idx] << endl;
    }
}


// diffusion simulations
double NMR_Simulation::diffusionSimulation_OMP(double gradientMagnitude, 
                                             double tinyDelta, 
                                             double giromagneticRatio)
{
    double begin_time = omp_get_wtime();

    cout << "initializing RW-PFGSE-NMR simulation... ";

    // reset walker's initial state with omp parallel for
// #pragma if(NMR_OPENMP) omp parallel for private(id) shared(walkers)
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        this->walkers[id].resetPosition();
        this->walkers[id].resetSeed();
        this->walkers[id].resetEnergy();
    }
    this->globalEnergy.clear();  // reset vector to store NMR decay

    // compute k value
    double K_value = compute_pfgse_k_value(gradientMagnitude, tinyDelta, giromagneticRatio);

    // set derivables 
    double globalPhase = 0.0;
    double globalSignal = 0.0;
    double walkerPhase;
    double walkerSignal;

    // main loop 
    for (uint id = 0; id < this->walkers.size(); id++)
    {  
        // make walkers walk througout image
        // #pragma omp parallel for if(NMR_OPENMP) private(id, step) shared(walkers, bitBlock, simulationSteps)
        for (uint step = 0; step < this->simulationSteps; step++)
        {
            this->walkers[id].walk(this->bitBlock);     
        }

        // get final individual signal
        walkerSignal = this->walkers[id].energy;

        // get final individual phase
        double z0 = (double) this->walkers[id].initialPosition.z;
        double zF = (double) this->walkers[id].position_z;
        double deltaZ = (zF - z0);
        double realMag = K_value * deltaZ * this->imageVoxelResolution;
        walkerPhase = walkerSignal * cos(realMag);

        // add contribution to global sum
        globalPhase += walkerPhase;
        globalSignal += walkerSignal;
    }

    double finish_time = omp_get_wtime();
    cout << "Completed."; printElapsedTime(begin_time, finish_time);
    return (globalPhase / globalSignal);
    
}

double NMR_Simulation::compute_pfgse_k_value(double gradientMagnitude, 
                                             double tinyDelta, 
                                             double giromagneticRatio)
{
    return (tinyDelta * 1.0e-03) * (TWO_PI * giromagneticRatio * 1.0e+06) * (gradientMagnitude * 1.0e-08);
}
