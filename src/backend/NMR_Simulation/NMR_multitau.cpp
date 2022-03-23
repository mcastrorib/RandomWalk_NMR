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
#include "NMR_multitau.h"
#include "NMR_cpmg.h"
#include "../Walker/walker.h"
#include "../FileHandler/fileHandler.h"

using namespace std;

NMR_multitau::NMR_multitau( NMR_Simulation &_NMR,  
                            multitau_config _multitauConfig,  
                            cpmg_config _cpmgConfig,
                            int _mpi_rank,
                            int _mpi_processes) : NMR(_NMR),
                                                  cpmg(NULL), 
                                                  MultiTau_config(_multitauConfig), 
                                                  CPMG_config(_cpmgConfig), 
                                                  mpi_rank(_mpi_rank), 
                                                  mpi_processes(_mpi_processes)
{
    // Initialize cpmg object
    this->cpmg = new NMR_cpmg(this->NMR, this->CPMG_config);

	// vectors object init
    vector<uint> requiredSteps();
    vector<double> signalTimes();
    vector<double> signalAmps();

    (*this).setName();
    (*this).createDirectoryForData();
    (*this).setTauSequence();
}


void NMR_multitau::setName()
{
    int precisionVal = 2;
    string tauMin = std::to_string(this->MultiTau_config.getTauMin());
    string tauMax = std::to_string(this->MultiTau_config.getTauMax());
    string points = std::to_string(this->MultiTau_config.getTauPoints());
    string scale = this->MultiTau_config.getTauScale();

    string trimmedTauMin = tauMin.substr(0, std::to_string(this->MultiTau_config.getTauMin()).find(".") + precisionVal + 1);
    string trimmedTauMax = tauMax.substr(0, std::to_string(this->MultiTau_config.getTauMax()).find(".") + precisionVal + 1);
	this->name = "/NMR_multitau_min=" + trimmedTauMin + "ms_max=" + trimmedTauMax + "ms_pts=" + points + "_scale=" + scale;
}

void NMR_multitau::createDirectoryForData()
{
	string path = this->NMR.getDBPath();
    createDirectory(path, this->NMR.simulationName + "/" + this->name);
    this->dir = (path + this->NMR.simulationName + "/" + this->name);
}

void NMR_multitau::setTauSequence()
{
    double tauMin = this->MultiTau_config.getTauMin();
    double tauMax = this->MultiTau_config.getTauMax();
    int tauPoints = this->MultiTau_config.getTauPoints();
    string scale = this->MultiTau_config.getTauScale();

    vector<double> times;
    if(scale == "log") times = (*this).logspace(log10(tauMin), log10(tauMax), tauPoints);
    else times = (*this).linspace(tauMin, tauMax, tauPoints);

    double timeInterval = this->NMR.getTimeInterval();
    if(this->requiredSteps.size() != 0) this->requiredSteps.clear();
    if(this->signalTimes.size() != 0) this->signalTimes.clear();
    uint minSteps = 0;
    for(uint idx = 0; idx < times.size(); idx++)
    {
        int steps = std::ceil(times[idx]/timeInterval);
        if(steps % 2 != 0) steps++;
        if(steps > minSteps)
        {
            requiredSteps.push_back(steps);
            minSteps = steps;
        } else
        {
            steps = minSteps + 2;
            requiredSteps.push_back(steps);
            minSteps = steps;
        }

        signalTimes.push_back(steps*timeInterval);
    }
}

void NMR_multitau::setExposureTime(uint index)
{
    this->NMR.setNumberOfStepsPerEcho(this->requiredSteps[index]);
    this->cpmg->setExposureTime(this->signalTimes[index]);
}

void NMR_multitau::setNMRTimeFramework()
{
    this->cpmg->setNMRTimeFramework(false);
}

void NMR_multitau::runCPMG()
{
    this->cpmg->run_simulation();
    int size = this->cpmg->signal_amps.size();
    for(int idx = 0; idx < size; idx++)
    {
        cout << "M[" << idx << "] = " << this->cpmg->signal_amps[idx] << endl;
    }
    cout << endl;
    if(size > 0)
    {
        this->signalAmps.push_back(this->cpmg->signal_amps[1]);
    }
}

void NMR_multitau::run()
{
    // before everything, reset conditions and map with highest time value
    double tick = omp_get_wtime();
    
    for(uint index = 0; index < this->requiredSteps.size(); index++)
    {
        (*this).setExposureTime(index);
        (*this).setNMRTimeFramework();
        (*this).runCPMG();
    }

    (*this).save();

    double time = omp_get_wtime() - tick;
    cout << endl << "multitau_time: " << time << " seconds." << endl;
}

// -- Savings
void NMR_multitau::save()
{
	double time = omp_get_wtime();
    cout << "saving results...";
       
    if(this->MultiTau_config.getSaveWalkers())
    {
        (*this).writeWalkers();
    }

    if(this->MultiTau_config.getSaveHistogram())
    {
        (*this).writeHistogram();
    }    

    if(this->MultiTau_config.getSaveHistogramList())
    {
        (*this).writeHistogramList();
    }

    // write multitau data
    if(this->MultiTau_config.getSaveDecay()) 
    {
        (*this).writeDecay();
    }

	time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}

void NMR_multitau::writeDecay()
{
    string filename = this->dir + "/multitau_decay.csv";

    ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    const int num_points = this->signalAmps.size();
    const int precision = std::numeric_limits<double>::max_digits10;

    file << "echo_time,signal" << endl;
    if(this->signalTimes.size() == this->signalAmps.size())
    {
        for (int idx = 0; idx < num_points; idx++)
        {
            file << setprecision(precision) << this->signalTimes[idx] << ",";
            file << setprecision(precision) << this->signalAmps[idx] << endl;    
        }
    }    
    file.close();
}

void NMR_multitau::writeWalkers()
{
    string filename = this->dir + "/multitau_walkers.csv";
    ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    file << "PositionXi";
    file << ",PositionYi";
    file << ",PositionZi";
    file << ",PositionXf";
    file << ",PositionYf";
    file << ",PositionZf";
    file << ",Collisions";
    file << ",XIRate";
    file << ",Energy"; 
    file << ",RNGSeed" << endl;

    const int precision = 6;
    for (uint index = 0; index < this->NMR.walkers.size(); index++)
    {
        file << setprecision(precision) << this->NMR.walkers[index].getInitialPositionX()
        << "," << this->NMR.walkers[index].getInitialPositionY()
        << "," << this->NMR.walkers[index].getInitialPositionZ()
        << "," << this->NMR.walkers[index].getPositionX() 
        << "," << this->NMR.walkers[index].getPositionY() 
        << "," << this->NMR.walkers[index].getPositionZ() 
        << "," << this->NMR.walkers[index].getCollisions() 
        << "," << this->NMR.walkers[index].getXIrate() 
        << "," << this->NMR.walkers[index].getEnergy() 
        << "," << this->NMR.walkers[index].getInitialSeed() << endl;
    }

    file.close();
}

void NMR_multitau::writeHistogram()
{
    string filename = this->dir + "/multitau_histogram.csv";
    ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    file << "Bins"; 
    file << ",Amps" << endl;
    const int num_points = this->NMR.histogram.getSize();
    const int precision = std::numeric_limits<double>::max_digits10;
    for (int i = 0; i < num_points; i++)
    {
        file << setprecision(precision) 
        << this->NMR.histogram.bins[i] 
        << "," << this->NMR.histogram.amps[i] << endl;
    }

    file.close();
}

void NMR_multitau::writeHistogramList()
{
    string filename = this->dir + "/multitau_histList.csv";
    ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    const int histograms = this->NMR.histogramList.size();

    for(int hIdx = 0; hIdx < histograms; hIdx++)
    {
        file << "Bins" << hIdx << ",";
        file << "Amps" << hIdx << ",";
    }
    file << endl;

    const int num_points = this->NMR.histogram.getSize();
    const int precision = std::numeric_limits<double>::max_digits10;
    for (int i = 0; i < num_points; i++)
    {
        for(int hIdx = 0; hIdx < histograms; hIdx++)
        {
            file << setprecision(precision) << this->NMR.histogramList[hIdx].bins[i] << ",";
            file << setprecision(precision) << this->NMR.histogramList[hIdx].amps[i] << ",";
        }

        file << endl;
    }

    file.close();
}

// Returns a vector<double> linearly space from @start to @end with @points
vector<double> NMR_multitau::linspace(double start, double end, uint points)
{
    vector<double> vec(points);
    double step = (end - start) / ((double) points - 1.0);
    
    for(int idx = 0; idx < points; idx++)
    {
        double x_i = start + step * idx;
        vec[idx] = x_i;
    }

    return vec;
}

// Returns a vector<double> logarithmly space from 10^@exp_start to 10^@end with @points
vector<double> NMR_multitau::logspace(double exp_start, double exp_end, uint points, double base)
{
    vector<double> vec(points);
    double step = (exp_end - exp_start) / ((double) points - 1.0);
    
    for(int idx = 0; idx < points; idx++)
    {
        double x_i = exp_start + step * idx;
        vec[idx] = pow(base, x_i);
    }

    return vec;
}

double NMR_multitau::sum(vector<double> _vec)
{
    double sum = 0.0;
    for(uint idx = 0; idx < _vec.size(); idx++)
    {
        sum += _vec[idx];
    }
    return sum;
}

int NMR_multitau::sum(vector<int> _vec)
{
    int sum = 0;
    for(uint idx = 0; idx < _vec.size(); idx++)
    {
        sum += _vec[idx];
    }
    return sum;
}

uint NMR_multitau::sum(vector<uint> _vec)
{
    uint sum = 0;
    for(uint idx = 0; idx < _vec.size(); idx++)
    {
        sum += _vec[idx];
    }
    return sum;
}