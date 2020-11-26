// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

// include OpenMP for multicore implementation
#include <omp.h>

//include
#include "NMR_defs.h"
#include "NMR_Simulation.h"
#include "NMR_pfgse.h"
#include "../Math/LeastSquareAdjust.h"
#include "../Walker/walker.h"
#include "../Math/Vector3D.h"
#include "../FileHandler/fileHandler.h"

using namespace cv;
using namespace std;

NMR_PFGSE::NMR_PFGSE(NMR_Simulation &_NMR,  
				     pfgse_config _pfgseConfig,
					 int _mpi_rank,
					 int _mpi_processes) : NMR(_NMR),
										   PFGSE_config(_pfgseConfig),
										   M0(0.0),
										   D_sat(0.0),
										   D_msd(0.0),
										   SVp(0.0),
										   mpi_rank(_mpi_rank),
										   mpi_processes(_mpi_processes)
{
	// vectors object init
	vector<double> exposureTimes();
	vector<double> gradient();
	vector<double> LHS();
	vector<double> RHS();
	vector<double> Mkt();
	vector<Vector3D> vecGradient();

	// read config file
	Vector3D gradient_max = this->PFGSE_config.getMaxGradient();
	this->gradient_X = gradient_max.getX();
	this->gradient_Y = gradient_max.getY();
	this->gradient_Z = gradient_max.getZ();	
	this->gradientPoints = this->PFGSE_config.getGradientSamples();
	this->exposureTimes = this->PFGSE_config.getTimeValues(); 
	this->pulseWidth = this->PFGSE_config.getPulseWidth();
	this->giromagneticRatio = this->PFGSE_config.getGiromagneticRatio();
	// this->NMR.setFreeDiffusionCoefficientthis(this->PFGSE_config.getD0());


	(*this).setThresholdFromRHSValue(numeric_limits<double>::max());
	(*this).setGradientVector();
	(*this).setVectorMkt();

}

NMR_PFGSE::NMR_PFGSE(NMR_Simulation &_NMR,  
				     Vector3D _gradient_max,
				     int _GPoints,
				  	 double _bigDelta,
				  	 double _pulseWidth, 
				  	 double _giromagneticRatio, 
				  	 int _mpi_rank, 
				  	 int _mpi_processes) : NMR(_NMR),
										   gradient_X(_gradient_max.x),
										   gradient_Y(_gradient_max.y),
										   gradient_Z(_gradient_max.z),
										   gradientPoints(_GPoints),
										   exposureTime(_bigDelta),
										   pulseWidth(_pulseWidth),
										   giromagneticRatio(_giromagneticRatio),
										   D_sat(0.0),
										   D_msd(0.0),
										   SVp(0.0),
										   M0(0.0),
										   mpi_rank(_mpi_rank),
										   mpi_processes(_mpi_processes)
{
	vector<double> gradient();
	vector<double> LHS();
	vector<double> RHS();
	vector<double> Mkt();
	vector<Vector3D> vecGradient();
	(*this).setThresholdFromRHSValue(numeric_limits<double>::max());
	(*this).setGradientVector();
	(*this).setVectorMkt();
	(*this).set();
}

void NMR_PFGSE::set()
{
	(*this).setName();
	(*this).createDirectoryForData();
	(*this).setNMRTimeFramework();
	(*this).setVectorLHS();
	(*this).setVectorRHS();
}

void NMR_PFGSE::run()
{
	for(uint timeSample = 0; timeSample < this->exposureTimes.size(); timeSample++)
	{
		(*this).setExposureTime((*this).getExposureTime(timeSample));
		(*this).set();
		(*this).run_sequence();

		// apply threshold for D(t) extraction
		string threshold_type = this->PFGSE_config.getThresholdType();
		double threshold = this->PFGSE_config.getThresholdValue();
		if(threshold_type != "none")
		{
			if(threshold_type == "lhs") (*this).setThresholdFromLHSValue(threshold);
			else if(threshold_type == "rhs") (*this).setThresholdFromRHSValue(threshold);
			else if(threshold_type == "samples") (*this).setThresholdFromSamples(int(threshold));
		}

		// D(t) extraction
		(*this).recoverD("sat");
		(*this).recoverD("msd");
		(*this).recoverSVp();
		
		// save results in disc
		(*this).save(); 
	}
}


void NMR_PFGSE::setName()
{
	string big_delta = std::to_string((int) this->exposureTime) + "-" 
					   + std::to_string(((int) (this->exposureTime * 100)) % 100);
	string tiny_delta = std::to_string((int) this->pulseWidth);
	string gamma = std::to_string((int) this->giromagneticRatio);
	this->name = "/NMR_pfgse_" + big_delta + "ms_" + tiny_delta + "ms_" + gamma + "_sT";
}

void NMR_PFGSE::createDirectoryForData()
{
	string path = this->NMR.rwNMR_config.getDBPath();
    createDirectory(path, this->NMR.simulationName + "/" + this->name);
    this->dir = (path + this->NMR.simulationName + "/" + this->name);
}

void NMR_PFGSE::setGradientVector(double _GF, int _GPoints)
{
	this->gradient_max = _GF;
	this->gradientPoints = _GPoints;
	(*this).setGradientVector();
}

void NMR_PFGSE::setVectorMkt()
{
	if(this->Mkt.size() > 0) this->Mkt.clear();
	this->Mkt.reserve(this->gradientPoints);
}

void NMR_PFGSE::setGradientVector()
{
	if(this->vecGradient.size() > 0) this->vecGradient.clear();
	this->vecGradient.reserve(this->gradientPoints);

	if(this->gradient.size() > 0) this->gradient.clear();
	this->gradient.reserve(this->gradientPoints);
	
	double gapX = (this->gradient_X) / ((double) (this->gradientPoints - 1));
	double gapY = (this->gradient_Y) / ((double) (this->gradientPoints - 1));
	double gapZ = (this->gradient_Z) / ((double) (this->gradientPoints - 1));
	
	double gvalueX = 0.0;
	double gvalueY = 0.0;
	double gvalueZ = 0.0;

	for(uint index = 0; index < this->gradientPoints; index++)
	{
		Vector3D newGradient(gvalueX, gvalueY, gvalueZ);
		vecGradient.push_back(newGradient);
		gradient.push_back(newGradient.getNorm());
		gvalueX += gapX;
		gvalueY += gapY;
		gvalueZ += gapZ;
	}
}

void NMR_PFGSE::setNMRTimeFramework()
{
	cout << endl << "running PFGSE simulation:" << endl;
	this->NMR.setTimeFramework(this->exposureTime);
	cout << "PFGSE exposure time: " << this->exposureTime << " ms";
	cout << " (" << this->NMR.simulationSteps << " RW-steps)" << endl;
	this->NMR.mapSimulation();
	// this->NMR.updateRelaxativity(rho); but what rho to adopt?
}

void NMR_PFGSE::setVectorRHS()
{
	if(this->RHS.size() > 0) this->RHS.clear();
	this->RHS.reserve(this->gradientPoints);

	for(uint idx = 0; idx < this->gradientPoints; idx++)
	{
		double rhs = (*this).computeRHS(this->gradient[idx]);
		this->RHS.push_back(rhs);
	}
}

void NMR_PFGSE::setThresholdFromRHSValue(double _value)
{
	if(this->RHS.size() > 0 && _value > fabs(this->RHS.back()))
		_value = fabs(this->RHS.back());

	this->RHS_threshold = _value;			
}

void NMR_PFGSE::setThresholdFromLHSValue(double _value)
{
	if(this->LHS.size() == 0) 
		return;

	if(_value > 0.0 && _value < 1.0)
	{
		int idx = 0;
		bool isGreater = true;
		double logValue = log(_value);

		while(idx < this->LHS.size() && isGreater == true)
		{
			if(this->LHS[idx] < logValue)
			{
				isGreater = false;
			}
			else
			{
				idx++;
			}
		}

		if(isGreater) idx--;
		this->RHS_threshold = fabs(this->RHS[idx]);
	}
}

void NMR_PFGSE::setThresholdFromSamples(int _samples)
{
	if(_samples < this->RHS.size() && _samples > 1)	
		this->RHS_threshold = fabs(this->RHS[_samples]);
}

void NMR_PFGSE::setThresholdFromFraction(double _fraction)
{
	if(_fraction > 0.0 && _fraction < 1.0)
	{
		int samples = (int) (_fraction * (double) this->gradientPoints);
		if(samples < 2) samples = 2;	
		this->RHS_threshold = fabs(this->RHS[samples]);

	}
}

double NMR_PFGSE::computeRHS(double _Gvalue)
{
	double gamma = this->giromagneticRatio;
	if(this->PFGSE_config.getUseWaveVectorTwoPi()) gamma *= TWO_PI;
	
	return (-1.0e-10) * (gamma * this->pulseWidth) * (gamma * this->pulseWidth) 
			* (this->exposureTime - ((this->pulseWidth) / 3.0)) 
			* _Gvalue *  _Gvalue ;  
}

void NMR_PFGSE::setVectorLHS()
{
	if(this->LHS.size() > 0) this->LHS.clear();
	this->LHS.reserve(this->gradientPoints);
}

double NMR_PFGSE::computeLHS(double _Mg, double _M0)
{
	return log(fabs(_Mg/_M0));
}

double NMR_PFGSE::computeWaveVectorK(double gradientMagnitude, double pulse_width, double giromagneticRatio)
{
    return (pulse_width * 1.0e-03) * (TWO_PI * giromagneticRatio * 1.0e+06) * (gradientMagnitude * 1.0e-08);
}


void NMR_PFGSE::run_sequence()
{
	// run pfgse experiment -- this method will fill Mkt vector
	(*this).simulation();

	// get M0 (reference value)
	int idx_begin = 0;
	int idx_end = this->gradientPoints;
	
	// ------------ Deprecated ---------------------------
	// if(this->vecGradient[idx_begin].getNorm() == 0.0) 
	// {	
	// 	this->M0 = this->LHS[0];
	// 	this->LHS[0] = (*this).computeLHS(M0, M0);
	// 	idx_begin++;
	// }	
	// else 
	// {
	// 	// it is necessary to run simulation for g = 0
	// 	this->M0 = this->NMR.PFG(0.0, this->pulseWidth, this->giromagneticRatio);
	// }

	// copy vector LHS to Mkt - old
	// for(uint idx = 0; idx < this->gradientPoints; idx++)
	// {
	// 	this->Mkt.push_back(this->LHS[idx]);
	// }

	this->M0 = this->Mkt[0];
	this->LHS[0] = (*this).computeLHS(M0, M0);
	idx_begin++;

	// run diffusion measurement for different G - old
	// for(uint point = idx_begin; point < idx_end; point++)
	// {
	// 	this->LHS[point] = (*this).computeLHS(this->LHS[point], M0);
	// }

	for(uint point = idx_begin; point < idx_end; point++)
	{
		this->LHS[point] = (*this).computeLHS(this->Mkt[point], M0);
	}
}


void NMR_PFGSE::simulation()
{
	if(this->NMR.gpu_use == true)
	{
		(*this).simulation_cuda();
		this->NMR.normalizeEnergyDecay();
	}
	else
	{
		(*this).simulation_omp();
	}
}

void NMR_PFGSE::recoverD(string _method)
{
	if(_method == "sat")
	{
		(*this).recoverD_sat();
	} else
	{
		if(_method == "msd")
		{
			(*this).recoverD_msd();
		}
	}
}

void NMR_PFGSE::recoverD_sat()
{
	LeastSquareAdjust lsa(this->RHS, this->LHS);
	lsa.setThreshold(this->RHS_threshold);
	lsa.solve();
	(*this).setD_sat(lsa.getB());
	cout << "Dnew (s&t) = " << (*this).getD_sat() << endl;
}

void NMR_PFGSE::recoverD_msd()
{
	double squaredDisplacement = 0.0;
	double displacementX, displacementY, displacementZ;
	double X0, Y0, Z0;
	double XF, YF, ZF;
	double normalizedDisplacement;
	double resolution = this->NMR.getImageVoxelResolution();
	double aliveWalkerFraction = 0.0;

	for(uint idx = 0; idx < this->NMR.numberOfWalkers; idx++)
	{
		Walker particle(this->NMR.walkers[idx]);

		// Get walker displacement
		// X:
		X0 = (double) particle.initialPosition.x;
		XF = (double) particle.position_x;
		displacementX = resolution * (XF - X0);
		
		// Y:
		Y0 = (double) particle.initialPosition.y;
		YF = (double) particle.position_y;
		displacementY = resolution * (YF - Y0);
		
		// Z:
		Z0 = (double) particle.initialPosition.z;
		ZF = (double) particle.position_z;
		displacementZ = resolution * (ZF - Z0);
		// displacementZ = 0.0;
		
		normalizedDisplacement = sqrt( (displacementX*displacementX + 
									    displacementY*displacementY + 
									    displacementZ*displacementZ));

		// get particle contribution / energy
		particle.resetEnergy();
		for(uint bump = 0; bump < particle.collisions; bump++)
		{
			particle.energy *= particle.decreaseFactor;
		}

		squaredDisplacement += ( particle.energy * (normalizedDisplacement * normalizedDisplacement));
		aliveWalkerFraction += particle.energy;
	}

	// set diffusion coefficient (see eq 2.18 - ref. Bergman 1995)
	squaredDisplacement = squaredDisplacement / aliveWalkerFraction;
	(*this).setD_msd(squaredDisplacement/(6 * this->exposureTime));
	
	cout << "Dnew (msd) = " << (*this).getD_msd();
	cout << "\t(mean displacement): " << sqrt(squaredDisplacement) << " um" << endl;
}

void NMR_PFGSE::recoverSVp(string method)
{
	double Dt;
	if(method == "sat") Dt = (*this).getD_sat();
	else Dt = (*this).getD_msd();

	double D0 = this->NMR.getDiffusionCoefficient();

	// recover S/V for short observation times (see ref. Sorland)
	double Sv;
	Sv = (1.0 - (Dt / D0));
	Sv *= 2.25 * sqrt((0.5*TWO_PI) / (D0 * (*this).getExposureTime()));
	(*this).setSVp(Sv);
	cout << "S/V ~= " << (*this).getSVp() << endl;
}

void NMR_PFGSE::reset(double newBigDelta)
{
	(*this).clear();
	(*this).setExposureTime(newBigDelta);
	(*this).set();
	(*this).setThresholdFromRHSValue(numeric_limits<double>::max());
}

void NMR_PFGSE::reset()
{
	(*this).clear();
	(*this).set();
	(*this).setThresholdFromRHSValue(numeric_limits<double>::max());
}



void NMR_PFGSE::clear()
{
	if(this->gradient.size() > 0) this->gradient.clear();
	if(this->LHS.size() > 0) this->LHS.clear();
	if(this->RHS.size() > 0) this->RHS.clear();
}

void NMR_PFGSE::save()
{
	double time = omp_get_wtime();
    cout << "saving results...";
    
    if(this->PFGSE_config.getSaveDecay()) 
    {
        this->NMR.saveEnergyDecay(this->dir);
	}
    
    if(this->PFGSE_config.getSaveCollisions())
    {
        this->NMR.saveWalkerCollisions(this->dir);
    }

    if(this->PFGSE_config.getSaveHistogram())
    {
        this->NMR.saveHistogram(this->dir);
    }    

    if(this->PFGSE_config.getSaveHistogramList())
    {
        this->NMR.saveHistogramList(this->dir);
    }

    // write pfgse data
	(*this).writeResults();

	time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}

void NMR_PFGSE::writeResults()
{
	// string big_delta = std::to_string((int) this->exposureTime) + "-" 
	// 				   + std::to_string(((int) (this->exposureTime * 100)) % 100);
	// string tiny_delta = std::to_string((int) this->pulseWidth);
	// string gamma = std::to_string((int) this->giromagneticRatio);
	string filename = this->dir + "/PFGSE_echoes.txt";

	ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    double threshold = fabs(this->RHS.back());
    if(this->RHS_threshold < threshold)
    	threshold = this->RHS_threshold;

	file << "RWNMR-PFGSE Results" << endl; 
	file << "Points, ";
    file << "Big Delta, ";
    file << "Tiny Delta, ";
    file << "Giromagnetic Ratio, ";
    file << "D_sat, ";
    file << "D_msd, ";
	file << "SVp, ";
    file << "RHS Threshold" << endl;
    file << this->gradientPoints << ", ";
    file << this->exposureTime << ", ";
    file << this->pulseWidth << ", ";
    file << this->giromagneticRatio << ", ";
    file << this->D_sat << ", ";
    file << this->D_msd << ", ";
    file << this->SVp << ", ";
    file << threshold << endl << endl;    

    file << "Stejskal-Tanner Equation" << endl;
    file << "id, ";
    file << "Gradient, ";
    file << "M(k,t), ";
    file << "LHS, ";
    file << "RHS" << endl;

    uint size = this->gradientPoints;
    for (uint index = 0; index < size; index++)
    {
        file << index << ", ";
        file << this->gradient[index] << ", ";
        file << this->Mkt[idx] << ", ";
        file << this->LHS[index] << ", ";
        file << this->RHS[index] << endl;
    }

    file.close();
}

// pfgse simulation cpu-only implementation -- needs revision! 
void NMR_PFGSE::simulation_omp()
{
	double begin_time = omp_get_wtime();

    cout << "initializing RW-PFGSE-NMR simulation... ";

    // reset walker's initial state with omp parallel for
// #pragma if(NMR_OPENMP) omp parallel for private(id) shared(walkers)
    for (uint id = 0; id < this->NMR.walkers.size(); id++)
    {
        this->NMR.walkers[id].resetPosition();
        this->NMR.walkers[id].resetSeed();
        this->NMR.walkers[id].resetEnergy();
    }
    this->NMR.globalEnergy.clear();  // reset vector to store NMR decay

    // compute k value
    double K_value = 0.0;

    // set derivables 
    double globalPhase = 0.0;
    double globalSignal = 0.0;
    double walkerPhase;
    double walkerSignal;

    // main loop 
    for (uint id = 0; id < this->NMR.walkers.size(); id++)
    {  
        // make walkers walk througout image
        // #pragma omp parallel for if(NMR_OPENMP) private(id, step) shared(walkers, bitBlock, simulationSteps)
        for (uint step = 0; step < this->NMR.simulationSteps; step++)
        {
            this->NMR.walkers[id].walk(this->NMR.bitBlock);     
        }

        // get final individual signal
        walkerSignal = this->NMR.walkers[id].energy;

        // get final individual phase
        double z0 = (double) this->NMR.walkers[id].initialPosition.z;
        double zF = (double) this->NMR.walkers[id].position_z;
        double deltaZ = (zF - z0);
        double realMag = K_value * deltaZ * this->NMR.imageVoxelResolution;
        walkerPhase = walkerSignal * cos(realMag);

        // add contribution to global sum
        globalPhase += walkerPhase;
        globalSignal += walkerSignal;
    }

    double finish_time = omp_get_wtime();
    cout << "Completed."; printElapsedTime(begin_time, finish_time);
    return;
}