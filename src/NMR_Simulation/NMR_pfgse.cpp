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
				     double _GF,
				     int _GPoints,
				  	 double _bigDelta,
				  	 double _pulseWidth, 
				  	 double _giromagneticRatio, 
				  	 int _mpi_rank, 
				  	 int _mpi_processes) : NMR(_NMR),
										   gradient_max(_GF),
										   gradientPoints(_GPoints),
										   exposureTime(_bigDelta),
										   pulseWidth(_pulseWidth),
										   giromagneticRatio(_giromagneticRatio),
										   diffusionCoefficient(0.0),
										   M0(0.0),
										   mpi_rank(_mpi_rank),
										   mpi_processes(_mpi_processes)
{
	vector<double> gradient();
	vector<double> LHS();
	vector<double> RHS();
	vector<Vector3D> vecGradient();
	(*this).setThresholdFromRHSValue(numeric_limits<double>::max());
	(*this).set_old();
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
										   diffusionCoefficient(0.0),
										   M0(0.0),
										   mpi_rank(_mpi_rank),
										   mpi_processes(_mpi_processes)
{
	vector<double> gradient();
	vector<double> LHS();
	vector<double> RHS();
	vector<Vector3D> vecGradient();
	(*this).setThresholdFromRHSValue(numeric_limits<double>::max());
	(*this).set();
}

void NMR_PFGSE::set()
{
	(*this).setName();
	(*this).createDirectoryForData();
	(*this).setNMRTimeFramework();
	(*this).setGradientVector();
	(*this).setVectorLHS();
	(*this).setVectorRHS();
}

void NMR_PFGSE::set_old()
{
	(*this).setNMRTimeFramework();
	(*this).setGradientVector_old();
	(*this).setVectorLHS();
	(*this).setVectorRHS();
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
	string path = DATA_PATH;
    createDirectory(path, this->NMR.simulationName + "/" + this->name);
    this->dir = (path + this->NMR.simulationName + "/" + this->name);
}

void NMR_PFGSE::setGradientVector(double _GF, int _GPoints)
{
	this->gradient_max = _GF;
	this->gradientPoints = _GPoints;
	(*this).setGradientVector();
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


void NMR_PFGSE::setGradientVector_old()
{
	if(this->gradient.size() > 0) this->gradient.clear();
	this->gradient.reserve(this->gradientPoints);
	
	double gap = (this->gradient_max) / (this->gradientPoints - 1);
	double gvalue = 0.0;
	for(uint index = 0; index < this->gradientPoints; index++)
	{
		gradient.push_back(gvalue);
		gvalue += gap;
	}
}


void NMR_PFGSE::setNMRTimeFramework()
{
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
	if(PFGSE_USE_TWOPI) gamma *= TWO_PI;
	
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


void NMR_PFGSE::run()
{
	// run pfgse experiment
	(*this).simulation();

	// get M0 (reference value)
	int idx_begin = 0;
	int idx_end = this->gradientPoints;
	if(this->vecGradient[idx_begin].getNorm() == 0.0) 
	{	
		this->M0 = this->LHS[0];
		this->LHS[0] = (*this).computeLHS(M0, M0);
		idx_begin++;
	}	
	else 
	{
		// it is necessary to run simulation for g = 0
		this->M0 = this->NMR.PFG(0.0, this->pulseWidth, this->giromagneticRatio);
	}

	// run diffusion measurement for different G
	for(uint point = idx_begin; point < idx_end; point++)
	{
		this->LHS[point] = (*this).computeLHS(this->LHS[point], M0);
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


void NMR_PFGSE::run_old()
{
	// run pfgse experiment
	(*this).simulation_old();

	// get M0 (reference value)
	int idx_begin = 0;
	int idx_end = this->gradientPoints;
	if(this->gradient[idx_begin] == 0.0) 
	{	
		this->M0 = this->LHS[0];
		this->LHS[0] = (*this).computeLHS(M0, M0);
		idx_begin++;
	}	
	else 
	{
		// it is necessary to run simulation for g = 0
		this->M0 = this->NMR.PFG(0.0, this->pulseWidth, this->giromagneticRatio);
	}

	// run diffusion measurement for different G
	for(uint point = idx_begin; point < idx_end; point++)
	{
		this->LHS[point] = (*this).computeLHS(this->LHS[point], M0);
	}
}

void NMR_PFGSE::simulation_old()
{
	if(this->NMR.gpu_use == true)
	{
		(*this).simulation_cuda();
	}
	else
	{
		(*this).simulation_omp();
	}
}


void NMR_PFGSE::recover_D(string _method)
{
	if(_method == "stejskal")
	{
		(*this).recover_Stejskal();
	} else
	{
		if(_method == "msd")
		{
			(*this).recover_meanSquaredDisplacement();
		}
	}
}

void NMR_PFGSE::recover_Stejskal()
{
	LeastSquareAdjust lsa(this->RHS, this->LHS);
	lsa.setThreshold(this->RHS_threshold);
	lsa.solve();
	(*this).setDiffusionCoefficient(lsa.getB());
	cout << "Dnew (s&t) = " << (*this).getDiffusionCoefficient() << endl;
}

void NMR_PFGSE::recover_meanSquaredDisplacement()
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

	// debug
	// cout << "(msd): " << squaredDisplacement;
	// cout << "\twalkers alive: " << aliveWalkerFraction;
	// cout << "\tobs time: " << this->exposureTime << " ms" << endl;

	// set diffusion coefficient (see eq 2.18 - ref. Bergman 1995)
	squaredDisplacement = squaredDisplacement / aliveWalkerFraction;
	(*this).setDiffusionCoefficient(squaredDisplacement/(6 * this->exposureTime));
	
	cout << "Dnew (msd) = " << (*this).getDiffusionCoefficient();
	cout << "\t(mean displacement): " << sqrt(squaredDisplacement) << " um" << endl;
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
	this->NMR.save(this->dir);
	(*this).writeResults();
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
    file << "Diffusion Coefficient, ";
    file << "RHS Threshold" << endl;
    file << this->gradientPoints << ", ";
    file << this->exposureTime << ", ";
    file << this->pulseWidth << ", ";
    file << this->giromagneticRatio << ", ";
    file << this->diffusionCoefficient << ", ";
    file << threshold << endl << endl;    

    file << "Stejskal-Tanner Equation" << endl;
    file << "ID, ";
    file << "Gradient, ";
    file << "LHS, ";
    file << "RHS, " << endl;

    uint size = this->gradientPoints;
    for (uint index = 0; index < size; index++)
    {
        file << index << ", ";
        file << this->gradient[index] << ", ";
        file << this->LHS[index] << ", ";
        file << this->RHS[index] << endl;
    }

    file.close();
}

void NMR_PFGSE::simulation_omp()
{
	cout << "omp pfgse simulation not implemented yet. try again later :)" << endl;
}