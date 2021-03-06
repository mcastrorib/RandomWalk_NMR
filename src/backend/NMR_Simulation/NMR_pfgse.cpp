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
#include "../Utils/OMPLoopEnabler.h"
#include "../Utils/myAllocator.h"

using namespace cv;
using namespace std;

NMR_PFGSE::NMR_PFGSE(NMR_Simulation &_NMR,  
				     pfgse_config _pfgseConfig,
					 int _mpi_rank,
					 int _mpi_processes) : NMR(_NMR),
										   PFGSE_config(_pfgseConfig),
										   M0(0.0),
										   D_sat(0.0),
										   D_sat_stdev(0.0),
										   D_msd(0.0),
										   D_msd_stdev(0.0),
										   msd(0.0),
										   msd_stdev(0.0),
										   vecMsd(0.0, 0.0, 0.0),
										   vecMsd_stdev(0.0, 0.0, 0.0),
										   vecDmsd(0.0, 0.0, 0.0),
										   vecDmsd_stdev(0.0, 0.0, 0.0),
										   stepsTaken(0),
										   currentTime(0),
										   DsatAdjustSamples(0),
										   mpi_rank(_mpi_rank),
										   mpi_processes(_mpi_processes)
{
	// vectors object init
	vector<double> exposureTimes();
	vector<double> gradient();
	vector<double> RHS();
	vector<double> Mkt();
	vector<double> Mkt_stdev();
	vector<double> LHS();
	vector<double> LHS_stdev();
	vector<Vector3D> vecGradient();
	vector<Vector3D> vecK();

	// read config file
	Vector3D gradient_max = this->PFGSE_config.getMaxGradient();
	this->gradient_X = gradient_max.getX();
	this->gradient_Y = gradient_max.getY();
	this->gradient_Z = gradient_max.getZ();	
	this->gradientPoints = this->PFGSE_config.getGradientSamples();
	this->exposureTimes = this->PFGSE_config.getTimeValues(); 
	this->pulseWidth = this->PFGSE_config.getPulseWidth();
	this->giromagneticRatio = this->PFGSE_config.getGiromagneticRatio();
	if(this->PFGSE_config.getUseWaveVectorTwoPi()) this->giromagneticRatio *= TWO_PI;

	(*this).setThresholdFromSamples(this->gradientPoints);
	(*this).setGradientVector();
	(*this).setVectorK();
}

void NMR_PFGSE::set()
{
	(*this).setName();
	(*this).createDirectoryForData();
	(*this).setNMRTimeFramework();
	(*this).setVectorMkt();
	(*this).setVectorLHS();
	(*this).setVectorRHS();
}

void NMR_PFGSE::run()
{
	// before everything, reset conditions and map with highest time value
	double tick = omp_get_wtime();
	cout << endl << "-- Pre-processing:" << endl;
	(*this).resetCurrentTime();
	(*this).correctExposureTimes();
	(*this).runInitialMapSimulation();
	(*this).resetNMR();
	cout << "-- Done in " << omp_get_wtime() - tick << " seconds." << endl;

	double interiorTick;
	for(uint timeSample = 0; timeSample < this->exposureTimes.size(); timeSample++)
	{
		interiorTick = omp_get_wtime();
		(*this).setExposureTime((*this).getExposureTime(timeSample));
		(*this).set();
		(*this).runSequence();

		// D(t) extraction
		cout << "-- Results:" << endl;
		(*this).recoverDsat();
		(*this).recoverDmsd();
		
		// save results in disc
		(*this).save(); 
		(*this).incrementCurrentTime();
		cout << "-- Done in " << omp_get_wtime() - interiorTick << " seconds." << endl << endl;
	}

	double time = omp_get_wtime() - tick;
	cout << endl << "pfgse_time: " << time << " seconds." << endl;
}

void NMR_PFGSE::applyThreshold()
{
	// apply threshold for D(t) extraction
	string threshold_type = this->PFGSE_config.getThresholdType();
	double threshold = this->PFGSE_config.getThresholdValue();
	if(threshold_type == "lhs") (*this).setThresholdFromLHSValue(threshold);
	else if(threshold_type == "samples") (*this).setThresholdFromSamples(int(threshold));
	else (*this).setThresholdFromSamples(this->gradientPoints);
}

void NMR_PFGSE::resetNMR()
{
	// reset walker's initial state with omp parallel for
	cout << "- Reseting walker initial state" << endl;

    if(this->NMR.rwNMR_config.getOpenMPUsage())
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->NMR.walkers.size();
        int loop_start, loop_finish;

        #pragma omp parallel private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish(); 

            for (uint id = loop_start; id < loop_finish; id++)
            {
                this->NMR.walkers[id].resetPosition();
                this->NMR.walkers[id].resetSeed();
                this->NMR.walkers[id].resetEnergy();
                this->NMR.walkers[id].resetCollisions();
            }
        }
    } else
    {
        for (uint id = 0; id < this->NMR.walkers.size(); id++)
        {
            this->NMR.walkers[id].resetPosition();
            this->NMR.walkers[id].resetSeed();
            this->NMR.walkers[id].resetEnergy();
            this->NMR.walkers[id].resetCollisions();
        }
    }   
}

void NMR_PFGSE::updateWalkersXIrate(uint _rwsteps)
{
	// update walker's xirate with omp parallel for

    if(this->NMR.rwNMR_config.getOpenMPUsage())
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->NMR.walkers.size();
        int loop_start, loop_finish;

        #pragma omp parallel private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish(); 

            for (uint id = loop_start; id < loop_finish; id++)
            {
                this->NMR.walkers[id].updateXIrate(_rwsteps);
            }
        }
    } else
    {
        for (uint id = 0; id < this->NMR.walkers.size(); id++)
        {
            this->NMR.walkers[id].updateXIrate(_rwsteps);
        }
    }   
}

void NMR_PFGSE::correctExposureTimes()
{
	
	cout << "- Correcting time samples to rw parameters" << endl;
	double timePerStep = this->NMR.getTimeInterval();
	double stepsPerEcho = (double) this->NMR.getStepsPerEcho();
	uint stepsPerExpTime;
	for(int time = 0; time < this->exposureTimes.size(); time++)
	{
		stepsPerExpTime = this->exposureTimes[time] / timePerStep;
		if(stepsPerExpTime < 1) stepsPerExpTime = 1;
		if(stepsPerExpTime % (uint) stepsPerEcho != 0) 
		{
			stepsPerExpTime += stepsPerExpTime % (uint) stepsPerEcho;
		}

		this->exposureTimes[time] = stepsPerExpTime * timePerStep;
	}
}

void NMR_PFGSE::setName()
{
	// string big_delta = std::to_string((int) this->exposureTime) + "-" 
	// 				   + std::to_string(((int) (this->exposureTime * 100)) % 100);
	// string tiny_delta = std::to_string((int) this->pulseWidth);
	// string gamma = std::to_string((int) this->giromagneticRatio);
	// this->name = "/NMR_pfgse_" + big_delta + "ms_" + tiny_delta + "ms_" + gamma + "_sT";
	this->name = "/NMR_pfgse_timesample_" + std::to_string((*this).getCurrentTime());
}

void NMR_PFGSE::createDirectoryForData()
{
	string path = this->NMR.getDBPath();
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

	if(this->Mkt_stdev.size() > 0) this->Mkt_stdev.clear();
	this->Mkt_stdev.reserve(this->gradientPoints);
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
		this->vecGradient.push_back(newGradient);
		this->gradient.push_back(newGradient.getNorm());
		gvalueX += gapX;
		gvalueY += gapY;
		gvalueZ += gapZ;
	}
}

void NMR_PFGSE::setVectorK()
{
	if(this->vecK.size() > 0) this->vecK.clear();
	this->vecK.reserve(this->gradientPoints);

	double Kx, Ky, Kz;
	for(uint index = 0; index < this->gradientPoints; index++)
	{
		Kx = (*this).computeWaveVectorK(this->vecGradient[index].getX(), (*this).getPulseWidth(), (*this).getGiromagneticRatio());
		Ky = (*this).computeWaveVectorK(this->vecGradient[index].getY(), (*this).getPulseWidth(), (*this).getGiromagneticRatio());
		Kz = (*this).computeWaveVectorK(this->vecGradient[index].getZ(), (*this).getPulseWidth(), (*this).getGiromagneticRatio());
		Vector3D Knew(Kx, Ky, Kz);
		this->vecK.push_back(Knew);
	}
}

void NMR_PFGSE::setNMRTimeFramework()
{
	cout << endl << "-- Exposure time: " << this->exposureTime << " ms";
	this->NMR.setTimeFramework(this->exposureTime);
	cout << " [" << this->NMR.simulationSteps << " RW-steps]" << endl;
}

void NMR_PFGSE::runInitialMapSimulation()
{
	if(this->exposureTimes.size() > 0)
	{	
		double longestTime = (*this).getExposureTime(this->exposureTimes.size() - 1);
		uint mapSteps = 40000;
		bool mapByTime = true;
		if(mapByTime) this->NMR.setTimeFramework(longestTime);
		else this->NMR.setTimeFramework(mapSteps);
		
		cout << "- Initial map time: ";
		if(mapByTime) cout << longestTime << " ms ";
		cout << "[" << this->NMR.simulationSteps << " RW-steps]" << endl;
		this->NMR.mapSimulation();
		(*this).updateWalkersXIrate(this->NMR.simulationSteps);
		// this->NMR.updateRelaxativity(); but what rho to adopt?

		string path = this->NMR.getDBPath();
		if(this->PFGSE_config.getSaveCollisions())
	    {
	        this->NMR.saveWalkerCollisions(path + this->NMR.simulationName);
	    }
	}
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
		this->DsatAdjustSamples = idx;
	}
}

void NMR_PFGSE::setThresholdFromSamples(int _samples)
{
	if(_samples <= this->gradientPoints && _samples > 1)	
	{
		this->DsatAdjustSamples = _samples;
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

	if(this->LHS_stdev.size() > 0) this->LHS_stdev.clear();
	this->LHS_stdev.reserve(this->gradientPoints);
}

double NMR_PFGSE::computeLHS(double _Mg, double _M0)
{
	return log(fabs(_Mg/_M0));
	// return log(fabs(_Mg));

}

double NMR_PFGSE::computeWaveVectorK(double gradientMagnitude, double pulse_width, double giromagneticRatio)
{
    return (pulse_width * 1.0e-03) *  (giromagneticRatio * 1.0e+06) * (gradientMagnitude * 1.0e-08);
}

void NMR_PFGSE::runSequence()
{
	// run pfgse experiment -- this method will fill Mkt vector
	(*this).simulation();
}

void NMR_PFGSE::simulation()
{
	if(this->NMR.gpu_use == true)
	{
		if(this->NMR.getBoundaryCondition() == "noflux") { (*this).simulation_cuda_noflux(); } 
		else if(this->NMR.getBoundaryCondition() == "periodic") { (*this).simulation_cuda_periodic(); } 
		else if(this->NMR.getBoundaryCondition() == "mirror") { (*this).simulation_cuda_mirror(); }
		else cout << "error: BC not set" << endl;
	}
	else
	{
		(*this).simulation_omp();
	}
}

void NMR_PFGSE::recoverDsat()
{
	cout << "- Stejskal-Tanner (s&t) ";
	double time = omp_get_wtime();

	if(this->PFGSE_config.getAllowWalkerSampling())
	{
		cout << "with sampling:" <<  endl;
		(*this).recoverDsatWithSampling();
	} else	
	{
		cout << "without sampling:" <<  endl;
		(*this).recoverDsatWithoutSampling();
	}

	cout << "in " << omp_get_wtime() - time << " seconds." << endl;
}

void NMR_PFGSE::recoverDmsd()
{
	cout << "- Mean squared displacement (msd) " << endl;
	double time = omp_get_wtime();

	if(this->PFGSE_config.getAllowWalkerSampling())
	{
		cout << "with sampling:" <<  endl;
		(*this).recoverDmsdWithSampling();
	} else	
	{
		cout << "without sampling:" <<  endl;
		(*this).recoverDmsdWithoutSampling();
	}

	cout << "in " << omp_get_wtime() - time << " seconds." << endl;
}

void NMR_PFGSE::recoverDsatWithoutSampling()
{
	// Get magnetization levels 
	// get M0 (reference value)
	int idx_begin = 0;
	int idx_end = this->gradientPoints;	

	this->M0 = this->Mkt[0];
	this->LHS.push_back((*this).computeLHS(M0, M0));
	idx_begin++;

	double lhs_value;
	for(uint point = idx_begin; point < idx_end; point++)
	{	
		lhs_value = (*this).computeLHS(this->Mkt[point], M0);
		this->LHS.push_back(lhs_value);
	}

	// fill standard deviation vectors with null values
	for(uint point = 0; point < this->gradientPoints; point++)
	{
		this->Mkt_stdev.push_back(0.0);
		this->LHS_stdev.push_back(0.0);
	}

	(*this).applyThreshold();
	cout << "points to sample: " << this->DsatAdjustSamples << endl;
	vector<double> RHS_buffer; RHS_buffer.reserve(this->DsatAdjustSamples);
	vector<double> LHS_buffer; LHS_buffer.reserve(this->DsatAdjustSamples);
	// fill RHS data buffer only once
	for(int point = 0; point < this->DsatAdjustSamples; point++)
	{
		RHS_buffer.push_back(this->RHS[point]);
		LHS_buffer.push_back(this->LHS[point]);
	}
	LeastSquareAdjust lsa(RHS_buffer, LHS_buffer);
	lsa.setPoints(this->DsatAdjustSamples);
	lsa.solve();
	(*this).setD_sat(lsa.getB());

	// log results
	cout << "D(" << (*this).getExposureTime((*this).getCurrentTime()) << " ms) {s&t} = " << (*this).getD_sat() << endl;	
}

double ** NMR_PFGSE::getSamplesMagnitude()
{
	if(this->NMR.gpu_use == true)
	{
		return (*this).computeSamplesMagnitudeWithGpu();
	}
	else
	{
		if(this->NMR.rwNMR_config.getOpenMPUsage())
		{
			return (*this).computeSamplesMagnitudeWithOmp();
		} else
		{	
			return (*this).computeSamplesMagnitude();
		}
	}
}

double ** NMR_PFGSE::computeSamplesMagnitudeWithOmp()
{
	/* 
		alloc table for Mkt data each row will represent a wavevector K value, 
		while each column represent a sample of random walkers
	*/
	double **Mkt_samples;
	Mkt_samples = new double*[this->gradientPoints];
	for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
	{
		Mkt_samples[kIdx] = new double[this->NMR.walkerSamples];
	}

	/*
		initialize each element in table with zeros
	*/
	for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
	{
		for(int sample = 0; sample < this->NMR.walkerSamples; sample++)
		{
			Mkt_samples[kIdx][sample] = 0.0;
		}
	}	

	double resolution = this->NMR.getImageVoxelResolution();
	double phase;
	double dX, dY, dZ;

    // set omp variables for parallel loop throughout walker list
    const int walkersPerSample = this->NMR.numberOfWalkers / this->NMR.walkerSamples;	
	const int num_cpu_threads = omp_get_max_threads();
    int loop_start, loop_finish;
	const int loop_size = this->NMR.walkerSamples;	

    /*
		collect sum of data from phaseMagnitudes table
	*/
	#pragma omp parallel shared(Mkt_samples, resolution) private(loop_start, loop_finish, phase, dX, dY, dZ) 
    {
        const int thread_id = omp_get_thread_num();
        OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
        loop_start = looper.getStart();
        loop_finish = looper.getFinish(); 

        cout << "thread " << thread_id << " here with omp ^^" << endl;

        for (uint sampleId = loop_start; sampleId < loop_finish; sampleId++)
        {			
			int offset = sampleId * walkersPerSample;
			for(uint idx = 0; idx < walkersPerSample; idx++)
	        {
	            // Get walker displacement
				dX = ((double) this->NMR.walkers[offset + idx].initialPosition.x - (double) this->NMR.walkers[offset + idx].position_x);
				dY = ((double) this->NMR.walkers[offset + idx].initialPosition.y - (double) this->NMR.walkers[offset + idx].position_y);
				dZ = ((double) this->NMR.walkers[offset + idx].initialPosition.z - (double) this->NMR.walkers[offset + idx].position_z);
				Vector3D dR(resolution * dX, resolution * dY, resolution * dZ);
					
				for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
				{
					phase = this->vecK[kIdx].dotProduct(dR);	
					Mkt_samples[kIdx][sampleId] += cos(phase) * this->NMR.walkers[idx].energy;	
				}
	        }

		}
	}

	return Mkt_samples;
 }

double ** NMR_PFGSE::computeSamplesMagnitude()
{
	/* 
		alloc table for Mkt data each row will represent a wavevector K value, 
		while each column represent a sample of random walkers
	*/
	double **Mkt_samples;
	Mkt_samples = new double*[this->gradientPoints];
	for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
	{
		Mkt_samples[kIdx] = new double[this->NMR.walkerSamples];
	}

	/*
		initialize each element in table with zeros
	*/
	for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
	{
		for(int sample = 0; sample < this->NMR.walkerSamples; sample++)
		{
			Mkt_samples[kIdx][sample] = 0.0;
		}
	}	

	double resolution = this->NMR.getImageVoxelResolution();
	double phase;
	double dX, dY, dZ;

	int walkersPerSample = this->NMR.numberOfWalkers / this->NMR.walkerSamples;	
	for(int sample = 0; sample < this->NMR.walkerSamples; sample++)
	{		
		int offset = sample * walkersPerSample;	
		for(uint idx = 0; idx < walkersPerSample; idx++)
		{
			// Get walker displacement
			dX = ((double) this->NMR.walkers[offset + idx].initialPosition.x - (double) this->NMR.walkers[offset + idx].position_x);
			dY = ((double) this->NMR.walkers[offset + idx].initialPosition.y - (double) this->NMR.walkers[offset + idx].position_y);
			dZ = ((double) this->NMR.walkers[offset + idx].initialPosition.z - (double) this->NMR.walkers[offset + idx].position_z);
			Vector3D dR(resolution * dX, resolution * dY, resolution * dZ);
			
			for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
			{
				phase = this->vecK[kIdx].dotProduct(dR);
				Mkt_samples[kIdx][sample] += cos(phase) * this->NMR.walkers[idx].energy;;	
			}
		}		 
	}

	return Mkt_samples;
}

void NMR_PFGSE::recoverDsatWithSampling()
{
	bool time_verbose = false;
	int walkersPerSample = this->NMR.numberOfWalkers / this->NMR.walkerSamples;	
	double tick, phaseTime, normTime, lhsTime, statTime, lsTime;	

	/* 
		Compute magnitude Mkt for each sample of walkers
	*/
	tick = omp_get_wtime();
	double **Mkt_samples;
	Mkt_samples = (*this).getSamplesMagnitude();	
	phaseTime = omp_get_wtime() - tick;

	// Normalize for k=0
	tick = omp_get_wtime();
	for(int sample = 0; sample < this->NMR.walkerSamples; sample++)
	{
		double M0t = Mkt_samples[0][sample];
		for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
		{
			Mkt_samples[kIdx][sample] /= M0t;
		}
	}
	normTime = omp_get_wtime() - tick;

	/* 
		alloc table for LHS data each row will represent a wavevector K value, 
		while each column represent a sample of random walkers
	*/
	double **LHS_samples;
	LHS_samples = new double*[this->gradientPoints];
	for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
	{
		LHS_samples[kIdx] = new double[this->NMR.walkerSamples];
	} 	
	
	/*
		compute lhs for each sample
	*/
	tick = omp_get_wtime();
	double lhs_value;
	for(int sample = 0; sample < this->NMR.walkerSamples; sample++)
	{
		for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
		{	
			LHS_samples[kIdx][sample] = (*this).computeLHS(Mkt_samples[kIdx][sample], Mkt_samples[0][sample]);
		}
	}
	lhsTime = omp_get_wtime() - tick;

	/*
		 get data statistics 
	*/
	tick = omp_get_wtime();
	vector<double> meanMkt; meanMkt.reserve(this->gradientPoints);
	vector<double> stDevMkt; stDevMkt.reserve(this->gradientPoints);
	vector<double> meanLHS; meanLHS.reserve(this->gradientPoints);
	vector<double> stDevLHS; stDevLHS.reserve(this->gradientPoints);
	for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
	{
		meanMkt.push_back((*this).mean(Mkt_samples[kIdx], this->NMR.walkerSamples));
		stDevMkt.push_back((*this).stdDev(Mkt_samples[kIdx], this->NMR.walkerSamples, meanMkt[kIdx]));
		meanLHS.push_back((*this).mean(LHS_samples[kIdx], this->NMR.walkerSamples));
		stDevLHS.push_back((*this).stdDev(LHS_samples[kIdx], this->NMR.walkerSamples, meanLHS[kIdx]));
	}
	statTime = omp_get_wtime() - tick;

	
	// copy data to class members
	this->Mkt = meanMkt;
	this->Mkt_stdev = stDevMkt;
	this->LHS = meanLHS;
	this->LHS_stdev = stDevLHS;

	/*
		Stejskal-Tanner (s&t)
	*/
	tick = omp_get_wtime();
	vector<double> Dsat; Dsat.reserve(this->NMR.walkerSamples);
	(*this).applyThreshold();
	cout << "points to sample: " << this->DsatAdjustSamples << endl;

	vector<double> RHS_buffer; RHS_buffer.reserve(this->DsatAdjustSamples);
	vector<double> LHS_buffer; LHS_buffer.reserve(this->DsatAdjustSamples);
	// fill RHS data buffer only once
	for(int point = 0; point < this->DsatAdjustSamples; point++)
	{
		RHS_buffer.push_back(this->RHS[point]);
	}

	for(int sample = 0; sample < this->NMR.walkerSamples; sample++)
	{
		// fill LHS data buffer for each sample
		if(LHS_buffer.size() > 0)
		{
			LHS_buffer.clear();
		}
		for(int point = 0; point < this->DsatAdjustSamples; point++)
		{
			LHS_buffer.push_back(LHS_samples[point][sample]);
		}

		LeastSquareAdjust lsa(RHS_buffer, LHS_buffer);
		lsa.setPoints(this->DsatAdjustSamples);
		lsa.solve();
		Dsat.push_back(lsa.getB());		
	}
	lsTime = omp_get_wtime() - tick;	

	// 
	double meanDsat = (*this).mean(Dsat);
	(*this).setD_sat(meanDsat);
	(*this).setD_sat_StdDev(((*this).stdDev(Dsat, meanDsat)));

	// log results	
	cout << "D(" << (*this).getExposureTime((*this).getCurrentTime()) << " ms) {s&t} = " << (*this).getD_sat();
	cout << " +/- " << (*this).getD_sat_stdev() << endl;

	if(time_verbose)
    {
        cout << "--- Time analysis ---" << endl;
        cout << "phase computation: " << phaseTime << " s" << endl;
        cout << "data normalization: " << normTime << " s" << endl;
        cout << "LHS computation: " << lhsTime << " s" << endl;
        cout << "stats computation: " << statTime << " s" << endl;
        cout << "least-squares computation: " << lsTime << " s" << endl;
        cout << "---------------------" << endl;
    }

    /*
		delete data from pointers
    */
	// free data for Mkt_samples
	for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
	{
		delete [] Mkt_samples[kIdx];
		Mkt_samples[kIdx] = NULL;
	}
	delete [] Mkt_samples;
	Mkt_samples = NULL; 

	// free data for LHS_samples
	for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
	{
		delete [] LHS_samples[kIdx];
		LHS_samples[kIdx] = NULL;
	}
	delete [] LHS_samples;
	LHS_samples = NULL; 
}

void NMR_PFGSE::recoverDmsdWithoutSampling()
{
	double squaredDisplacement = 0.0;
	double displacementX, displacementY, displacementZ;
	double X0, Y0, Z0;
	double XF, YF, ZF;
	double normalizedDisplacement;
	double nDx = 0.0; double nDy = 0.0; double nDz = 0.0;
	double resolution = this->NMR.getImageVoxelResolution();
	double aliveWalkerFraction = 0.0;

	// debug
	// int imgX, imgY, imgZ;
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

		nDx += (particle.energy * displacementX * displacementX);
		nDy += (particle.energy * displacementY * displacementY);
		nDz += (particle.energy * displacementZ * displacementZ);

		normalizedDisplacement = displacementX*displacementX + 
								 displacementY*displacementY + 
								 displacementZ*displacementZ;

		squaredDisplacement += (particle.energy * normalizedDisplacement);
		aliveWalkerFraction += particle.energy;
 	}

	// set diffusion coefficient (see eq 2.18 - ref. Bergman 1995)
	squaredDisplacement = squaredDisplacement / aliveWalkerFraction;
	(*this).setD_msd(squaredDisplacement/(6.0 * (*this).getExposureTime()));
	(*this).setMsd(squaredDisplacement);

	nDx /= aliveWalkerFraction;
	nDy /= aliveWalkerFraction;
	nDz /= aliveWalkerFraction;
	(*this).setVecMsd(nDx, nDy, nDz);
	(*this).setVecDmsd((nDx / (2.0 * (*this).getExposureTime())), 
					   (nDy / (2.0 * (*this).getExposureTime())), 
					   (nDz / (2.0 * (*this).getExposureTime()))); 

	
	cout << "D(" << (*this).getExposureTime((*this).getCurrentTime()) << " ms) {msd} = " << (*this).getD_msd() << endl;
	cout << "Dxx = " << this->vecDmsd.getX() << ", \t";
	cout << "Dyy = " << this->vecDmsd.getY() << ", \t";
	cout << "Dzz = " << this->vecDmsd.getZ() << endl;
}

void NMR_PFGSE::recoverDmsdWithSampling()
{
	double time = omp_get_wtime();
	int walkersPerSample = this->NMR.numberOfWalkers / this->NMR.walkerSamples;
	double squaredDisplacement;
	double displacementX, displacementY, displacementZ;
	double X0, Y0, Z0;
	double XF, YF, ZF;
	double normalizedDisplacement;
	double nDx, nDy, nDz;
	double resolution = this->NMR.getImageVoxelResolution();
	double aliveWalkerFraction;

	vector<double> Dmsd; Dmsd.reserve(this->NMR.walkerSamples);
	vector<double> DmsdX; DmsdX.reserve(this->NMR.walkerSamples);
	vector<double> DmsdY; DmsdY.reserve(this->NMR.walkerSamples);
	vector<double> DmsdZ; DmsdZ.reserve(this->NMR.walkerSamples);
	vector<double> msd; msd.reserve(this->NMR.walkerSamples);
	vector<double> msdX; msdX.reserve(this->NMR.walkerSamples);
	vector<double> msdY; msdY.reserve(this->NMR.walkerSamples);
	vector<double> msdZ; msdZ.reserve(this->NMR.walkerSamples);

	// measure msd and Dmsd for each sample of walkers
	for(int sample = 0; sample < this->NMR.walkerSamples; sample++)
	{	
		squaredDisplacement = 0.0;
		nDx = 0.0; nDy = 0.0; nDz = 0.0;
		aliveWalkerFraction = 0.0;

		for(uint idx = 0; idx < walkersPerSample; idx++)
		{
			int offset = sample * walkersPerSample;
			Walker particle(this->NMR.walkers[idx + offset]);

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

			nDx += (particle.energy * displacementX * displacementX);
			nDy += (particle.energy * displacementY * displacementY);
			nDz += (particle.energy * displacementZ * displacementZ);

			normalizedDisplacement = displacementX*displacementX + 
									 displacementY*displacementY + 
									 displacementZ*displacementZ;

			squaredDisplacement += (particle.energy * normalizedDisplacement);
			aliveWalkerFraction += particle.energy;
	 	}

		// set diffusion coefficient (see eq 2.18 - ref. Bergman 1995)
		squaredDisplacement = squaredDisplacement / aliveWalkerFraction;
		Dmsd.push_back(squaredDisplacement/(6.0 * (*this).getExposureTime()));
		msd.push_back(squaredDisplacement);

		nDx /= aliveWalkerFraction;
		nDy /= aliveWalkerFraction;
		nDz /= aliveWalkerFraction;

		msdX.push_back(nDx);
		msdY.push_back(nDy);
		msdZ.push_back(nDz);
		DmsdX.push_back((nDx / (2.0 * (*this).getExposureTime())));
		DmsdY.push_back((nDy / (2.0 * (*this).getExposureTime())));
		DmsdZ.push_back((nDz / (2.0 * (*this).getExposureTime())));

	}

	// measure mean value among all the samples
	double meanDmsd = (*this).mean(Dmsd);
	double meanDmsdX = (*this).mean(DmsdX);
	double meanDmsdY = (*this).mean(DmsdY);
	double meanDmsdZ = (*this).mean(DmsdZ);
	double meanMsd = (*this).mean(msd);
	double meanMsdX = (*this).mean(msdX);
	double meanMsdY = (*this).mean(msdY);
	double meanMsdZ = (*this).mean(msdZ);
	
	// set mean value among all the samples
	(*this).setMsd(meanMsd);
	(*this).setD_msd(meanDmsd);
	(*this).setVecMsd(meanMsdX, meanMsdY, meanMsdZ);
	(*this).setVecDmsd(meanDmsdX, meanDmsdY, meanDmsdZ);

	// set std deviation among all the samples
	(*this).setD_msd_StdDev((*this).stdDev(Dmsd, meanDmsd));
	(*this).setMsdStdDev((*this).stdDev(msd, meanMsd));
	(*this).setVecDmsdStdDev((*this).stdDev(DmsdX, meanDmsdX), (*this).stdDev(DmsdY, meanDmsdY), (*this).stdDev(DmsdZ, meanDmsdZ));
	(*this).setVecMsdStdDev((*this).stdDev(msdX, meanMsdX), (*this).stdDev(msdY, meanMsdY), (*this).stdDev(msdZ, meanMsdZ));
	
	// print results
	cout << "D(" << (*this).getExposureTime((*this).getCurrentTime()) << " ms) {msd} = " << (*this).getD_msd();
	cout << " +/- " << (*this).getD_msd_stdev() << endl;
	cout << "Dxx = " << this->vecDmsd.getX() << " +/- " << this->vecDmsd_stdev.getX() << endl;
	cout << "Dyy = " << this->vecDmsd.getY() << " +/- " << this->vecDmsd_stdev.getY() << endl;
	cout << "Dzz = " << this->vecDmsd.getZ() << " +/- " << this->vecDmsd_stdev.getZ() << endl;	
}

void NMR_PFGSE::reset(double newBigDelta)
{
	(*this).clear();
	(*this).setExposureTime(newBigDelta);
	(*this).set();
	(*this).setThresholdFromSamples(this->gradientPoints);
}

void NMR_PFGSE::reset()
{
	(*this).clear();
	(*this).set();
	(*this).setThresholdFromSamples(this->gradientPoints);
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
    cout << "- saving results...";
    
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
	if(this->PFGSE_config.getSavePFGSE())
	{
		(*this).writeResults();
	}
	
	time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}

void NMR_PFGSE::writeResults()
{
	(*this).writeParameters();
	(*this).writeEchoes();
	(*this).writeGvector();
	(*this).writeMsd();
}

void NMR_PFGSE::writeEchoes()
{
	string filename = this->dir + "/PFGSE_echoes.txt";

	ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    file << "PFGSE - Echoes and Stejskal-Tanner equation terms" << endl;
    file << "id, ";
    file << "Gradient, ";
    file << "M(k,t) [mean, std], ";
    file << "LHS [mean, std], ";
    file << "RHS" << endl;

    uint size = this->gradientPoints;
    for (uint index = 0; index < size; index++)
    {
        file << index << ", ";
        file << this->gradient[index] << ", ";
        file << this->Mkt[index] << ", ";
        file << this->Mkt_stdev[index] << ", ";
        file << this->LHS[index] << ", ";
        file << this->LHS_stdev[index] << ", ";
        file << this->RHS[index] << endl;
    }

    file.close();
}

void NMR_PFGSE::writeParameters()
{
	string filename = this->dir + "/PFGSE_parameters.txt";

	ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

	file << "RWNMR-PFGSE Results" << endl; 
	file << "Points: " << this->gradientPoints << endl;
    file << "Time: " << this->exposureTime << endl;
    file << "Pulse width: " << this->pulseWidth << endl;
    file << "Giromagnetic Ratio: " << this->giromagneticRatio << endl;
	file << "D_0: " << this->NMR.getDiffusionCoefficient() << endl;
    file << "D_sat: " << this->D_sat << endl;
    file << "D_sat (stdev): " << this->D_sat_stdev << endl;
    file << "D_sat adjust points: " << this->DsatAdjustSamples << endl;
    file << "D_msd: " << this->D_msd << endl;
    file << "D_msd (stdev): " << this->D_msd_stdev << endl;
    file << "MSD: " << this->msd << endl;
    file << "MSD (stdev): " << this->msd_stdev << endl;   

    file.close();
}

void NMR_PFGSE::writeGvector()
{
	string filename = this->dir + "/PFGSE_gradient.txt";

	ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    file << "PFGSE - Gradient values" << endl;
    file << "id, ";
    file << "Gx, ";
    file << "Gy, ";
    file << "Gz, ";
    file << "Kx, ";
    file << "Ky, ";
    file << "Kz" << endl;

    uint size = this->gradientPoints;
    for (uint index = 0; index < size; index++)
    {
        file << index << ", ";
        file << this->vecGradient[index].getX() << ", ";
        file << this->vecGradient[index].getY() << ", ";
        file << this->vecGradient[index].getZ() << ", ";
        file << this->vecK[index].getX() << ", ";
        file << this->vecK[index].getY() << ", ";
        file << this->vecK[index].getZ() << endl;
    }

    file.close();
}

void NMR_PFGSE::writeMsd()
{
	string filename = this->dir + "/PFGSE_msd.txt";

	ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

	int walkersPerSample = this->NMR.numberOfWalkers;
	if(this->PFGSE_config.getAllowWalkerSampling())
		walkersPerSample /= this->NMR.walkerSamples;
    
	file << "PFGSE - Mean squared displacement values - " << this->NMR.walkerSamples << " samples of " << walkersPerSample << " walkers" << endl;
    file << "msdX[mean, std], ";
    file << "msdX[mean, std], ";
    file << "msdZ[mean, std], ";
    file << "DmsdX[mean, std], ";
    file << "DmsdY[mean, std], ";
    file << "DmsdZ[mean, std]" << endl;

    file << this->vecMsd.getX() << ", " << this->vecMsd_stdev.getX() << ", ";
    file << this->vecMsd.getY() << ", " << this->vecMsd_stdev.getY() << ", ";
    file << this->vecMsd.getZ() << ", " << this->vecMsd_stdev.getZ() << ", ";
    file << this->vecDmsd.getX() << ", " << this->vecDmsd_stdev.getX() << ", ";
    file << this->vecDmsd.getY() << ", " << this->vecDmsd_stdev.getY() << ", ";
    file << this->vecDmsd.getZ() << ", " << this->vecDmsd_stdev.getZ();  

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

    // reset vector to store energy decay
    this->NMR.resetGlobalEnergy();
    this->NMR.globalEnergy.reserve(this->NMR.numberOfEchoes + 1); // '+1' to accomodate time 0.0

    // get initial energy global state
    double energySum = ((double) this->NMR.walkers.size()) * this->NMR.walkers[0].getEnergy();
    this->NMR.globalEnergy.push_back(energySum);


    // set derivables
    double gamma = this->giromagneticRatio;
    if(!this->PFGSE_config.getUseWaveVectorTwoPi()) gamma /= TWO_PI;

	myAllocator arrayFactory; 
	double *globalPhase = arrayFactory.getDoubleArray(this->gradientPoints);
    double globalEnergy = 0.0;
    double resolution = this->NMR.imageVoxelResolution;
    
    // main loop
	// reset walker's initial state with omp parallel for
    if(this->NMR.rwNMR_config.getOpenMPUsage())
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->NMR.walkers.size();
        int loop_start, loop_finish; 

		#pragma omp parallel shared(gamma, globalPhase, globalEnergy, resolution) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish(); 

			double walkerPhase;
			double walkerEnergy;

            for(uint id = loop_start; id < loop_finish; id++)
            {
				// reset energy
				this->NMR.walkers[id].resetPosition();
				this->NMR.walkers[id].resetSeed();
				this->NMR.walkers[id].resetEnergy();
				
				// make walkers walk througout image
				for (uint step = 0; step < this->NMR.simulationSteps; step++)
				{
					this->NMR.walkers[id].walk(this->NMR.bitBlock);     
				}

				// get final individual signal
				walkerEnergy = this->NMR.walkers[id].energy;
				#pragma omp critical
				{
					globalEnergy += walkerEnergy;
				}

				// get final individual phase
				double dX = ((double) this->NMR.walkers[id].position_x) - ((double) this->NMR.walkers[id].initialPosition.x);
				double dY = ((double) this->NMR.walkers[id].position_y) - ((double) this->NMR.walkers[id].initialPosition.y);
				double dZ = ((double) this->NMR.walkers[id].position_z) - ((double) this->NMR.walkers[id].initialPosition.z);

				Vector3D dR(dX,dY,dZ);
				Vector3D wavevector_k;
				for(int point = 0; point < this->gradientPoints; point++)
				{ 
					double kx = computeWaveVectorK(this->vecGradient[point].getX(), (*this).getPulseWidth(), gamma);
					double ky = computeWaveVectorK(this->vecGradient[point].getY(), (*this).getPulseWidth(), gamma);
					double kz = computeWaveVectorK(this->vecGradient[point].getZ(), (*this).getPulseWidth(), gamma);
					wavevector_k.setX(kx);
					wavevector_k.setY(ky);
					wavevector_k.setZ(kz);
					wavevector_k.setNorm();
					
					walkerPhase = walkerEnergy * cos(wavevector_k.dotProduct(dR) * resolution);

					// add contribution to global sum
					globalPhase[point] += walkerPhase;
				}
			}
		}
	} else
	{
		double walkerPhase;
		double walkerEnergy;

		for(uint id = 0; id < this->NMR.walkers.size(); id++)
		{
			// reset energy
			this->NMR.walkers[id].resetPosition();
			this->NMR.walkers[id].resetSeed();
			this->NMR.walkers[id].resetEnergy();
			
			// make walkers walk througout image
			for (uint step = 0; step < this->NMR.simulationSteps; step++)
			{
				this->NMR.walkers[id].walk(this->NMR.bitBlock);     
			}

			// get final individual signal
			walkerEnergy = this->NMR.walkers[id].energy;
			globalEnergy += walkerEnergy;
			

			// get final individual phase
			double dX = ((double) this->NMR.walkers[id].position_x) - ((double) this->NMR.walkers[id].initialPosition.x);
			double dY = ((double) this->NMR.walkers[id].position_y) - ((double) this->NMR.walkers[id].initialPosition.y);
			double dZ = ((double) this->NMR.walkers[id].position_z) - ((double) this->NMR.walkers[id].initialPosition.z);

			Vector3D dR(dX,dY,dZ);
			Vector3D wavevector_k;
			for(int point = 0; point < this->gradientPoints; point++)
			{ 
				double kx = computeWaveVectorK(this->vecGradient[point].getX(), (*this).getPulseWidth(), gamma);
				double ky = computeWaveVectorK(this->vecGradient[point].getY(), (*this).getPulseWidth(), gamma);
				double kz = computeWaveVectorK(this->vecGradient[point].getZ(), (*this).getPulseWidth(), gamma);
				wavevector_k.setX(kx);
				wavevector_k.setY(ky);
				wavevector_k.setZ(kz);
				wavevector_k.setNorm();
				
				walkerPhase = walkerEnergy * cos(wavevector_k.dotProduct(dR) * resolution);

				// add contribution to global sum
				globalPhase[point] += walkerPhase;
			}
		}
	}
	
	
	// get magnitudes M(k,t)
    for(int point = 0; point < this->gradientPoints; point++)
    {
        this->Mkt.push_back((globalPhase[point]/globalEnergy));
    }

	// delete global phase array
	delete [] globalPhase;
	globalPhase = NULL;

    double finish_time = omp_get_wtime();
    cout << "Completed."; printElapsedTime(begin_time, finish_time);
    return;
}

double NMR_PFGSE::mean(vector<double> &_vec)
{
	double sum = 0;
	double size = (double) _vec.size();

    for (uint id = 0; id < _vec.size(); id++)
    {
        sum += _vec[id];
    }

    return (sum / size);
}

double NMR_PFGSE::mean(double *_vec, int _size)
{
	double sum = 0;

    for (uint id = 0; id < _size; id++)
    {
        sum += _vec[id];
    }

    return (sum / (double) _size);
}

double NMR_PFGSE::stdDev(vector<double> &_vec)
{
    return (*this).stdDev(_vec, (*this).mean(_vec));
}

double NMR_PFGSE::stdDev(vector<double> &_vec, double mean)
{
	double sum = 0.0;
    int size = _vec.size();

    for(uint idx = 0; idx < _vec.size(); idx++)
    {
        sum += (_vec[idx] - mean) * (_vec[idx] - mean); 
    }

    return sqrt(sum/((double) size));
}

double NMR_PFGSE::stdDev(double *_vec, int _size)
{
    return (*this).stdDev(_vec, _size, (*this).mean(_vec, _size));
}

double NMR_PFGSE::stdDev(double *_vec, int _size, double mean)
{
	double sum = 0.0;

    for(uint idx = 0; idx < _size; idx++)
    {
        sum += (_vec[idx] - mean) * (_vec[idx] - mean); 
    }

    return sqrt(sum/((double) _size));
}