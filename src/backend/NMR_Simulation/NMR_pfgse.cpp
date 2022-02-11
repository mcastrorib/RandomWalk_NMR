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
										   D_sat(0.0),
										   D_sat_error(0.0),
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
	vector<double> rawNoise();
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
	(*this).setApplyBulkRelaxation(this->PFGSE_config.getApplyBulk());
	(*this).setNoiseAmp(this->PFGSE_config.getNoiseAmp());
	(*this).setThresholdFromSamples(this->gradientPoints);
	(*this).setGradientVector();
	(*this).setVectorK();

	// new
	(*this).setName();
	(*this).createDirectoryForData();
}

void NMR_PFGSE::set()
{
	// (*this).setName();
	// (*this).createDirectoryForData();
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
	(*this).presave();
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
	double threshold_value = this->PFGSE_config.getThresholdValue();
	uint threshold_window = this->PFGSE_config.getThresholdWindow();
	if(threshold_type == "lhs") (*this).setThresholdFromLHS(threshold_value, threshold_window);
	else if(threshold_type == "samples") (*this).setThresholdFromSamples(int(threshold_value));
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
	// apply 'physical' scaling
    if(this->PFGSE_config.getApplyScaleFactor())
    {
        double scale_factor = (this->PFGSE_config.getInspectionLength() * this->PFGSE_config.getInspectionLength()) / this->NMR.getDiffusionCoefficient();
        cout << "applying scale factor: " << scale_factor << endl;
        for(int time = 0; time < this->exposureTimes.size(); time++)
        	this->exposureTimes[time] *= scale_factor;
    }
	
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
	}
}

void NMR_PFGSE::setName()
{
	// this->name = "/NMR_pfgse_timesample_" + std::to_string((*this).getCurrentTime());

	// new
	this->name = "/NMR_pfgse";
}

void NMR_PFGSE::createDirectoryForData()
{
	string path = this->NMR.getDBPath() + this->NMR.getSimulationName();
    createDirectory(path, this->name);
    this->dir = (path + "/" + this->name);
    createDirectory(this->dir, "/timesamples");
    
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

		// Update xi_rate and relaxivity of walkers
		vector<double> rho = this->NMR.rwNMR_config.getRho();
		if(this->NMR.rwNMR_config.getRhoType() == "uniform")    
        {
        	this->NMR.updateWalkersRelaxativity(rho[0]);
    	} 
    	else if(this->NMR.rwNMR_config.getRhoType() == "sigmoid")
        {
        	this->NMR.updateWalkersRelaxativity(rho);
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

void NMR_PFGSE::setThresholdFromLHS(double _value, uint _window)
{

	if(this->LHS.size() < _window) 
		return;

	if(_value > 0.0 && _value < 1.0)
	{
		if((*this).getNoiseAmp() == 0.0)
		{
			(*this).setThresholdFromLHSValue(_value, _window);
		} 
		else
		{
			(*this).setThresholdFromLHSWindow(_value, _window);
		}
	}
}

void NMR_PFGSE::setThresholdFromLHSValue(double _value, uint _window)
{
	uint minSize = _window;
	int idx = minSize - 1;
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

void NMR_PFGSE::setThresholdFromLHSWindow(double _value, uint _window)
{
	if(this->LHS.size() < _window) 
		return;

	vector<double> windowValues;
	for(uint idx = 0; idx < _window; idx++) 
		windowValues.push_back(this->LHS[idx]);
	int idx = _window;
	bool isGreater = true;
	double logValue = log(_value);

	if((*this).mean(windowValues) < logValue)
	{
		isGreater = false;
	}
	
	while(idx < this->LHS.size() && isGreater == true)
	{
		uint currentIdx = idx % _window;
		windowValues[currentIdx] = this->LHS[idx];

		if((*this).mean(windowValues) < logValue)
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

	// apply bulk relaxation to signal
	if((*this).getApplyBulkRelaxation())
	{
		(*this).applyBulk();
	}

	// apply white noise to signal
	(*this).createNoiseVector();
	if((*this).getNoiseAmp() > 0.0)
	{
		(*this).applyNoiseToSignal();
	}
}

void NMR_PFGSE::applyBulk()
{
	double bulkTime = -1.0 / this->NMR.getBulkRelaxationTime();
	double bulkMagnitude = exp(bulkTime * (*this).getExposureTime());
	
	// Apply bulk relaxation in simulated signal
	for(uint kIdx = 0; kIdx < this->Mkt.size(); kIdx++)
	{
		this->Mkt[kIdx] *= bulkMagnitude;
	}
}

void NMR_PFGSE::createNoiseVector()
{
	if(this->rawNoise.size() != (*this).getGradientPoints()) 
		this->rawNoise.clear();

	this->rawNoise = getNormalDistributionSamples(0.0, 1.0, (*this).getGradientPoints());
	double M0 = (double) this->NMR.getNumberOfWalkers(); 
	for(int idx = 0; idx < (*this).getGradientPoints(); idx++)
	{
		this->rawNoise[idx] *= M0 * (*this).getNoiseAmp();
	}
}

void NMR_PFGSE::applyNoiseToSignal()
{
	// Add noise to signal
	if((*this).getNoiseAmp() > 0.0 and this->Mkt.size() == this->getGradientPoints())
	{
		for(uint kIdx = 0; kIdx < this->getGradientPoints(); kIdx++)
		{
			this->Mkt[kIdx] += this->rawNoise[kIdx];		
		}			
	}
}

void NMR_PFGSE::simulation()
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

void NMR_PFGSE::recoverDsat()
{
	cout << "- Stejskal-Tanner (s&t) ";
	double time = omp_get_wtime();

	if((this->NMR.getWalkerSamples() > 1) and this->PFGSE_config.getAllowWalkerSampling())
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

	// Normalize for k=0
	double M0 = this->Mkt[0];
	vector<double> normMkt;
	for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
	{
		normMkt.push_back(this->Mkt[kIdx] / M0);
	}
	

	for(uint point = idx_begin; point < idx_end; point++)
	{	
		// this->LHS.push_back((*this).computeLHS(this->Mkt[point], this->Mkt[0]));
		this->LHS.push_back((*this).computeLHS(normMkt[point], normMkt[0]));
	}

	// fill standard deviation vectors with null values
	for(uint point = 0; point < this->gradientPoints; point++)
	{
		this->Mkt_stdev.push_back(0.0);
		this->LHS_stdev.push_back(0.0);
	}

	(*this).applyThreshold();
	cout << "points to sample: " << this->DsatAdjustSamples << " where lhs: " << exp(this->LHS[this->DsatAdjustSamples-1]) << endl;
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
	double DstdError = sqrt(lsa.getMSE() * (((double) this->DsatAdjustSamples) /((double) this->DsatAdjustSamples - 1.0)));
	(*this).setD_sat_error(DstdError);

	// log results
	cout << "D(" << (*this).getExposureTime((*this).getCurrentTime()) << " ms) {s&t} = " << (*this).getD_sat();
	cout << " +/- " << 1.96 * (*this).getD_sat_error() << endl;	
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


	tick = omp_get_wtime();

	// Apply bulk relaxation in simulated signal
	if((*this).getApplyBulkRelaxation())
	{
		double bulkTime = -1.0 / this->NMR.getBulkRelaxationTime();
		double bulkMagnitude = exp(bulkTime * (*this).getExposureTime());
		for(int sample = 0; sample < this->NMR.walkerSamples; sample++)
		{
			for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
			{
				Mkt_samples[kIdx][sample] *= bulkMagnitude;
			}
		}
	}

	// Add noise to signal
	if((*this).getNoiseAmp() > 0.0)
	{
		// this factor is applied beacuse of the decreased magnetization Mkt
		double M0 = 1.0 / ((double) this->NMR.walkerSamples);
		for(int sample = 0; sample < this->NMR.walkerSamples; sample++)
		{
			for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
			{
				Mkt_samples[kIdx][sample] += M0 * this->rawNoise[kIdx];
			}
		}	
	}

	// Normalize for k=0
	for(int sample = 0; sample < this->NMR.walkerSamples; sample++)
	{
		double M0 = Mkt_samples[0][sample];
		for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
		{
			Mkt_samples[kIdx][sample] /= M0;
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
	vector<double> Dsat_error; Dsat_error.reserve(this->NMR.walkerSamples);
	double DstdError;
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
		DstdError = sqrt(lsa.getMSE() * (((double) this->DsatAdjustSamples) /((double) this->DsatAdjustSamples - 1.0)));
		Dsat_error.push_back(DstdError);		
	}
	lsTime = omp_get_wtime() - tick;	

	// 
	double meanDsat = (*this).mean(Dsat);
	double meanDsatError = (*this).mean(Dsat_error);
	(*this).setD_sat(meanDsat);
	(*this).setD_sat_error(meanDsatError);
	(*this).setD_sat_StdDev(((*this).stdDev(Dsat, meanDsat)));

	// log results	
	cout << "D(" << (*this).getExposureTime((*this).getCurrentTime()) << " ms) {s&t} = " << (*this).getD_sat();
	cout << " +/- " << 1.96 * (*this).getD_sat_error();
	cout << " [+/- " << (*this).getD_sat_stdev() << "]"<< endl;

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
	
	// Relaxation / Absorption equivalence
	double absorptionFraction;
	double absorption = 0.0;
	double nonAbsorption = 1.0;
	if(this->PFGSE_config.getApplyAbsorption())
	{
		absorption = 1.0;
		nonAbsorption = 0.0;
	}

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
		displacementX *= displacementX;
		
		// Y:
		Y0 = (double) particle.initialPosition.y;
		YF = (double) particle.position_y;
		displacementY = resolution * (YF - Y0);
		displacementY *= displacementY;

		// Z:
		Z0 = (double) particle.initialPosition.z;
		ZF = (double) particle.position_z;
		displacementZ = resolution * (ZF - Z0);
		displacementZ *= displacementZ;

		absorptionFraction = (absorption * particle.energy + nonAbsorption);
		aliveWalkerFraction += absorptionFraction;
		nDx += (absorptionFraction * displacementX);
		nDy += (absorptionFraction * displacementY);
		nDz += (absorptionFraction * displacementZ);
		normalizedDisplacement = displacementX + displacementY + displacementZ;
		squaredDisplacement += (absorptionFraction * normalizedDisplacement);
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

	// Relaxation / Absorption equivalence
	double absorptionFraction;
	double absorption = 0.0;
	double nonAbsorption = 1.0;
	if(this->PFGSE_config.getApplyAbsorption())
	{
		absorption = 1.0;
		nonAbsorption = 0.0;
	}

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
			displacementX *= displacementX;
			
			// Y:
			Y0 = (double) particle.initialPosition.y;
			YF = (double) particle.position_y;
			displacementY = resolution * (YF - Y0);
			displacementY *= displacementY;

			// Z:
			Z0 = (double) particle.initialPosition.z;
			ZF = (double) particle.position_z;
			displacementZ = resolution * (ZF - Z0);
			displacementZ *= displacementZ;

			absorptionFraction = (absorption * particle.energy + nonAbsorption);
			aliveWalkerFraction += absorptionFraction;
			nDx += (absorptionFraction * displacementX);
			nDy += (absorptionFraction * displacementY);
			nDz += (absorptionFraction * displacementZ);
			normalizedDisplacement = displacementX + displacementY + displacementZ;
			squaredDisplacement += (absorptionFraction * normalizedDisplacement);
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
	cout << "Dxx = " << this->vecDmsd.getX() << " +/- " << 1.96 * this->vecDmsd_stdev.getX() << endl;
	cout << "Dyy = " << this->vecDmsd.getY() << " +/- " << 1.96 * this->vecDmsd_stdev.getY() << endl;
	cout << "Dzz = " << this->vecDmsd.getZ() << " +/- " << 1.96 * this->vecDmsd_stdev.getZ() << endl;	
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

void NMR_PFGSE::presave()
{
	// write pfgse data
	if(this->PFGSE_config.getSavePFGSE())
	{
		(*this).createResultsFile();
		(*this).writeParameters();
		(*this).writeGvector();
	}
}

void NMR_PFGSE::save()
{
	double time = omp_get_wtime();
    cout << "- saving results...";
    
    // write pfgse data
	if(this->PFGSE_config.getSavePFGSE())
	{
		(*this).writeResults();
		(*this).writeEchoes();
		(*this).writeMsd();
	}

    if(this->PFGSE_config.getSaveWalkers())
    {
        (*this).writeWalkers();
    }

    if(this->PFGSE_config.getSaveHistogram())
    {
    	this->NMR.updateHistogram();
    	(*this).writeHistogram();
    }    

    if(this->PFGSE_config.getSaveHistogramList())
    {
        (*this).writeHistogramList();;
    }  
	
	time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
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

    Vector3D maxGradient(this->vecGradient[this->vecGradient.size() - 1]);
    const int precision = std::numeric_limits<double>::max_digits10;  
	file << "RWNMR-PFGSE Parameters" << endl; 
	file << setprecision(precision) << "D_0: " << this->NMR.getDiffusionCoefficient() << endl;  
    file << setprecision(precision) << "Pulse width: " << this->pulseWidth << endl;
    file << setprecision(precision) << "Giromagnetic Ratio: " << this->giromagneticRatio << endl;
	file << setprecision(precision) << "Gradient direction: {" 
		 << (maxGradient.getX() / maxGradient.getNorm()) << ", "
		 << (maxGradient.getY() / maxGradient.getNorm()) << ", "
		 << (maxGradient.getZ() / maxGradient.getNorm()) << "}" << endl;
	file << setprecision(precision) << "Gradients: " << this->gradientPoints << endl;
	file << "Times: [";
	for(int time = 0; time < this->exposureTimes.size(); time++)
	{
		if(time > 0) file << ", ";
		file << setprecision(precision) << this->exposureTimes[time];
	}
	file << "]" << endl; 
    
    file.close();
}

void NMR_PFGSE::writeGvector()
{
	string filename = this->dir + "/PFGSE_gradient.csv";

	ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    file << "Id,";
    file << "Gx,";
    file << "Gy,";
    file << "Gz,";
    file << "Kx,";
    file << "Ky,";
    file << "Kz" << endl;

    uint size = this->gradientPoints;
    const int precision = std::numeric_limits<double>::max_digits10;    
    for (uint index = 0; index < size; index++)
    {
        file << setprecision(precision) << index
        << "," << this->vecGradient[index].getX()
        << "," << this->vecGradient[index].getY()
        << "," << this->vecGradient[index].getZ()
        << "," << this->vecK[index].getX()
        << "," << this->vecK[index].getY()
        << "," << this->vecK[index].getZ() << endl;
    }

    file.close();
}

void NMR_PFGSE::writeEchoes()
{
	string filename = this->dir + "/timesamples/echoes_" + std::to_string((*this).getCurrentTime()) + ".csv";

	ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    file << "Idx,";
    file << "Gradient,";
    file << "NMR_signal(mean),NMR_signal(noise),NMR_signal(std),";
    file << "SAT_lhs(mean), SAT_lhs(std),";
    file << "SAT_rhs" << endl;

    uint size = this->gradientPoints;
    const int precision = std::numeric_limits<double>::max_digits10;
    for (uint index = 0; index < size; index++)
    {
        file << setprecision(precision) << index
        << "," << this->gradient[index]
        << "," << this->Mkt[index]
        << "," << this->rawNoise[index]
        << "," << this->Mkt_stdev[index]
        << "," << this->LHS[index]
        << "," << this->LHS_stdev[index]
        << "," << this->RHS[index] << endl;
    }

    file.close();
}

void NMR_PFGSE::writeMsd()
{
	string filename = this->dir + "/timesamples/msd_" + std::to_string((*this).getCurrentTime()) + ".csv";
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
    
    file << "msdX(mean),msdX(std),";
    file << "msdY(mean),msdY(std),";
    file << "msdZ(mean),msdZ(std),";
    file << "DmsdX(mean),DmsdX(std),";
    file << "DmsdY(mean),DmsdY(std),";
    file << "DmsdZ(mean),DmsdZ(std)" << endl;

    const int precision = std::numeric_limits<double>::max_digits10;
    file << setprecision(precision) << this->vecMsd.getX() << "," << this->vecMsd_stdev.getX() << ",";
    file << setprecision(precision) << this->vecMsd.getY() << "," << this->vecMsd_stdev.getY() << ",";
    file << setprecision(precision) << this->vecMsd.getZ() << "," << this->vecMsd_stdev.getZ() << ",";
    file << setprecision(precision) << this->vecDmsd.getX() << "," << this->vecDmsd_stdev.getX() << ",";
    file << setprecision(precision) << this->vecDmsd.getY() << "," << this->vecDmsd_stdev.getY() << ",";
    file << setprecision(precision) << this->vecDmsd.getZ() << "," << this->vecDmsd_stdev.getZ();  

    file.close();
}

void NMR_PFGSE::writeWalkers()
{
	string filename = this->dir + "/timesamples/walkers_" + std::to_string((*this).getCurrentTime()) + ".csv";
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

void NMR_PFGSE::writeHistogram()
{
	string filename = this->dir + "/timesamples/histogram_" + std::to_string((*this).getCurrentTime()) + ".csv";
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

void NMR_PFGSE::writeHistogramList()
{
	string filename = this->dir + "/timesamples/histList_" + std::to_string((*this).getCurrentTime()) + ".csv";
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
			file << setprecision(precision)	<< this->NMR.histogramList[hIdx].bins[i] << ",";
			file << setprecision(precision)	<< this->NMR.histogramList[hIdx].amps[i] << ",";
		}

		file << endl;
	}

	file.close();
}

void NMR_PFGSE::createResultsFile()
{
	string filename = this->dir + "/PFGSE_results.csv";

	ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

	file << "Time";
    file << ",D_sat";
    file << ",D_sat(error)";
    file << ",D_sat(std)";
    file << ",D_sat(pts)";
    file << ",D_msd";
    file << ",D_msd(std)";
    file << ",D_msdX";
    file << ",D_msdX(std)";
    file << ",D_msdY";
    file << ",D_msdY(std)";
    file << ",D_msdZ";
    file << ",D_msdZ(std)";
    file << endl;
    file.close();
}

void NMR_PFGSE::writeResults()
{
	string filename = this->dir + "/PFGSE_results.csv";

	ofstream file;
    file.open(filename, ios::app);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    const int precision = std::numeric_limits<double>::max_digits10;
    file << setprecision(precision)  << this->exposureTimes[this->getCurrentTime()]
    << "," << this->D_sat
    << "," << this->D_sat_error
    << "," << this->D_sat_stdev
    << "," << this->DsatAdjustSamples
    << "," << this->D_msd
    << "," << this->D_msd_stdev
    << "," << this->vecDmsd.getX()
    << "," << this->vecDmsd_stdev.getX()
    << "," << this->vecDmsd.getY()
    << "," << this->vecDmsd_stdev.getY()
    << "," << this->vecDmsd.getZ()
    << "," << this->vecDmsd_stdev.getZ()
    << endl; 
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

double NMR_PFGSE::sum(vector<double> &_vec)
{
	double sum = 0;
	double size = (double) _vec.size();

    for (uint id = 0; id < _vec.size(); id++)
    {
        sum += _vec[id];
    }

    return sum;
}

double NMR_PFGSE::sum(double *_vec, int _size)
{
	double sum = 0;

    for (uint id = 0; id < _size; id++)
    {
        sum += _vec[id];
    }

    return sum;
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

vector<double> NMR_PFGSE::getNormalDistributionSamples(const double loc, const double std, const int size)
{
	std::default_random_engine generator(this->NMR.getInitialSeed());
	std::normal_distribution<double> distribution(loc, std);
	vector<double> normal_dist;
	normal_dist.reserve(size);
	for (int i = 0; i < size; i++)
	{
		normal_dist.emplace_back(distribution(generator));
	}
	return normal_dist;
}