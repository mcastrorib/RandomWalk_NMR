#ifndef NMR_PFGSE_H
#define NMR_PFGSE_H

#include <vector>
#include <string>

// include configuration file classes
#include "../ConfigFiles/pfgse_config.h"

#include "NMR_defs.h"
#include "NMR_Simulation.h"
#include "../Walker/walker.h"
#include "../Math/Vector3D.h"


using namespace std;

class NMR_PFGSE
{
public:
	NMR_Simulation &NMR;
	pfgse_config PFGSE_config;
	string name;
	string dir;
	vector<double> gradient;
	double gradient_max;
	double noiseAmp;	
	vector<double> rawNoise;

	double gradient_X;
	double gradient_Y;
	double gradient_Z;
	vector<Vector3D> vecGradient;
	vector<Vector3D> vecK;
	vector<double> RHS;
	
	int gradientPoints;
	vector<double> exposureTimes;
	double exposureTime;
	double pulseWidth;
	double giromagneticRatio;
	bool applyBulkRelaxation;

	vector<double> Mkt;
	vector<double> Mkt_stdev;
	vector<double> LHS;
	vector<double> LHS_stdev;
	
	int DsatAdjustSamples;
	double D_sat;
	double D_sat_stdev;
	double D_msd;
	double D_msd_stdev;
	double msd;
	double msd_stdev;
	Vector3D vecMsd;
	Vector3D vecMsd_stdev;	
	Vector3D vecDmsd;
	Vector3D vecDmsd_stdev;

	uint stepsTaken;
	int currentTime;	

	NMR_PFGSE(NMR_Simulation &_NMR, 
			  pfgse_config _pfgseConfig,
			  int _mpi_rank = 0, 
			  int _mpi_processes = 0);	
	
	virtual ~NMR_PFGSE(){};

	void setNMRTimeFramework();
	void correctExposureTimes();
	void runInitialMapSimulation();
	void setGradientVector(double _GF, int _GPoints);
	void setGradientVector();
	void createNoiseVector();
	void setVectorK();
	void setVectorMkt();
	void setVectorRHS();
    void setThresholdFromLHSValue(double _value);
    void setThresholdFromSamples(int _samples);
    void applyThreshold();
	double computeRHS(double _Gvalue);
	void setVectorLHS();
	double computeLHS(double _Mg, double _M0);
	double computeWaveVectorK(double gradientMagnitude, double pulse_width, double giromagneticRatio);
	double ** getSamplesMagnitude();
	double ** computeSamplesMagnitude();
	double ** computeSamplesMagnitudeWithOmp();
	double ** computeSamplesMagnitudeWithGpu();
	void computeMktSmallPopulation(double **Mkt_samples, bool time_verbose);
	void computeMktSmallPopulation2(double **Mkt_samples, bool time_verbose);
	void computeMktSmallSamples(double **Mkt_samples, bool time_verbose);
	void computeMktBigSamples(double **Mkt_samples, bool time_verbose);
	void set();
	void run();
	void runSequence();
	void runSequenceWithoutSampling();
	void runSequenceWithSampling();
	void simulation();
	void applyBulk();
	void recoverD(string _method = "sat");
	void recoverDsat();
	void recoverDsatWithoutSampling();
	void recoverDsatWithSampling();
	void recoverDmsd();
	void recoverDmsdWithoutSampling();
	void recoverDmsdWithSampling();	
	void clear();
	void resetNMR();
	void updateWalkersXIrate(uint _rwsteps);
	void reset(double _newBigDelta);
	void reset();
	void save();
	void writeResults();
	void writeParameters();
	void writeEchoes();
	void writeGvector();
	void writeMsd();
	void setName();
	void createDirectoryForData();

	// Inline methods
	void setExposureTime(double _value){ this->exposureTime = _value; }
	void setPulseWidth(double _value){ this->pulseWidth = _value; }
	void setGiromagneticRatio(double _value){ this->giromagneticRatio = _value; }
	void setApplyBulkRelaxation(bool _bulk) { this->applyBulkRelaxation = _bulk; }
	void setD_sat(double _value) { this->D_sat = _value; }
	void setD_sat_StdDev(double _value) { this->D_sat_stdev = _value; }
	void setD_msd(double _value) { this->D_msd = _value; }
	void setD_msd_StdDev(double _value) { this->D_msd_stdev = _value; }
	void setNoiseAmp(double _amp) { this->noiseAmp = _amp; }
	
	void setMsd(double _value) { this->msd = _value; }
	void setMsdStdDev(double _value) { this->msd_stdev = _value; }	
	void setVecMsd(double msdX, double msdY, double msdZ) 
	{
		this->vecMsd.setX(msdX);
		this->vecMsd.setY(msdY);
		this->vecMsd.setZ(msdZ);
		this->vecMsd.setNorm();
	}
	void setVecDmsd(double DmsdX, double DmsdY, double DmsdZ) 
	{
		this->vecDmsd.setX(DmsdX);
		this->vecDmsd.setY(DmsdY);
		this->vecDmsd.setZ(DmsdZ);
		this->vecDmsd.setNorm();
	}
	void setVecMsdStdDev(double msdX_stdev, double msdY_stdev, double msdZ_stdev) 
	{
		this->vecMsd_stdev.setX(msdX_stdev);
		this->vecMsd_stdev.setY(msdY_stdev);
		this->vecMsd_stdev.setZ(msdZ_stdev);
		this->vecMsd_stdev.setNorm();
	}
	void setVecDmsdStdDev(double DmsdX_stdev, double DmsdY_stdev, double DmsdZ_stdev) 
	{
		this->vecDmsd_stdev.setX(DmsdX_stdev);
		this->vecDmsd_stdev.setY(DmsdY_stdev);
		this->vecDmsd_stdev.setZ(DmsdZ_stdev);
		this->vecDmsd_stdev.setNorm();
	}

	void resetCurrentTime() { this->currentTime = 0; }
	void incrementCurrentTime() { this->currentTime++; }

	int getGradientPoints() { return this->gradientPoints; }
	double getExposureTime() {return this->exposureTime; }
	double getExposureTime(uint _idx) {return this->exposureTimes[_idx]; }
	double getPulseWidth() {return this->pulseWidth; }
	double getGiromagneticRatio() {return this->giromagneticRatio; }
	bool getApplyBulkRelaxation() { return this->applyBulkRelaxation; }
	double getNoiseAmp() { return this->noiseAmp; }
	double getD_sat() { return this->D_sat; }
	double getD_sat_stdev() { return this->D_sat_stdev; }
	double getD_msd() { return this->D_msd; }
	double getD_msd_stdev() { return this->D_msd_stdev; }
	double getMsd() { return this->msd; }
	double getMsd_stdev() { return this->msd_stdev; }
	Vector3D getVecMsd() { return this->vecMsd; }
	Vector3D getVecMsd_stdev() { return this->vecMsd_stdev; }
	Vector3D getVecDmsd() { return this->vecDmsd; }
	Vector3D getVecDmsd_stdev() { return this->vecDmsd_stdev; }
	int getCurrentTime() { return this->currentTime; }
	

private:
	int mpi_rank;
	int mpi_processes;

	void simulation_cuda();
	void simulation_omp();
	double mean(vector<double> &_vec);
	double mean(double *_vec, int _size);
	double stdDev(vector<double> &_vec);
	double stdDev(vector<double> &_vec, double mean);
	double stdDev(double *_vec, int _size);
	double stdDev(double *_vec, int _size, double mean);
	vector<double> getNormalDistributionSamples(const double loc, const double std, const int size);
};

#endif