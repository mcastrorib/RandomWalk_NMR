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

	double gradient_X;
	double gradient_Y;
	double gradient_Z;
	vector<Vector3D> vecGradient;

	int gradientPoints;

	double exposureTime;
	double pulseWidth;
	double giromagneticRatio;

	vector<double> LHS;
	vector<double> RHS;
	double M0;
	double RHS_threshold;
	double diffusionCoefficient;

	NMR_PFGSE(NMR_Simulation &_NMR, 
			  pfgse_config _pfgseConfig,
			  uint time_sample, 
			  int _mpi_rank = 0, 
			  int _mpi_processes = 0);	

	NMR_PFGSE(NMR_Simulation &_NMR,  
			  Vector3D gradient_max,
			  int gradientPoints,
			  double _bigDelta,
			  double _pulseWidth , 
			  double _giromagneticRatio, 
			  int _mpi_rank = 0, 
			  int _mpi_processes = 0);
	
	virtual ~NMR_PFGSE(){};

	void setNMRTimeFramework();
	void setGradientVector(double _GF, int _GPoints);
	void setGradientVector_old();
	void setGradientVector();
	void setVectorRHS();
	void setThresholdFromRHSValue(double _value);
    void setThresholdFromLHSValue(double _value);
    void setThresholdFromFraction(double _fraction);
    void setThresholdFromSamples(int _samples);
	double computeRHS(double _Gvalue);
	void setVectorLHS();
	double computeLHS(double _Mg, double _M0);
	void set_old();
	void set();
	void run_old();
	void run();
	void simulation_old();
	void simulation();
	void recover_D(string _method = "stejskal");
	void recover_Stejskal();
	void recover_meanSquaredDisplacement();
	void clear();
	void reset(double _newBigDelta);
	void reset();
	void save();
	void writeResults();
	void setName();
	void createDirectoryForData();

	void setExposureTime(double _value){ this->exposureTime = _value; }
	void setPulseWidth(double _value){ this->pulseWidth = _value; }
	void setGiromagneticRatio(double _value){ this->giromagneticRatio = _value; }
	void setDiffusionCoefficient(double _value){ this->diffusionCoefficient = _value; }

	double getExposureTime() {return this->exposureTime; }
	double getPulseWidth() {return this->pulseWidth; }
	double getGiromagneticRatio() {return this->giromagneticRatio; }
	double getDiffusionCoefficient() {return this->diffusionCoefficient; }

private:
	int mpi_rank;
	int mpi_processes;

	void simulation_cuda_old();
	void simulation_cuda();
	void simulation_omp();
};

#endif