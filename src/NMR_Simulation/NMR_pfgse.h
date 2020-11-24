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
	vector<double> exposureTimes;
	double exposureTime;
	double pulseWidth;
	double giromagneticRatio;

	vector<double> LHS;
	vector<double> RHS;
	vector<double> Mkt;
	double M0;
	double RHS_threshold;
	double D_sat;
	double D_msd;
	double SVp;

	NMR_PFGSE(NMR_Simulation &_NMR, 
			  pfgse_config _pfgseConfig,
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
	void setGradientVector();
	void setVectorMkt();
	void setVectorRHS();
	void setThresholdFromRHSValue(double _value);
    void setThresholdFromLHSValue(double _value);
    void setThresholdFromFraction(double _fraction);
    void setThresholdFromSamples(int _samples);
	double computeRHS(double _Gvalue);
	void setVectorLHS();
	double computeLHS(double _Mg, double _M0);
	double computeWaveVectorK(double gradientMagnitude, double pulse_width, double giromagneticRatio);
	void set();
	void run();
	void run_sequence();
	void simulation();
	void recoverD(string _method = "sat");
	void recoverD_sat();
	void recoverD_msd();
	void recoverSVp(string _method = "sat");
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
	void setD_sat(double _value) { this->D_sat = _value; }
	void setD_msd(double _value) { this->D_msd = _value; }
	void setSVp(double _value) { this->SVp = _value; }

	double getExposureTime() {return this->exposureTime; }
	double getExposureTime(uint _idx) {return this->exposureTimes[_idx]; }
	double getPulseWidth() {return this->pulseWidth; }
	double getGiromagneticRatio() {return this->giromagneticRatio; }
	double getD_sat() { return this->D_sat; }
	double getD_msd() { return this->D_msd; }
	double getSVp() { return this->SVp; }
	

private:
	int mpi_rank;
	int mpi_processes;

	void simulation_cuda();
	void simulation_omp();
};

#endif