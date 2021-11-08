#ifndef NMR_CPMG_H
#define NMR_CPMG_H

#include <vector>
#include <string>

// include configuration file classes
#include "../ConfigFiles/cpmg_config.h"

#include "NMR_defs.h"
#include "NMR_Simulation.h"
#include "../Walker/walker.h"
#include "../Math/Vector3D.h"


using namespace std;

class NMR_cpmg
{
public:
	NMR_Simulation &NMR;
	cpmg_config CPMG_config;
	string name;
	string dir;
    double exposureTime;
    bool applyBulkRelaxation;
    string method;
    vector<double> T2_bins;
    vector<double> T2_amps;
    vector<double> noise;
    


	NMR_cpmg(NMR_Simulation &_NMR, 
			  cpmg_config _pfgseConfig,
			  int _mpi_rank = 0, 
			  int _mpi_processes = 0);	

	virtual ~NMR_cpmg(){};

    // -- Essentials
	void setNMRTimeFramework();
    void set();
    void run();
    void applyBulk();
    void applyLaplace();
    void setNoise(vector<double> _rawNoise) { this->noise = _rawNoise; }
	void save();
	void writeResults();
	void saveT2decay();
	void saveT2dist();
	void setName();
	void createDirectoryForData();

    // -- Set methods
	void setExposureTime(double _value){ this->exposureTime = _value; }
	void setApplyBulkRelaxation(bool _bulk){ this->applyBulkRelaxation = _bulk; }
    void setMethod(string _method){ this->method = _method; }
	
    // -- Get methods
	double getExposureTime() { return this->exposureTime; }
	bool getApplyBulkRelaxation(){ return this->applyBulkRelaxation; }
    string getMethod() { return this->method; }

    // -- Simulations
    void run_simulation();
    void image_simulation_cuda();
	void image_simulation_omp();
    void histogram_simulation();
	

private:
	int mpi_rank;
	int mpi_processes;
};

#endif