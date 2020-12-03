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
    string method;
    vector<double> T2_bins;
    vector<double> T2_amps;
    


	NMR_cpmg(NMR_Simulation &_NMR, 
			  cpmg_config _pfgseConfig,
			  int _mpi_rank = 0, 
			  int _mpi_processes = 0);	

	virtual ~NMR_cpmg(){};

    // -- Essentials
	void setNMRTimeFramework();
    void set();
    void run();
    void applyLaplace();
	void save();
	void writeResults();
	void setName();
	void createDirectoryForData();

    // -- Set methods
	void setExposureTime(double _value){ this->exposureTime = _value; }
    void setMethod(string _method){ this->method = _method; }
	
    // -- Get methods
	double getExposureTime() {return this->exposureTime; }
    string getMethod() {return this->method;}

    // -- Simulations
    void run_simulation();
    void simulation_img_cuda();
	void simulation_img_omp();
    void simulation_histogram();
	

private:
	int mpi_rank;
	int mpi_processes;
};

#endif