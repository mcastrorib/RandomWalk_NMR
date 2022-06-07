#ifndef NMR_MULTITAU_H
#define NMR_MULTITAU_H

#include <vector>
#include <string>

// include configuration file classes
#include "../ConfigFiles/multitau_config.h"
#include "../ConfigFiles/cpmg_config.h"
#include "NMR_defs.h"
#include "NMR_Simulation.h"
#include "NMR_cpmg.h"


using namespace std;

class NMR_multitau
{
public:
	NMR_Simulation &NMR;
	NMR_cpmg *cpmg;
	multitau_config MultiTau_config;
	cpmg_config CPMG_config;
	string name;
	string dir;
	vector<uint> requiredSteps;
    vector<double> signalTimes;
    vector<double> signalAmps;
    
	NMR_multitau(NMR_Simulation &_NMR, 
				 multitau_config _multitauConfig, 
				 cpmg_config _cpmgConfig, 
				 int _mpi_rank = 0, 
				 int _mpi_processes = 0);	

	virtual ~NMR_multitau()
	{
		if(this->cpmg != NULL)
		{
			delete this->cpmg;
        	this->cpmg = NULL;
		}
	};

	void setName();
	void createDirectoryForData();
	void setTauSequence();
	void setExposureTime(uint index);
	void setCPMG(uint index);
	void runCPMG();
	void saveCPMG();
	void run();
	void save();
	void writeDecay();
	void writeWalkers();
	void writeHistogram();
	void writeHistogramList();

private:
	int mpi_rank;
	int mpi_processes;

	// Returns a vector<double> linearly space from @start to @end with @points
    vector<double> linspace(double start, double end, uint points);

    // Returns a vector<double> logarithmly space from 10^@exp_start to 10^@end with @points
    vector<double> logspace(double exp_start, double exp_end, uint points, double base=10.0);

    // Returns the sum of elements of a vector
    int sum(vector<int> _vec);
    uint sum(vector<uint> _vec);
    double sum(vector<double> _vec);
};

#endif