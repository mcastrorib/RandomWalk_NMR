#ifndef NMR_CPMG_H
#define NMR_CPMG_H

#include <vector>
#include <string>

// include configuration file classes
#include "../ConfigFiles/cpmg_config.h"

#include "NMR_defs.h"
#include "NMR_Simulation.h"
#include "InternalField.h"
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
    vector<double> signal_amps;
    vector<double> signal_times;
    vector<double> T2_bins;
    vector<double> T2_amps;
    vector<double> noise;
    double *penalties;
    InternalField *internalField;
    


	NMR_cpmg(NMR_Simulation &_NMR, 
			  cpmg_config _pfgseConfig,
			  int _mpi_rank = 0, 
			  int _mpi_processes = 0);	

	virtual ~NMR_cpmg()
	{
		if(this->penalties != NULL)
        {
            delete[] this->penalties;
            this->penalties = NULL;
        }

        if(this->internalField != NULL)
        {
        	delete internalField;
        	this->internalField = NULL;
        }
	};

    // -- Essentials
	void setNMRTimeFramework();
    void set();
    void run();
    void resetSignal();
    void normalizeSignal();
    void applyBulk();
    void applyLaplace();
    void setNoise(vector<double> _rawNoise) { this->noise = _rawNoise; }
	void save();
	void writeWalkers();
	void writeHistogram();
	void writeHistogramList();
	void writeT2decay();
	void writeT2dist();
	void setName();
	void createDirectoryForData();

    // -- Set methods
	void setExposureTime(double _value){ this->exposureTime = _value; }
	void setApplyBulkRelaxation(bool _bulk){ this->applyBulkRelaxation = _bulk; }
    void setMethod(string _method){ this->method = _method; }
    void setInternalField(string _mode);
	
    // -- Get methods
	double getExposureTime() { return this->exposureTime; }
	bool getApplyBulkRelaxation(){ return this->applyBulkRelaxation; }
    string getMethod() { return this->method; }
    vector<double> getSignalAmps() { return this->signal_amps; }
    vector<double> getSignalTimes() { return this->signal_times; }
    InternalField *getInternalField() {return this->internalField; }
    double *getInternalFieldData() { return (this->internalField == NULL) ? NULL : this->internalField->getData(); }
    double getInternalFieldSize() { return (this->internalField == NULL) ? 0 : this->internalField->getSize(); }


    // -- Simulations
    void run_simulation();
    void image_simulation_cuda();
	void image_simulation_omp();
	void createPenaltiesVector(vector<double> &_sigmoid);
    void createPenaltiesVector(double rho);
    void histogram_simulation();
	

private:
	int mpi_rank;
	int mpi_processes;
};

#endif