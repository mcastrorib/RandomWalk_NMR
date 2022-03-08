#ifndef CPMG_CONFIG_H_
#define CPMG_CONFIG_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>
#include "configFiles_defs.h"

using namespace std;

class cpmg_config
{
public:
    string config_filepath;
    // --- Physical attributes.
    double D0;
    bool APPLY_BULK;
    double OBS_TIME;
    string METHOD;
    bool TIME_VERBOSE;

    // --- Diffusive relaxation.
    string RESIDUAL_FIELD;
    double PORE_FIELD;
    double MAT_FIELD;
    string PATH_TO_FIELD;

    // LAPLACE INVERSION
    double MIN_T2;
    double MAX_T2;
    bool USE_T2_LOGSPACE; 
    int NUM_T2_BINS;
    double MIN_LAMBDA;
    double MAX_LAMBDA;
    int NUM_LAMBDAS;
    int PRUNE_NUM;
    double NOISE_AMP;

    // --- cpmg SAVE. 
    bool SAVE_MODE;
    bool SAVE_DECAY;
    bool SAVE_T2;
    bool SAVE_WALKERS;
    bool SAVE_HISTOGRAM;
    bool SAVE_HISTOGRAM_LIST;
    


    // default constructors
    cpmg_config(){};
    cpmg_config(const string configFile);

    //copy constructors
    cpmg_config(const cpmg_config &otherConfig);

    // default destructor
    virtual ~cpmg_config()
    {
        // cout << "OMPLoopEnabler object destroyed succesfully" << endl;
    } 

    void readConfigFile(const string configFile);
    
    // -- Read methods
    void readD0(string s);
    void readApplyBulk(string s);
    void readTimeVerbose(string s);
    void readObservationTime(string s);
    void readMethod(string s);

    void readResidualField(string s);
    void readPoreField(string s);
    void readMatField(string s);
    void readPathToField(string s);


    void readMinT2(string s);
    void readMaxT2(string s);
    void readUseT2Logspace(string s);
    void readNumT2Bins(string s);
    void readMinLambda(string s);
    void readMaxLambda(string s);
    void readNumLambdas(string s);
    void readPruneNum(string s);
    void readNoiseAmp(string s);
    

    void readSaveMode(string s);
    void readSaveWalkers(string s);
    void readSaveDecay(string s);
    void readSaveHistogram(string s);
    void readSaveHistogramList(string s);   
    void readSaveT2(string s);   

    // -- Get methods
    string getConfigFilepath() {return this->config_filepath; }
    double getD0(){return this->D0; }
    bool getApplyBulk(){ return this->APPLY_BULK; }
    bool getTimeVerbose(){ return this->TIME_VERBOSE; }
    double getObservationTime(){return this->OBS_TIME; }
    string getMethod() { return this->METHOD; }

    string getResidualField() { return this->RESIDUAL_FIELD; }
    double getPoreField() { return this->PORE_FIELD; }
    double getMatField() { return this->MAT_FIELD; }
    string getPathToField() { return this->PATH_TO_FIELD; }

    double getMinT2(){ return this->MIN_T2; }
    double getMaxT2(){ return this->MAX_T2; }
    bool getUseT2Logspace(){ return this->USE_T2_LOGSPACE; }
    int getNumT2Bins(){ return this->NUM_T2_BINS; }
    double getMinLambda(){ return this->MIN_LAMBDA; }
    double getMaxLambda(){ return this->MAX_LAMBDA; }
    int getNumLambdas(){ return this->NUM_LAMBDAS; }
    int getPruneNum(){ return this->PRUNE_NUM; }
    double getNoiseAmp(){ return this->NOISE_AMP; }

    bool getSaveMode(){return this->SAVE_MODE; }
    bool getSaveWalkers(){return this->SAVE_WALKERS; }
    bool getSaveDecay() {return this->SAVE_DECAY; }
    bool getSaveHistogram() {return this->SAVE_HISTOGRAM; }
    bool getSaveHistogramList() {return this->SAVE_HISTOGRAM_LIST; }  
    bool getSaveT2() {return this->SAVE_T2; }  
};

#endif