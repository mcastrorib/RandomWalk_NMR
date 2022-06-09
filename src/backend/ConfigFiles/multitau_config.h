#ifndef MULTITAU_CONFIG_H_
#define MULTITAU_CONFIG_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>
#include "configFiles_defs.h"

using namespace std;

class multitau_config
{
public:
    string config_filepath;
    double TAU_MIN;
    double TAU_MAX;
    int TAU_POINTS;
    vector<double> TAU_VALUES;
    string TAU_SCALE;
    bool COMPLETE_DECAY;

    // --- cpmg SAVE. 
    bool SAVE_MODE;
    bool SAVE_DECAY;
    bool SAVE_WALKERS;
    bool SAVE_HISTOGRAM;
    bool SAVE_HISTOGRAM_LIST;
    
    // default constructors
    multitau_config(){};
    multitau_config(const string configFile);

    //copy constructors
    multitau_config(const multitau_config &otherConfig);

    // default destructor
    virtual ~multitau_config(){} 

    void readConfigFile(const string configFile);
    
    // -- Read methods
    void readTauMin(string s);
    void readTauMax(string s);
    void readTauPoints(string s);
    void readTauScale(string s);
    void readTauValues(string s);
    void readCompleteDecay(string s);
    void readSaveMode(string s);
    void readSaveDecay(string s);
    void readSaveWalkers(string s);
    void readSaveHistogram(string s);
    void readSaveHistogramList(string s);   
    
    // -- Get methods
    double getTauMin(){ return this->TAU_MIN; }
    double getTauMax(){ return this->TAU_MAX; }
    int getTauPoints(){ return this->TAU_POINTS; }
    vector<double> getTauValues(){ return this->TAU_VALUES; }
    string getTauScale(){ return this->TAU_SCALE; }
    bool getCompleteDecay(){ return this->COMPLETE_DECAY; }
    bool getSaveMode(){return this->SAVE_MODE; }
    bool getSaveDecay() {return this->SAVE_DECAY; }
    bool getSaveWalkers(){return this->SAVE_WALKERS; }
    bool getSaveHistogram() {return this->SAVE_HISTOGRAM; }
    bool getSaveHistogramList() {return this->SAVE_HISTOGRAM_LIST; }  
};

#endif