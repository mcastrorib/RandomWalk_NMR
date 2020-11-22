#ifndef CPMG_CONFIG_H_
#define CPMG_CONFIG_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class cpmg_config
{
public:
    // --- Physical attributes.
    double D0;
    double OBS_TIME;

    // --- cpmg SAVE. 
    bool SAVE_MODE;
    bool SAVE_T2;
    bool SAVE_COLLISIONS;
    bool SAVE_DECAY;
    bool SAVE_HISTOGRAM;
    bool SAVE_HISTOGRAM_LIST;
    


    // default constructors
    cpmg_config(const string configFile);

    //copy constructors
    cpmg_config(const cpmg_config &otherConfig);

    // default destructor
    virtual ~cpmg_config()
    {
        // cout << "OMPLoopEnabler object destroyed succesfully" << endl;
    } 

    void readConfigFile(const string configFile);
    
    // -- 
    void readD0(string s);
    void readObservationTime(string s);
    void readSaveMode(string s);
    void readSaveCollisions(string s);
    void readSaveDecay(string s);
    void readSaveHistogram(string s);
    void readSaveHistogramList(string s);   
    void readSaveT2(string s);   
};

#endif