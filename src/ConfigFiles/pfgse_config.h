#ifndef PFGSE_CONFIG_H_
#define PFGSE_CONFIG_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class pfgse_config
{
public:
    // --- Physical attributes.
    double GIROMAGNETIC_RATIO;
    double D0;
    double PULSE_WIDTH;
    double MAX_GRADIENT;
    uint GRADIENT_SAMPLES;

    // --- Time sequence 
    string TIME_SEQ;
    uint TIME_SAMPLES;
    vector<double> TIME_VALUES;
    double TIME_MIN;
    double TIME_MAX;
    bool APPLY_SCALE_FACTOR;
    double INSPECTION_LENGTH;

    // --- Threshold application for D(t) recovering.
    string THRESHOLD_TYPE;
    double THRESHOLD_VALUE;

    // --- Wave-vector 'k' computation.
    bool USE_WAVEVECTOR_TWOPI;

    // --- PFGSE SAVE. 
    bool SAVE_MODE;
    bool SAVE_PFGSE;
    bool SAVE_COLLISIONS;
    bool SAVE_DECAY;
    bool SAVE_HISTOGRAM;
    bool SAVE_HISTOGRAM_LIST;


    // default constructors
    pfgse_config(const string configFile);

    //copy constructors
    pfgse_config(const pfgse_config &otherConfig);

    // default destructor
    virtual ~pfgse_config()
    {
        // cout << "OMPLoopEnabler object destroyed succesfully" << endl;
    } 

    void readConfigFile(const string configFile);
    void createTimeSamples();
    
    // -- 
    void readGiromagneticRatio(string s);
    void readD0(string s);
    void readPulseWidth(string s);
    void readMaxGradient(string s);
    void readGradientSamples(string s);
    void readTimeSequence(string s);
    void readTimeSamples(string s); 
    void readTimeValues(string s);
    void readTimeMin(string s);
    void readTimeMax(string s);
    void readApplyScaleFactor(string s);
    void readInspectionLength(string s);
    void readThresholdType(string s);
    void readThresholdValue(string s);
    void readUseWaveVectorTwoPi(string s);
    void readSaveMode(string s);
    void readSavePFGSE(string s);
    void readSaveCollisions(string s);
    void readSaveDecay(string s);
    void readSaveHistogram(string s);
    void readSaveHistogramList(string s);   

private:
    // Returns a vector<double> linearly space from @start to @end with @points
    vector<double> linspace(double start, double end, uint points);

    // Returns a vector<double> logarithmly space from 10^@exp_start to 10^@end with @points
    vector<double> logspace(double exp_start, double exp_end, uint points, double base=10.0);
    
};

#endif