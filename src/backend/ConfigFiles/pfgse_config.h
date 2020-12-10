#ifndef PFGSE_CONFIG_H_
#define PFGSE_CONFIG_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>

#include "../Math/Vector3D.h"

#ifndef CONFIG_ROOT
#define CONFIG_ROOT "/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/config/"
#endif

#ifndef PFGSE_CONFIG_DEFAULT
#define PFGSE_CONFIG_DEFAULT "./default/pfgse.config"
#endif

using namespace std;

class pfgse_config
{
public:
    string config_filepath;
    // --- Physical attributes.
    double GIROMAGNETIC_RATIO;
    double D0;
    double PULSE_WIDTH;
    Vector3D MAX_GRADIENT;
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
    pfgse_config(){};
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
    
    // -- Read methods
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

    // -- Get methods
    string getConfigFilepath() {return this->config_filepath; } 
    double getGiromagneticRatio() { return this->GIROMAGNETIC_RATIO ; }
    double getD0() { return this->D0 ; }
    double getPulseWidth() { return this->PULSE_WIDTH ; }
    Vector3D getMaxGradient() { return this->MAX_GRADIENT ; }
    uint getGradientSamples() { return this->GRADIENT_SAMPLES ; }
    string getTimeSequence() { return this->TIME_SEQ ; }
    uint getTimeSamples() { return this->TIME_SAMPLES ; } 
    vector<double> getTimeValues() { return this->TIME_VALUES ; }
    double getTimeMin() { return this->TIME_MIN; }
    double getTimeMax() { return this->TIME_MAX; }
    bool getApplyScaleFactor() { return this->APPLY_SCALE_FACTOR; }
    double getInspectionLength() { return this->INSPECTION_LENGTH; }
    string getThresholdType() { return this->THRESHOLD_TYPE; }
    double getThresholdValue() { return this->THRESHOLD_VALUE; }
    bool getUseWaveVectorTwoPi() { return this->USE_WAVEVECTOR_TWOPI; }
    bool getSaveMode() { return this->SAVE_MODE; }
    bool getSavePFGSE() { return this->SAVE_PFGSE; }
    bool getSaveCollisions() { return this->SAVE_COLLISIONS; }
    bool getSaveDecay() { return this->SAVE_DECAY; }
    bool getSaveHistogram() { return this->SAVE_HISTOGRAM; }
    bool getSaveHistogramList() { return this->SAVE_HISTOGRAM_LIST; }  

private:
    // Returns a vector<double> linearly space from @start to @end with @points
    vector<double> linspace(double start, double end, uint points);

    // Returns a vector<double> logarithmly space from 10^@exp_start to 10^@end with @points
    vector<double> logspace(double exp_start, double exp_end, uint points, double base=10.0);
    
};

#endif