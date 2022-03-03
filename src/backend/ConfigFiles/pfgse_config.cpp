// include C++ standard libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>
#include <stdint.h>

#include "pfgse_config.h"

using namespace std;

// default constructors
pfgse_config::pfgse_config(const string configFile) : config_filepath(configFile), THRESHOLD_WINDOW(1), NOISE_AMP(0.0), TARGET_SNR(-1.0)
{
    vector<double> TIME_VALUES();

    string default_dirpath = CONFIG_ROOT;
    string default_filename = PFGSE_CONFIG_DEFAULT;
    (*this).readConfigFile(default_dirpath + default_filename);
	(*this).readConfigFile(configFile);
    (*this).createTimeSamples();
}

//copy constructors
pfgse_config::pfgse_config(const pfgse_config &otherConfig)
{
    this->config_filepath = otherConfig.config_filepath;
    // --- Physical attributes.
    this->GIROMAGNETIC_RATIO = otherConfig.GIROMAGNETIC_RATIO;
    this->D0 = otherConfig.D0;
    this->APPLY_BULK = otherConfig.APPLY_BULK;
    this->PULSE_WIDTH = otherConfig.PULSE_WIDTH;
    this->MAX_GRADIENT = otherConfig.MAX_GRADIENT;
    this->GRADIENT_SAMPLES = otherConfig.GRADIENT_SAMPLES;

    // --- Time sequence 
    this->TIME_SEQ = otherConfig.TIME_SEQ;
    this->TIME_SAMPLES = otherConfig.TIME_SAMPLES;
    this->TIME_VALUES = otherConfig.TIME_VALUES;
    this->TIME_MIN = otherConfig.TIME_MIN;
    this->TIME_MAX = otherConfig.TIME_MAX;
    this->APPLY_SCALE_FACTOR = otherConfig.APPLY_SCALE_FACTOR;
    this->INSPECTION_LENGTH = otherConfig.INSPECTION_LENGTH;

    // --- Threshold application for D(t) recovering.
    this->NOISE_AMP = otherConfig.NOISE_AMP;
    this->TARGET_SNR = otherConfig.TARGET_SNR;
    this->THRESHOLD_TYPE = otherConfig.THRESHOLD_TYPE;
    this->THRESHOLD_VALUE = otherConfig.THRESHOLD_VALUE;
    this->THRESHOLD_WINDOW = otherConfig.THRESHOLD_WINDOW;

    // --- Wave-vector 'k' computation.
    this->USE_WAVEVECTOR_TWOPI = otherConfig.USE_WAVEVECTOR_TWOPI;
    this->ALLOW_WALKER_SAMPLING = otherConfig.ALLOW_WALKER_SAMPLING;
    this->APPLY_ABSORPTION = otherConfig.APPLY_ABSORPTION;

    // --- PFGSE SAVE. 
    this->SAVE_MODE = otherConfig.SAVE_MODE;
    this->SAVE_PFGSE = otherConfig.SAVE_PFGSE;
    this->SAVE_WALKERS = otherConfig.SAVE_WALKERS;
    this->SAVE_HISTOGRAM = otherConfig.SAVE_HISTOGRAM;
    this->SAVE_HISTOGRAM_LIST = otherConfig.SAVE_HISTOGRAM_LIST;
}

// read config file
void pfgse_config::readConfigFile(const string configFile)
{
    ifstream fileObject;
    fileObject.open(configFile, ios::in);
    if (fileObject.fail())
    {
        cout << "Could not open pfgse config file from disc." << endl;
        exit(1);
    }

    string line;
    while(fileObject)
    {
    	getline(fileObject, line);

    	string s = line;
    	string delimiter = ": ";
		size_t pos = 0;
		string token, content;
    	while ((pos = s.find(delimiter)) != std::string::npos) 
    	{
			token = s.substr(0, pos);
			content = s.substr(pos + delimiter.length(), s.length());
			s.erase(0, pos + delimiter.length());

			if(token == "GIROMAGNETIC_RATIO")	(*this).readGiromagneticRatio(content);
            else if(token == "D0") (*this).readD0(content);
            else if(token == "APPLY_BULK") (*this).readApplyBulk(content);
			else if(token == "PULSE_WIDTH") (*this).readPulseWidth(content);
            else if(token == "MAX_GRADIENT") (*this).readMaxGradient(content);
            else if(token == "GRADIENT_SAMPLES") (*this).readGradientSamples(content);
            else if(token == "TIME_SEQ") (*this).readTimeSequence(content);
            else if(token == "TIME_SAMPLES") (*this).readTimeSamples(content);
            else if(token == "TIME_VALUES") (*this).readTimeValues(content);
            else if(token == "TIME_MIN") (*this).readTimeMin(content);
            else if(token == "TIME_MAX") (*this).readTimeMax(content);
            else if(token == "APPLY_SCALE_FACTOR") (*this).readApplyScaleFactor(content);
            else if(token == "NOISE_AMP") (*this).readNoiseAmp(content);
            else if(token == "TARGET_SNR") (*this).readTargetSNR(content);
            else if(token == "INSPECTION_LENGTH") (*this).readInspectionLength(content);
            else if(token == "THRESHOLD_TYPE") (*this).readThresholdType(content);
            else if(token == "THRESHOLD_VALUE") (*this).readThresholdValue(content);
            else if(token == "THRESHOLD_WINDOW") (*this).readThresholdWindow(content);
            else if(token == "USE_WAVEVECTOR_TWOPI") (*this).readUseWaveVectorTwoPi(content);
            else if(token == "ALLOW_WALKER_SAMPLING") (*this).readAllowWalkerSampling(content);
            else if(token == "APPLY_ABSORPTION") (*this).readApplyAbsorption(content);
            else if(token == "SAVE_MODE") (*this).readSaveMode(content);
            else if(token == "SAVE_PFGSE") (*this).readSavePFGSE(content);
            else if(token == "SAVE_WALKERS") (*this).readSaveWalkers(content);
            else if(token == "SAVE_HISTOGRAM") (*this).readSaveHistogram(content);
            else if(token == "SAVE_HISTOGRAM_LIST") (*this).readSaveHistogramList(content); 			
		}
    } 

    fileObject.close();
}

void pfgse_config::readGiromagneticRatio(string s)
{
	this->GIROMAGNETIC_RATIO = std::stod(s);
}

void pfgse_config::readD0(string s)
{
	this->D0 = std::stod(s);
}

void pfgse_config::readApplyBulk(string s)
{
    if(s == "true") this->APPLY_BULK = true;
    else this->APPLY_BULK = false;
}

void pfgse_config::readPulseWidth(string s)
{
	this->PULSE_WIDTH = std::stod(s);
}

void pfgse_config::readMaxGradient(string s)
{
    vector<double> buffer;
    if(s.compare(0, 1, "{") == 0 and s.compare(s.length() - 1, 1, "}") == 0)
    {
        string strvec = s.substr(1, s.length() - 2);
        string delimiter = ",";
        size_t pos = 0;
        string token, content;
        while ((pos = strvec.find(delimiter)) != std::string::npos) 
        {
            token = strvec.substr(0, pos);
            content = strvec.substr(pos + delimiter.length(), strvec.length());
            strvec.erase(0, pos + delimiter.length());

            // add value to RHO attribute
            buffer.push_back(std::stod(token));
        }
        // add value to RHO attribute
        buffer.push_back(std::stod(strvec));
    }

    if(buffer.size() >= 3)
    {
        this->MAX_GRADIENT.setX(buffer[0]);
        this->MAX_GRADIENT.setY(buffer[1]);
        this->MAX_GRADIENT.setZ(buffer[2]);
        this->MAX_GRADIENT.setNorm();
    }
}

void pfgse_config::readGradientSamples(string s)
{
    this->GRADIENT_SAMPLES = std::stoi(s);
}

void pfgse_config::readTimeSequence(string s)
{
    this->TIME_SEQ = s;
}

void pfgse_config::readTimeSamples(string s)
{
    this->TIME_SAMPLES = std::stoi(s);
}

void pfgse_config::readTimeValues(string s)
{
    if(this->TIME_VALUES.size() > 0) this->TIME_VALUES.clear();
    if(this->TIME_SEQ == "manual")
    {
        // parse vector
		if(s.compare(0, 1, "{") == 0 and s.compare(s.length() - 1, 1, "}") == 0)
		{
			string strvec = s.substr(1, s.length() - 2);
			string delimiter = ",";
			size_t pos = 0;
			string token, content;
			while ((pos = strvec.find(delimiter)) != std::string::npos) 
	    	{
				token = strvec.substr(0, pos);
				content = strvec.substr(pos + delimiter.length(), strvec.length());
				strvec.erase(0, pos + delimiter.length());

				// add value to RHO attribute
				this->TIME_VALUES.push_back(std::stod(token));
			}
			// add value to RHO attribute
			this->TIME_VALUES.push_back(std::stod(strvec));
		}
    } 
}

void pfgse_config::createTimeSamples()
{
    if(this->TIME_SEQ == "log")
    {
        // set logspace vector
        this->TIME_VALUES = (*this).logspace(this->TIME_MIN, this->TIME_MAX, this->TIME_SAMPLES);        
    } 
    else if(this->TIME_SEQ == "linear")
    {
        // set linspace vector
        this->TIME_VALUES = (*this).linspace(this->TIME_MIN, this->TIME_MAX, this->TIME_SAMPLES);
        
    } 
    else if(this->TIME_SEQ == "manual")
    {
        // Sort island individuals by fitness
        sort(this->TIME_VALUES.begin(), 
             this->TIME_VALUES.end(), 
             [](double const &a, double &b) 
             { return a < b; });
    }    
}


void pfgse_config::readTimeMin(string s)
{
    this->TIME_MIN = std::stod(s);
}

void pfgse_config::readTimeMax(string s)
{
    this->TIME_MAX = std::stod(s);
}

void pfgse_config::readApplyScaleFactor(string s)
{
    if(s == "true") this->APPLY_SCALE_FACTOR = true;
    else this->APPLY_SCALE_FACTOR = false;
}

void pfgse_config::readInspectionLength(string s)
{
    this->INSPECTION_LENGTH = std::stod(s);
}

void pfgse_config::readNoiseAmp(string s)
{
    double amp = std::stod(s);
    if(amp > 0.0) this->NOISE_AMP = amp;
}

void pfgse_config::readTargetSNR(string s)
{
    double snr = std::stod(s);
    if(snr > 0.0) this->TARGET_SNR = snr;
}

void pfgse_config::readThresholdType(string s)
{
    this->THRESHOLD_TYPE = s;
}

void pfgse_config::readThresholdValue(string s)
{
    this->THRESHOLD_VALUE = std::stod(s);
}

void pfgse_config::readThresholdWindow(string s)
{
    this->THRESHOLD_WINDOW = std::stoi(s);
}

void pfgse_config::readUseWaveVectorTwoPi(string s)
{
    if(s == "true") this->USE_WAVEVECTOR_TWOPI = true;
    else this->USE_WAVEVECTOR_TWOPI = false;
}

void pfgse_config::readAllowWalkerSampling(string s)
{
    if(s == "true") this->ALLOW_WALKER_SAMPLING = true;
    else this->ALLOW_WALKER_SAMPLING = false;
}

void pfgse_config::readApplyAbsorption(string s)
{
    if(s == "true") this->APPLY_ABSORPTION = true;
    else this->APPLY_ABSORPTION = false;
}

void pfgse_config::readSaveMode(string s)
{
    if(s == "true") this->SAVE_MODE = true;
    else this->SAVE_MODE = false;
}

void pfgse_config::readSavePFGSE(string s)
{
    if(s == "true") this->SAVE_PFGSE = true;
    else this->SAVE_PFGSE = false;
}

void pfgse_config::readSaveWalkers(string s)
{
    if(s == "true") this->SAVE_WALKERS = true;
    else this->SAVE_WALKERS = false;
}

void pfgse_config::readSaveHistogram(string s)
{
    if(s == "true") this->SAVE_HISTOGRAM = true;
    else this->SAVE_HISTOGRAM = false;
}

void pfgse_config::readSaveHistogramList(string s)
{
    if(s == "true") this->SAVE_HISTOGRAM_LIST = true;
    else this->SAVE_HISTOGRAM_LIST = false;
}

// Returns a vector<double> linearly space from @start to @end with @points
vector<double> pfgse_config::linspace(double start, double end, uint points)
{
    vector<double> vec(points);
    double step = (end - start) / ((double) points - 1.0);
    
    for(int idx = 0; idx < points; idx++)
    {
        double x_i = start + step * idx;
        vec[idx] = x_i;
    }

    return vec;
}

// Returns a vector<double> logarithmly space from 10^@exp_start to 10^@end with @points
vector<double> pfgse_config::logspace(double exp_start, double exp_end, uint points, double base)
{
    vector<double> vec(points);
    double step = (exp_end - exp_start) / ((double) points - 1.0);
    
    for(int idx = 0; idx < points; idx++)
    {
        double x_i = exp_start + step * idx;
        vec[idx] = pow(base, x_i);
    }

    return vec;
}
