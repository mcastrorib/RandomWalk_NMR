// include C++ standard libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>

#include "cpmg_config.h"

using namespace std;

// default constructors
cpmg_config::cpmg_config(const string configFile) : config_filepath(configFile)
{
    vector<double> TIME_VALUES();
	(*this).readConfigFile(configFile);
}

//copy constructors
cpmg_config::cpmg_config(const cpmg_config &otherConfig) 
{
    this->config_filepath = otherConfig.config_filepath;
    // --- Physical attributes.
    this->D0 = otherConfig.D0;
    this->OBS_TIME = otherConfig.OBS_TIME;
    this->METHOD = otherConfig.METHOD;

    this->MIN_T2 = otherConfig.MIN_T2;
    this->MAX_T2 = otherConfig.MAX_T2;
    this->USE_T2_LOGSPACE = otherConfig.USE_T2_LOGSPACE;
    this->NUM_T2_BINS = otherConfig.NUM_T2_BINS;
    this->MIN_LAMBDA = otherConfig.MIN_LAMBDA;
    this->MAX_LAMBDA = otherConfig.MAX_LAMBDA;
    this->NUM_LAMBDAS = otherConfig.NUM_LAMBDAS;
    this->PRUNE_NUM = otherConfig.PRUNE_NUM;
    this->NOISE_AMP = otherConfig.NOISE_AMP;

    // --- cpmg SAVE. 
    this->SAVE_MODE = otherConfig.SAVE_MODE;
    this->SAVE_T2 = otherConfig.SAVE_T2;
    this->SAVE_COLLISIONS = otherConfig.SAVE_COLLISIONS;
    this->SAVE_DECAY = otherConfig.SAVE_DECAY;
    this->SAVE_HISTOGRAM = otherConfig.SAVE_HISTOGRAM;
    this->SAVE_HISTOGRAM_LIST = otherConfig.SAVE_HISTOGRAM_LIST;
}

// read config file
void cpmg_config::readConfigFile(const string configFile)
{
	cout << "reading cpmg configs from file...";

    ifstream fileObject;
    fileObject.open(configFile, ios::in);
    if (fileObject.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    string line;
    while(fileObject)
    {
    	getline(fileObject, line);
    	// cout << line << endl;

    	string s = line;
    	string delimiter = ": ";
		size_t pos = 0;
		string token, content;
    	while ((pos = s.find(delimiter)) != std::string::npos) 
    	{
			token = s.substr(0, pos);
			content = s.substr(pos + delimiter.length(), s.length());
			s.erase(0, pos + delimiter.length());

			if(token == "D0") (*this).readD0(content);
			else if(token == "OBS_TIME") (*this).readObservationTime(content);  
            else if(token == "METHOD") (*this).readMethod(content);            
            else if(token == "MIN_T2") (*this).readMinT2(content);
            else if(token == "MAX_T2") (*this).readMaxT2(content);
            else if(token == "USE_T2_LOGSPACE") (*this).readUseT2Logspace(content);
            else if(token == "NUM_T2_BINS") (*this).readNumT2Bins(content);
            else if(token == "MIN_LAMBDA") (*this).readMinLambda(content);
            else if(token == "MAX_LAMBDA") (*this).readMaxLambda(content);
            else if(token == "NUM_LAMBDAS") (*this).readNumLambdas(content);
            else if(token == "PRUNE_NUM") (*this).readPruneNum(content);
            else if(token == "NOISE_AMP") (*this).readNoiseAmp(content);                      
            else if(token == "SAVE_MODE") (*this).readSaveMode(content);
            else if(token == "SAVE_T2") (*this).readSaveT2(content);
            else if(token == "SAVE_COLLISIONS") (*this).readSaveCollisions(content);
            else if(token == "SAVE_DECAY") (*this).readSaveDecay(content);
            else if(token == "SAVE_HISTOGRAM") (*this).readSaveHistogram(content);
            else if(token == "SAVE_HISTOGRAM_LIST") (*this).readSaveHistogramList(content); 
			
		}
    } 

    cout << "Ok" << endl;
    fileObject.close();
}

void cpmg_config::readD0(string s)
{
	this->D0 = std::stod(s);
}

void cpmg_config::readObservationTime(string s)
{
    this->OBS_TIME = std::stod(s);
}

void cpmg_config::readMethod(string s)
{
    this->METHOD = s;
}

void cpmg_config::readMinT2(string s)
{
    this->MIN_T2 = std::stod(s);
}

void cpmg_config::readMaxT2(string s)
{
    this->MAX_T2 = std::stod(s);
}

void cpmg_config::readUseT2Logspace(string s)
{
    if(s == "true") this->USE_T2_LOGSPACE = true;
    else this->USE_T2_LOGSPACE = false;
}

void cpmg_config::readNumT2Bins(string s)
{
    this->NUM_T2_BINS = std::stoi(s);
}

void cpmg_config::readMinLambda(string s)
{
    this->MIN_LAMBDA = std::stod(s);
}

void cpmg_config::readMaxLambda(string s)
{
    this->MAX_LAMBDA = std::stod(s);
}

void cpmg_config::readNumLambdas(string s)
{
    this->NUM_LAMBDAS = std::stoi(s);
}

void cpmg_config::readPruneNum(string s)
{
    this->PRUNE_NUM = std::stoi(s);
}

void cpmg_config::readNoiseAmp(string s)
{
    this->NOISE_AMP = std::stod(s);
}

void cpmg_config::readSaveMode(string s)
{
    if(s == "true") this->SAVE_MODE = true;
    else this->SAVE_MODE = false;
}

void cpmg_config::readSaveCollisions(string s)
{
    if(s == "true") this->SAVE_COLLISIONS = true;
    else this->SAVE_COLLISIONS = false;
}

void cpmg_config::readSaveDecay(string s)
{
    if(s == "true") this->SAVE_DECAY = true;
    else this->SAVE_DECAY = false;
}

void cpmg_config::readSaveHistogram(string s)
{
    if(s == "true") this->SAVE_HISTOGRAM = true;
    else this->SAVE_HISTOGRAM = false;
}

void cpmg_config::readSaveHistogramList(string s)
{
    if(s == "true") this->SAVE_HISTOGRAM_LIST = true;
    else this->SAVE_HISTOGRAM_LIST = false;
}

void cpmg_config::readSaveT2(string s)
{
    if(s == "true") this->SAVE_T2 = true;
    else this->SAVE_T2 = false;
}
