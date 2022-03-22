// include C++ standard libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>

#include "multitau_config.h"

using namespace std;

// default constructors
multitau_config::multitau_config(const string configFile) : config_filepath(configFile)
{
    string default_dirpath = CONFIG_ROOT;
    string default_filename = CPMG_CONFIG_DEFAULT;
    (*this).readConfigFile(default_dirpath + default_filename);
	if(configFile != (default_dirpath + default_filename)) (*this).readConfigFile(configFile);
}

//copy constructors
multitau_config::multitau_config(const multitau_config &otherConfig) 
{
    this->config_filepath = otherConfig.config_filepath;

    // --- Physical attributes.
    this->TAU_MIN = otherConfig.TAU_MIN;
    this->TAU_MAX = otherConfig.TAU_MAX;
    this->TAU_POINTS = otherConfig.TAU_POINTS;
    this->TAU_SCALE = otherConfig.TAU_SCALE;

    // --- cpmg SAVE. 
    this->SAVE_MODE = otherConfig.SAVE_MODE;
    this->SAVE_DECAY = otherConfig.SAVE_DECAY;
    this->SAVE_WALKERS = otherConfig.SAVE_WALKERS;
    this->SAVE_HISTOGRAM = otherConfig.SAVE_HISTOGRAM;
    this->SAVE_HISTOGRAM_LIST = otherConfig.SAVE_HISTOGRAM_LIST;
}

// read config file
void multitau_config::readConfigFile(const string configFile)
{
	ifstream fileObject;
    fileObject.open(configFile, ios::in);
    if (fileObject.fail())
    {
        cout << "Could not open cpmg config file from disc." << endl;
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

			if(token == "TAU_MIN") (*this).readTauMin(content);
            else if(token == "TAU_MAX") (*this).readTauMax(content);
            else if(token == "TAU_POINTS") (*this).readTauPoints(content);
            else if(token == "TAU_SCALE") (*this).readTauScale(content);
            else if(token == "SAVE_MODE") (*this).readSaveMode(content);
            else if(token == "SAVE_DECAY") (*this).readSaveDecay(content);
            else if(token == "SAVE_WALKERS") (*this).readSaveWalkers(content);
            else if(token == "SAVE_HISTOGRAM") (*this).readSaveHistogram(content);
            else if(token == "SAVE_HISTOGRAM_LIST") (*this).readSaveHistogramList(content); 
			
		}
    } 

    fileObject.close();
}

void multitau_config::readTauMin(string s)
{
	this->TAU_MIN = std::stod(s);
}

void multitau_config::readTauMax(string s)
{
    this->TAU_MAX = std::stod(s);
}

void multitau_config::readTauPoints(string s)
{
    this->TAU_POINTS = std::stoi(s);
}

void multitau_config::readTauScale(string s)
{
    if(s == "log") this->TAU_SCALE = s;
    else this->TAU_SCALE = "linear";
}

void multitau_config::readSaveMode(string s)
{
    if(s == "true") this->SAVE_MODE = true;
    else this->SAVE_MODE = false;
}

void multitau_config::readSaveWalkers(string s)
{
    if(s == "true") this->SAVE_WALKERS = true;
    else this->SAVE_WALKERS = false;
}

void multitau_config::readSaveDecay(string s)
{
    if(s == "true") this->SAVE_DECAY = true;
    else this->SAVE_DECAY = false;
}

void multitau_config::readSaveHistogram(string s)
{
    if(s == "true") this->SAVE_HISTOGRAM = true;
    else this->SAVE_HISTOGRAM = false;
}

void multitau_config::readSaveHistogramList(string s)
{
    if(s == "true") this->SAVE_HISTOGRAM_LIST = true;
    else this->SAVE_HISTOGRAM_LIST = false;
}
