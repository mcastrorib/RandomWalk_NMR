// include C++ standard libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <math.h>
#include <stdint.h>

#include "pfgse_config.h"

using namespace std;

// default constructors
pfgse_config::pfgse_config(const string configFile)
{
	(*this).readConfigFile(configFile);
}

//copy constructors
pfgse_config::pfgse_config(const pfgse_config &otherConfig)
{}

// read config file
void pfgse_config::readConfigFile(const string configFile)
{
	cout << "reading pfgse configs from file...";

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

			if(token == "GIROMAGNETIC_RATIO")	(*this).readGiromagneticRatio(content);
			else if(token == "PULSE_WIDTH") (*this).readPulseWidth(content);
            else if(token == "MAX_GRADIENT") (*this).readMaxGradient(content);
            
			
		}
    } 

    cout << "Ok" << endl;
    fileObject.close();
}

void pfgse_config::readGiromagneticRatio(string s)
{
	this->GIROMAGNETIC_RATIO = std::stod(s);
}

void pfgse_config::readPulseWidth(string s)
{
	this->PULSE_WIDTH = std::stod(s);
}

void pfgse_config::readMaxGradient(string s)
{
    this->MAX_GRADIENT = std::stod(s);
}

void pfgse_config::readGradientSamples(string s)
{
    this->GRADIENT_SAMPLES = std::stoi(s);
}





