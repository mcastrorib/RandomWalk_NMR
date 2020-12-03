// include C++ standard libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <stdint.h>

#include "uct_config.h"

using namespace std;

// default constructors
uct_config::uct_config(const string configFile) : config_filepath(configFile)
{
	(*this).readConfigFile(configFile);
}

//copy constructors
uct_config::uct_config(const uct_config &otherConfig)
{
	this->config_filepath = otherConfig.config_filepath;
    this->DIR_PATH = otherConfig.DIR_PATH;
    this->FILENAME = otherConfig.FILENAME;
    this->FIRST_IDX = otherConfig.FIRST_IDX;
    this->DIGITS = otherConfig.DIGITS;
    this->EXTENSION = otherConfig.EXTENSION;
    this->SLICES = otherConfig.SLICES;
    this->RESOLUTION = otherConfig.RESOLUTION;
    this->VOXEL_DIVISION = otherConfig.VOXEL_DIVISION;
}

// read config file
void uct_config::readConfigFile(const string configFile)
{
	cout << "reading uct configs from file...";

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

			if(token == "DIR_PATH")	(*this).readDirPath(content);
			else if(token == "FILENAME") (*this).readFilename(content);
			else if(token == "FIRST_IDX") (*this).readFirstIdx(content);
			else if(token == "DIGITS") (*this).readDigits(content);
			else if(token == "EXTENSION") (*this).readExtension(content);
			else if(token == "SLICES") (*this).readSlices(content);
			else if(token == "RESOLUTION") (*this).readResolution(content);
			else if(token == "VOXEL_DIVISION") (*this).readVoxelDivision(content);
			
		}
    } 

    cout << "Ok" << endl;
    fileObject.close();
}

void uct_config::readDirPath(string s)
{
	this->DIR_PATH = s;
}

void uct_config::readFilename(string s)
{
	this->FILENAME = s;
}

void uct_config::readFirstIdx(string s)
{
	this->FIRST_IDX = std::stoi(s);
}

void uct_config::readDigits(string s)
{
	this->DIGITS = std::stoi(s);
}

void uct_config::readExtension(string s)
{
	this->EXTENSION = s;
}

void uct_config::readSlices(string s)
{
	this->SLICES = std::stoi(s);
}

void uct_config::readResolution(string s)
{
	this->RESOLUTION = std::stod(s);
}

void uct_config::readVoxelDivision(string s)
{
	this->VOXEL_DIVISION = std::stoi(s);
}


