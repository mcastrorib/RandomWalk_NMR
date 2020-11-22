// include C++ standard libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>

#include "rwnmr_config.h"

using namespace std;

// default constructors
rwnmr_config::rwnmr_config(const string configFile)
{
	vector<double> RHO();
	(*this).readConfigFile(configFile);
}

//copy constructors
rwnmr_config::rwnmr_config(const rwnmr_config &otherConfig)
{}

// read config file
void rwnmr_config::readConfigFile(const string configFile)
{
	cout << "reading rwnmr configs from file...";

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

			if(token == "NAME")	(*this).readName(content);
			else if(token == "DB_PATH") (*this).readDBPath(content);
			else if(token == "WALKERS") (*this).readWalkers(content);
			else if(token == "RHO_TYPE") (*this).readRhoType(content);
			else if(token == "RHO") (*this).readRho(content);
			else if(token == "STEPS_PER_ECHO") (*this).readStepsPerEcho(content);
			else if(token == "D0") (*this).readD0(content);
			else if(token == "SEED") (*this).readSeed(content);
			else if(token == "SAVE_IMG_INFO") (*this).readSaveImgInfo(content);
			else if(token == "SAVE_BINIMG") (*this).readSaveBinImg(content);
			else if(token == "HISTOGRAMS") (*this).readHistograms(content);
			else if(token == "HISTOGRAM_SIZE") (*this).readHistogramSize(content);
			else if(token == "OPENMP_USAGE") (*this).readOpenMPUsage(content);
			else if(token == "OPENMP_THREADS") (*this).readOpenMPThreads(content);
			else if(token == "GPU_USAGE") (*this).readGPUUsage(content);
			else if(token == "BLOCKS") (*this).readBlocks(content);
			else if(token == "THREADSPERBLOCK") (*this).readThreadsPerBlock(content);
			else if(token == "ECHOESPERKERNEL") (*this).readEchoesPerKernel(content);
			else if(token == "REDUCE_IN_GPU") (*this).readReduceInGPU(content);
			else if(token == "MAX_RWSTEPS") (*this).readMaxRWSteps(content);
			else if(token == "BITBLOCKS_BATCHES_SIZE") (*this).readBitBlockBatchesSize(content);
			else if(token == "BITBLOCK_PROP_SIZE") (*this).readBitBlockPropertiesSize(content);
			else if(token == "NMR_T2_SIZE") (*this).readNMRT2Size(content);
			else if(token == "NMR_START_TAG") (*this).readStartTag(content);
			else if(token == "NMR_BITBLOCK_TAG") (*this).readBitBlockTag(content);
			else if(token == "NMR_BATCH_TAG") (*this).readBatchTag(content);
			else if(token == "NMR_T2_TAG") (*this).readT2Tag(content);
			else if(token == "NMR_END_TAG") (*this).readEndTag(content);
		}
    } 

    cout << "Ok" << endl;
    fileObject.close();
}

void rwnmr_config::readName(string s)
{
	this->NAME = s;
}

void rwnmr_config::readDBPath(string s)
{
	this->DB_PATH = s;
}

void rwnmr_config::readWalkers(string s)
{
	this->WALKERS = std::stoi(s);
}

void rwnmr_config::readRhoType(string s)
{
	if(s == "'uniform'") this->RHO_TYPE = "uniform";
	else if(s == "'sigmoid'") this->RHO_TYPE = "sigmoid";
	else this->RHO_TYPE = "undefined";
}

void rwnmr_config::readRho(string s) // vector?
{
	if(this->RHO_TYPE == "uniform")
	{
		this->RHO.push_back(std::stod(s)); 
	} else 
	if(this->RHO_TYPE == "sigmoid")
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
				this->RHO.push_back(std::stod(token));
			}
			// add value to RHO attribute
			this->RHO.push_back(std::stod(strvec));
		}
	}
}

void rwnmr_config::readD0(string s)
{
	this->D0 = std::stod(s);
}

void rwnmr_config::readStepsPerEcho(string s)
{
	this->STEPS_PER_ECHO = std::stoi(s);
}

void rwnmr_config::readSeed(string s)
{
	this->SEED = std::stol(s);
}

// -- Saving
void rwnmr_config::readSaveImgInfo(string s)
{
	if(s == "true") this->SAVE_IMG_INFO = true;
	else this->SAVE_IMG_INFO = false;
}

void rwnmr_config::readSaveBinImg(string s)
{
	if(s == "true") this->SAVE_BINIMG = true;
	else this->SAVE_BINIMG = false;
}

// Histograms
void rwnmr_config::readHistograms(string s)
{
	this->HISTOGRAMS = std::stoi(s);
}

void rwnmr_config::readHistogramSize(string s)
{
	this->HISTOGRAM_SIZE = std::stoi(s);
}

// -- OpenMP
void rwnmr_config::readOpenMPUsage(string s)
{
	if(s == "true") this->OPENMP_USAGE = true;
	else this->OPENMP_USAGE = false;
}

void rwnmr_config::readOpenMPThreads(string s)
{
	this->OPENMP_THREADS = std::stoi(s);
}

// -- CUDA/GPU Params
void rwnmr_config::readGPUUsage(string s)
{
	if(s == "true") this->GPU_USAGE = true;
	else this->GPU_USAGE = false;
}

void rwnmr_config::readBlocks(string s)
{
	this->BLOCKS = std::stoi(s);
}

void rwnmr_config::readThreadsPerBlock(string s)
{
	this->THREADSPERBLOCK = std::stoi(s);
}

void rwnmr_config::readEchoesPerKernel(string s)
{
	this->ECHOESPERKERNEL = std::stoi(s);
}

void rwnmr_config::readMaxRWSteps(string s)
{
	this->MAX_RWSTEPS = std::stoi(s);
}

void rwnmr_config::readReduceInGPU(string s)
{
	if(s == "true") this->REDUCE_IN_GPU = true;
	else this->REDUCE_IN_GPU = false;
}

// -- MPI Params
void rwnmr_config::readBitBlockBatchesSize(string s)
{
	this->BITBLOCKS_BATCHES_SIZE = std::stoi(s);
}

void rwnmr_config::readBitBlockPropertiesSize(string s)
{
	this->BITBLOCK_PROP_SIZE = std::stoi(s);
}

void rwnmr_config::readNMRT2Size(string s)
{
	this->NMR_T2_SIZE = std::stoi(s);
}

void rwnmr_config::readStartTag(string s)
{
	this->NMR_START_TAG = std::stoi(s);
}

void rwnmr_config::readBitBlockTag(string s)
{
	this->NMR_BITBLOCK_TAG = std::stoi(s);
}

void rwnmr_config::readBatchTag(string s)
{
	this->NMR_BATCH_TAG = std::stoi(s);
}

void rwnmr_config::readT2Tag(string s)
{
	this->NMR_T2_TAG = std::stoi(s);
}

void rwnmr_config::readEndTag(string s)
{
	this->NMR_END_TAG = std::stoi(s);
}
