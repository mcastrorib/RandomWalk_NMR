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
rwnmr_config::rwnmr_config(const string configFile) : config_filepath(configFile)
{
	vector<double> RHO();
	
	string default_dirpath = CONFIG_ROOT;
	string default_filename = RWNMR_CONFIG_DEFAULT;
	(*this).readConfigFile(default_dirpath + default_filename);
	if(configFile != (default_dirpath + default_filename)) (*this).readConfigFile(configFile);
}

//copy constructors
rwnmr_config::rwnmr_config(const rwnmr_config &otherConfig)
{
	this->config_filepath = otherConfig.config_filepath;
    this->NAME = otherConfig.NAME;
    this->WALKERS = otherConfig.WALKERS;
    this->WALKERS_PLACEMENT = otherConfig.WALKERS_PLACEMENT;
    this->PLACEMENT_DEVIATION = otherConfig.PLACEMENT_DEVIATION;
    this->RHO_TYPE = otherConfig.RHO_TYPE;
    this->RHO = otherConfig.RHO;
    this->D0 = otherConfig.D0; 
    this->STEPS_PER_ECHO = otherConfig.STEPS_PER_ECHO;
    this->SEED = otherConfig.SEED;
    this->BC = otherConfig.BC;

    // SAVE MODE
    this->SAVE_IMG_INFO = otherConfig.SAVE_IMG_INFO;
    this->SAVE_BINIMG = otherConfig.SAVE_BINIMG;

    // HISTOGRAM SIZE
    this->HISTOGRAMS = otherConfig.HISTOGRAMS;  
    this->HISTOGRAM_SIZE = otherConfig.HISTOGRAM_SIZE;

    // -- OPENMP MODE
    this->OPENMP_USAGE = otherConfig.OPENMP_USAGE;
    this->OPENMP_THREADS = otherConfig.OPENMP_THREADS;

    // -- CUDA/GPU PARAMETERS
    this->GPU_USAGE = otherConfig.GPU_USAGE;
    this->BLOCKS = otherConfig.BLOCKS;
    this->THREADSPERBLOCK = otherConfig.THREADSPERBLOCK;
    this->ECHOESPERKERNEL = otherConfig.ECHOESPERKERNEL;
    this->MAX_RWSTEPS = otherConfig.MAX_RWSTEPS;
    this->REDUCE_IN_GPU = otherConfig.REDUCE_IN_GPU;
    

    // -- MPI COMMUNICATION
    this->BITBLOCKS_BATCHES_SIZE = otherConfig.BITBLOCKS_BATCHES_SIZE;
    this->BITBLOCK_PROP_SIZE = otherConfig.BITBLOCK_PROP_SIZE;
    this->NMR_T2_SIZE = otherConfig.NMR_T2_SIZE;
    this->NMR_START_TAG = otherConfig.NMR_START_TAG;
    this->NMR_BITBLOCK_TAG = otherConfig.NMR_BITBLOCK_TAG;
    this->NMR_BATCH_TAG = otherConfig.NMR_BATCH_TAG;
    this->NMR_T2_TAG = otherConfig.NMR_T2_TAG;
    this->NMR_END_TAG = otherConfig.NMR_END_TAG;
}

// read config file
void rwnmr_config::readConfigFile(const string configFile)
{
    ifstream fileObject;
    fileObject.open(configFile, ios::in);
    if (fileObject.fail())
    {
        cout << "Could not open rwnmr config file from disc." << endl;
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

			if(token == "NAME")	(*this).readName(content);
			else if(token == "WALKERS") (*this).readWalkers(content);
			else if(token == "WALKERS_PLACEMENT") (*this).readWalkersPlacement(content);
			else if(token == "PLACEMENT_DEVIATION") (*this).readPlacementDeviation(content);
			else if(token == "RHO_TYPE") (*this).readRhoType(content);
			else if(token == "RHO") (*this).readRho(content);
			else if(token == "STEPS_PER_ECHO") (*this).readStepsPerEcho(content);
			else if(token == "D0") (*this).readD0(content);
			else if(token == "SEED") (*this).readSeed(content);
			else if(token == "BC") (*this).readBC(content);
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

    fileObject.close();
}

void rwnmr_config::readName(string s)
{
	this->NAME = s;
}

void rwnmr_config::readWalkers(string s)
{
	this->WALKERS = std::stoi(s);
}

void rwnmr_config::readWalkersPlacement(string s)
{
	this->WALKERS_PLACEMENT = s;
}

void rwnmr_config::readPlacementDeviation(string s)
{
	this->PLACEMENT_DEVIATION = std::stoi(s);
}

void rwnmr_config::readRhoType(string s)
{
	if(s == "uniform") this->RHO_TYPE = "uniform";
	else if(s == "sigmoid") this->RHO_TYPE = "sigmoid";
	else this->RHO_TYPE = "undefined";
}

void rwnmr_config::readRho(string s) // vector?
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
		} else
		{
			this->RHO.push_back(std::stod(s));
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

void rwnmr_config::readBC(string s)
{
	if(s == "periodic") this->BC = "periodic";
	else this->BC = "noflux";
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
