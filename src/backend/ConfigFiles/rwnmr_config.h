#ifndef RWNMR_CONFIG_H_
#define RWNMR_CONFIG_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>
#include "configFiles_defs.h"

using namespace std;

class rwnmr_config
{
public:
    string config_filepath;
    string NAME;
    uint WALKERS;
    uint WALKER_SAMPLES;
    string WALKERS_PLACEMENT;
    uint PLACEMENT_DEVIATION;
    string RHO_TYPE;
    vector<double> RHO;
    double D0; 
    uint STEPS_PER_ECHO;
    uint64_t SEED;
    string BC;

    // SAVE MODE
    bool SAVE_IMG_INFO;
    bool SAVE_BINIMG;

    // HISTOGRAM SIZE
    uint HISTOGRAMS;  
    uint HISTOGRAM_SIZE;

    // -- OPENMP MODE
    bool OPENMP_USAGE;
    uint OPENMP_THREADS;

    // -- CUDA/GPU PARAMETERS
    bool GPU_USAGE;
    uint BLOCKS;
    uint THREADSPERBLOCK;
    uint ECHOESPERKERNEL;
    uint MAX_RWSTEPS;
    bool REDUCE_IN_GPU;
    

    // -- MPI COMMUNICATION
    uint BITBLOCKS_BATCHES_SIZE;
    uint BITBLOCK_PROP_SIZE;
    uint NMR_T2_SIZE;
    uint NMR_START_TAG;
    uint NMR_BITBLOCK_TAG;
    uint NMR_BATCH_TAG;
    uint NMR_T2_TAG;
    uint NMR_END_TAG;

    // default constructors
    rwnmr_config(){};
    rwnmr_config(const string configFile);

    //copy constructors
    rwnmr_config(const rwnmr_config &otherConfig);    

    // default destructor
    virtual ~rwnmr_config()
    {
        // cout << "OMPLoopEnabler object destroyed succesfully" << endl;
    } 

    void readConfigFile(const string configFile);
    
    // Read methods
    // -- RW Params
    void readName(string s);
    void readWalkers(string s);
    void readWalkerSamples(string s);
    void readWalkersPlacement(string s);
    void readPlacementDeviation(string s);
    void readRhoType(string s);
    void readRho(string s); // vector?
    void readD0(string s); 
    void readStepsPerEcho(string s);
    void readSeed(string s);
    void readBC(string s);

    // -- Saving
    void readSaveImgInfo(string s);
    void readSaveBinImg(string s);

    // Histograms
    void readHistograms(string s);  
    void readHistogramSize(string s);

    // -- OpenMP
    void readOpenMPUsage(string s);
    void readOpenMPThreads(string s);

    // -- CUDA/GPU Params
    void readGPUUsage(string s);
    void readBlocks(string s);
    void readThreadsPerBlock(string s);
    void readEchoesPerKernel(string s);
    void readMaxRWSteps(string s);
    void readReduceInGPU(string s);
    
    // -- MPI Params
    void readBitBlockBatchesSize(string s);
    void readBitBlockPropertiesSize(string s);
    void readNMRT2Size(string s);
    void readStartTag(string s);
    void readBitBlockTag(string s);
    void readBatchTag(string s);
    void readT2Tag(string s);
    void readEndTag(string s);

    // Get methods
    // -- RW Params
    string getConfigFilepath() {return this->config_filepath; }
    string getName(){ return this->NAME;}
    uint getWalkers(){ return this->WALKERS;}
    uint getWalkerSamples(){ return this->WALKER_SAMPLES;}
    string getWalkersPlacement(){ return this->WALKERS_PLACEMENT;}
    uint getPlacementDeviation(){ return this->PLACEMENT_DEVIATION;}
    string getRhoType(){ return this->RHO_TYPE;}
    vector<double> getRho(){ return this->RHO;}
    double getD0(){ return this->D0;}
    uint getStepsPerEcho(){ return this->STEPS_PER_ECHO;}
    uint64_t getSeed(){ return  this->SEED;}
    string getBC() { return this->BC; }

    // -- Saving
    bool getSaveImgInfo(){ return this->SAVE_IMG_INFO;}
    bool getSaveBinImg(){ return this->SAVE_BINIMG;}

    // Histograms
    uint getHistograms(){ return this->HISTOGRAMS;}
    uint getHistogramSize(){ return this->HISTOGRAM_SIZE;}

    // -- OpenMP
    bool getOpenMPUsage(){ return this->OPENMP_USAGE;}
    uint getOpenMPThreads(){ return this->OPENMP_THREADS;}

    // -- CUDA/GPU Params
    bool getGPUUsage(){ return this->GPU_USAGE;}
    uint getBlocks(){ return this->BLOCKS;}
    uint getThreadsPerBlock(){ return this->THREADSPERBLOCK;}
    uint getEchoesPerKernel(){ return this->ECHOESPERKERNEL;}
    uint getMaxRWSteps(){ return this->MAX_RWSTEPS;}
    bool getReduceInGPU(){ return this->REDUCE_IN_GPU;}
    
    // -- MPI Params
    uint getBitBlockBatchesSize(){ return this->BITBLOCKS_BATCHES_SIZE;}
    uint getBitBlockPropertiesSize(){ return this->BITBLOCK_PROP_SIZE;}
    uint getNMRT2Size(){ return this->NMR_T2_SIZE;}
    uint getStartTag(){ return this->NMR_START_TAG;}
    uint getBitBlockTag(){ return this->NMR_BITBLOCK_TAG;}
    uint getBatchTag(){ return this->NMR_BATCH_TAG;}
    uint getT2Tag(){ return this->NMR_T2_TAG;}
    uint getEndTag(){ return this->NMR_END_TAG;}
};

#endif