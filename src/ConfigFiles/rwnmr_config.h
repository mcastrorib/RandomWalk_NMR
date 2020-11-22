#ifndef RWNMR_CONFIG_H_
#define RWNMR_CONFIG_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class rwnmr_config
{
public:
    string NAME;
    string DB_PATH;
    uint WALKERS;
    string RHO_TYPE;
    vector<double> RHO;
    double D0; 
    uint STEPS_PER_ECHO;
    uint64_t SEED;

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
    rwnmr_config(const string configFile);

    //copy constructors
    rwnmr_config(const rwnmr_config &otherConfig);

    // default destructor
    virtual ~rwnmr_config()
    {
        // cout << "OMPLoopEnabler object destroyed succesfully" << endl;
    } 

    void readConfigFile(const string configFile);
    
    // -- RW Params
    void readName(string s);
    void readDBPath(string s);
    void readWalkers(string s);
    void readRhoType(string s);
    void readRho(string s); // vector?
    void readD0(string s); 
    void readStepsPerEcho(string s);
    void readSeed(string s);

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
};

#endif