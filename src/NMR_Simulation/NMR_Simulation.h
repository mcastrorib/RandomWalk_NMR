#ifndef NMR_SIMULATION_H_
#define NMR_SIMULATION_H_

// include string stream manipulation functions
#include <sstream>
#include <iomanip>

// include OpenCV core functions
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>

// STL vector container
#include <vector>

// include OpenMP for multicore implementation
#include <omp.h>

#include "NMR_defs.h"
#include "CollisionHistogram.h"
#include "../ConsoleInput/consoleInput.h"
#include "../BitBlock/bitBlock.h"
#include "../Walker/walker.h"

using namespace std;
using namespace cv;

class NMR_Simulation
{
public:
    // Class attributes:

    // RW simulation parameters
    string simulationName;
    string simulationDirectory;
    uint simulationSteps;
    uint stepsPerEcho;
    uint numberOfEchoes;
    uint64_t initialSeed;
    bool seedFlag;
    bool gpu_use;

    uint numberOfPores;
    double porosity;
    double walkerOccupancy;
    uint numberOfWalkers;

    // vector objects
    vector<Pore> pores;
    vector<uint> walkersIDList;
    vector<Walker> walkers;
    vector<double> globalEnergy;
    vector<double> decayTimes;

    // T2 distribution vectors
    string inputT2File;
    vector<double> T2_bins;
    vector<double> T2_input;
    vector<double> T2_simulated;

    // physical properties
    double timeInterval; // time interval between each walker step
    double diffusionCoefficient;

    // image attributes
    ImagePath imagePath;
    uint numberOfImages;
    double imageResolution;
    double imageVoxelResolution;    
    uint height;
    uint width;
    uint depth;
    vector<Mat> binaryMap;
    BitBlock bitBlock;

    // digital voxel/image amplification
    int voxelDivision;
    bool voxelDivisionApplied;

    // histogram used in fast simulations
    CollisionHistogram histogram;
    vector<CollisionHistogram> histogramList;
    double *penalties;

    // NMR_3D methods:
    // default constructors
    NMR_Simulation(string _name = "NMR_simulation");

    //copy constructors
    NMR_Simulation(const NMR_Simulation &_otherImage);

    // default destructor
    virtual ~NMR_Simulation()
    {
        delete[] penalties;
        penalties = NULL;
        cout << "NMR_simulation object destroyed." << endl;
    }

    void reset()
    {
        if (this->walkers.size() > 0)
        {
            walkers.clear();
        }
        if (this->globalEnergy.size() > 0)
        {
            globalEnergy.clear();
        }

        // free(this->bitBlock.blocks);
        // this->bitBlock.blocks = NULL;
    }

    void resetGlobalEnergy()
    {
        if (this->globalEnergy.size() > 0)
        {
            globalEnergy.clear();
        }
    }

    void resetT2Distribution()
    {
        if (this->T2_simulated.size() > 0)
        {
            T2_simulated.clear();
        }
    }

    void clear()
    {
        // RW simulation parameters
        this->numberOfPores = 0;
        this->numberOfWalkers = 0;

        // vector objects
        this->pores.clear();
        this->walkers.clear();
        this->globalEnergy.clear();
        this->decayTimes.clear();

        // T2 distribution vectors
        this->T2_bins.clear();
        this->T2_input.clear();
        this->T2_simulated.clear();

        // image attributes
        this->binaryMap.clear();
        this->bitBlock.clear();

        // histogram used in fast simulations
        this->histogram.clear();
        this->histogramList.clear();
        if(this->penalties != NULL)
        {
            delete[] this->penalties;
            this->penalties = NULL;
        }
    }

    // Class methods:
    // read
    void setImage(ImagePath path, uint images);
    void readImage();
    void loadRockImage();
    void createBinaryMap(Mat &_rockImage, uint slice);
    void createBitBlockMap();

    // set walkers
    void setSimulation(double occupancy, uint64_t seed, bool use_GPU);
    void setGPU(bool _useGPU);
    void setImageOccupancy(double _occupancy);
    void setInitialSeed(uint64_t _seed);
    void setFreeDiffusionCoefficient(double _bulk);
    void setImageResolution(double _resolution);
    void setImageVoxelResolution();
    void setTimeInterval();
    void setVoxelDivision(uint _shifts);
    void applyVoxelDivision(uint _shifts);
    void setNumberOfStepsPerEcho(uint _stepsPerEcho);
    void setNumberOfStepsFromTime(double time);
    void setTimeFramework(uint _steps);
    void setTimeFramework(double _time);
    void setWalkers(Point3D _point, uint _numberOfWalkers = 0);
    void setWalkers(Point3D _point1, Point3D _point2, uint _numberOfWalkers = 0);
    void setWalkers(uint _numberOfWalkers = 0, bool _randomInsertion = false);
    void countPoresInBinaryMap();
    void countPoresInBitBlock();
    void countPoresInCubicSpace(Point3D _vertex1, Point3D _vertex2);
    void updatePorosity();
    void updateNumberOfPores();
    void createPoreList();
    void createPoreList(Point3D _vertex1, Point3D _vertex2);
    void freePoreList(){ if(this->pores.size() > 0) this->pores.clear();}
    void setNumberOfWalkers(uint _numberOfWalkers = 0);
    void updateWalkerOccupancy();
    void createWalkersIDList();
    uint removeRandomIndexFromPool(vector<uint> &_pool, uint _randomIndex);
    void createWalkers();
    void placeWalkers();
    void placeWalkersUniformly();
    void placeWalkersByChance();
    void placeWalkersInSamePoint(uint x = 0, uint y = 0, uint z = 0);
    void placeWalkersInCubicSpace(Point3D _vertex1, Point3D _vertex2);

    // histogram
    void initHistogramList();
    void createHistogram();
    void createHistogram(uint histID, uint _steps);

    // laplace inversion
    void normalizeEnergyDecay();
    void applyLaplaceInversion();

    // cost function methods
    void updateWalkersRelaxativity(vector<double> &parameters);
    void updateWalkersRelaxativity(double rho);
    double correlateT2();
    double leastSquaresT2();

    typedef void (NMR_Simulation::*mapSim_ptr)();
    typedef void (NMR_Simulation::*walkSim_ptr)();

private:
    mapSim_ptr mapSimulationPointer;
    walkSim_ptr walkSimulationPointer;

    // simulations using openMP library for multicore CPU application
    void mapSimulation_OMP();
    void walkSimulation_OMP();
    

    // simulations in CUDA language for GPU application
    void mapSimulation_CUDA_2D();
    void mapSimulation_CUDA_2D_histograms();
    void walkSimulation_CUDA_2D();

    void mapSimulation_CUDA_3D();
    void mapSimulation_CUDA_3D_histograms();
    void walkSimulation_CUDA_3D();

    double diffusionSimulation_CUDA(double gradientMagnitude = DEFAULT_GRADIENT, 
                                  double tinyDelta = PFGSE_TINY_DELTA, 
                                  double giromagneticRatio = GIROMAGNETIC_RATIO);

    double diffusionSimulation_OMP(double gradientMagnitude = DEFAULT_GRADIENT, 
                                 double tinyDdelta = PFGSE_TINY_DELTA, 
                                 double giromagneticRatio = GIROMAGNETIC_RATIO);

public:
    void mapSimulation(void)
    {
        (this->*mapSimulationPointer)();
    }
    void walkSimulation(void)
    {
        (this->*walkSimulationPointer)();
        (*this).normalizeEnergyDecay();
    }
    void associateMapSimulation();
    void associateWalkSimulation();

    // PFG NMR simulation
    double PFG(double gradientMagnitude = DEFAULT_GRADIENT, 
             double tinyDelta = PFGSE_TINY_DELTA, 
             double giromagneticRatio = GIROMAGNETIC_RATIO)
    { 
        if(this->gpu_use)
            return (*this).diffusionSimulation_CUDA(gradientMagnitude, tinyDelta, giromagneticRatio);
        else
            return (*this).diffusionSimulation_OMP(gradientMagnitude, tinyDelta, giromagneticRatio);

    }
    double compute_pfgse_k_value(double gradientMagnitude, 
                                 double tiny_delta = PFGSE_TINY_DELTA, 
                                 double giromagneticRatio = GIROMAGNETIC_RATIO);

    void fastSimulation();
    void histSimulation();
    void createPenaltiesVector(vector<double> &_sigmoid);
    void createPenaltiesVector(double rho = DEFAULT_RELAXATIVITY);

    // Class supermethod:
    void saveInfo();
    void save();
    void save(string _otherDir);
    void getT2Distribution();

    uint pickRandomIndex(uint _maxValue);
    Pore removeRandomPore(vector<Pore> &_pores, uint _randomIndex);

    void printInputDetails();
    void info();
    void dummy(){ cout << "hey, I'm here dude." << endl;}

    // 'get' inline methods
    // simulation parameters
    inline uint getSimulationSteps() { return this->simulationSteps; }
    inline uint getStepsPerEcho() { return this->stepsPerEcho; }
    inline uint getNumberOfEchoes() { return this->numberOfEchoes; }
    inline uint64_t getInitialSeed() { return this->initialSeed; }
    inline bool getSeedFlag() { return this->seedFlag; }

    // pore e walkers
    inline uint getNumberOfPores() { return this->numberOfPores; }
    inline double getWalkerOccupancy() { return this->walkerOccupancy; }
    inline uint getNumberOfWalkers() { return this->numberOfWalkers; }
    inline vector<Pore> getPores() { return this->pores; }
    inline vector<Walker> getWalkers() { return this->walkers; }
    inline vector<double> getGlobalEnergy() { return this->globalEnergy; }

    // physical attributes
    inline double getTimeInterval() { return this->timeInterval; }
    inline double getDiffusionCoefficient() { return this->diffusionCoefficient; }

    // image attributes
    inline string getImagePath() { return this->imagePath.completePath; }
    inline uint getNumberOfImages() { return this->numberOfImages; }
    inline double getImageResolution() { return this->imageResolution; }
    inline double getImageVoxelResolution() { return this->imageVoxelResolution; }
    inline uint getVoxelDivision() { return this->voxelDivision; }
    inline uint getImageHeight() { return this->height; }
    inline uint getImageWidth() { return this->width; }
    inline uint getImageDepth() { return this->depth; }
    inline vector<Mat> getBinaryMap() { return this->binaryMap; }
    inline BitBlock getBitBlock() { return this->bitBlock; }
    inline string convertFileIDToString(uint id, uint digits)
    {
        stringstream result;
        result << std::setfill('0') << std::setw(digits) << id;
        return result.str();
    }

    // output generation class methods
    string createDirectoryForResults();
    void saveImageInfo(string filedir);
    void saveEnergyDecay(string filedir);
    void saveWalkerCollisions(string filedir);
    void saveBitBlock(string filedir);
    void saveNMRT2(string filedir);
    void saveHistogram(string filedir);
    void saveHistogramList(string filedir);

    // read T2 data from file
    void readT2fromFile(string filename);

    // methods to set and read input T2 file
    void setInputT2(string inputT2_filename)
    {
        this->inputT2File = inputT2_filename;
    }
    void readInputT2();
};

#endif