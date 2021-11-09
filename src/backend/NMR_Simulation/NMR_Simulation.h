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

// include configuration file classes
#include "../ConfigFiles/rwnmr_config.h"
#include "../ConfigFiles/uct_config.h"

#include "NMR_defs.h"
#include "CollisionHistogram.h"
#include "../Utils/ImagePath.h"
#include "../BitBlock/bitBlock.h"
#include "../Walker/walker.h"

using namespace std;
using namespace cv;

class NMR_Simulation
{
public:
    // Class attributes:
    // Config attributes
    rwnmr_config rwNMR_config;
    uct_config uCT_config;

    // RW simulation parameters
    string simulationName;
    string simulationDirectory;
    string DBPath;
    uint simulationSteps;
    uint stepsPerEcho;
    uint numberOfEchoes;
    uint64_t initialSeed;
    bool seedFlag;
    bool gpu_use;

    uint numberOfPores;
    double porosity;
    uint interfacePoreMatrix;
    double SVp;
    double walkerOccupancy;
    uint numberOfWalkers;
    uint walkerSamples;

    // vector objects
    vector<Pore> pores;
    vector<uint> walkersIDList;
    vector<Walker> walkers;

    // physical properties
    double timeInterval; // time interval between each walker step
    double diffusionCoefficient;
    double bulkRelaxationTime;

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
    string boundaryCondition;

    // digital voxel/image amplification
    int voxelDivision;
    bool voxelDivisionApplied;

    // Collision histogram 
    CollisionHistogram histogram;
    vector<CollisionHistogram> histogramList;

    // NMR_3D methods:
    // default constructors
    NMR_Simulation(rwnmr_config _rwNMR_config, uct_config _uCT_config, string _project_root);

    //copy constructors
    // copy constructor
    NMR_Simulation(const NMR_Simulation &_otherSimulation)
    {
        this->rwNMR_config = _otherSimulation.rwNMR_config;
        this->uCT_config = _otherSimulation.uCT_config;
        this->simulationName = _otherSimulation.simulationName;
        this->simulationDirectory = _otherSimulation.simulationDirectory;
        this->simulationSteps = _otherSimulation.simulationSteps;
        this->stepsPerEcho = _otherSimulation.stepsPerEcho;
        this->numberOfEchoes = _otherSimulation.numberOfEchoes;
        this->initialSeed = _otherSimulation.initialSeed;
        this->seedFlag = _otherSimulation.seedFlag;
        this->gpu_use = _otherSimulation.gpu_use;
        this->boundaryCondition = _otherSimulation.boundaryCondition;

        this->numberOfPores = _otherSimulation.numberOfPores;
        this->porosity = _otherSimulation.porosity;
        this->walkerOccupancy = _otherSimulation.walkerOccupancy;
        this->numberOfWalkers = _otherSimulation.numberOfWalkers;
        this->walkerSamples = _otherSimulation.walkerSamples;

        // vectors attributes copy pointers to otherImage's vectors
        // should be tested if it works or if it should be done explicitly
        this->pores = _otherSimulation.pores;
        this->walkersIDList = _otherSimulation.walkersIDList;
        this->walkers = _otherSimulation.walkers;

        this->timeInterval = _otherSimulation.timeInterval;
        this->diffusionCoefficient = _otherSimulation.diffusionCoefficient;
        this->bulkRelaxationTime = _otherSimulation.bulkRelaxationTime;

        this->imagePath = _otherSimulation.imagePath;
        this->numberOfImages = _otherSimulation.numberOfImages;
        this->imageResolution = _otherSimulation.imageResolution;
        this->imageVoxelResolution = _otherSimulation.imageVoxelResolution;
        this->voxelDivision = _otherSimulation.voxelDivision;
        this->voxelDivisionApplied = _otherSimulation.voxelDivisionApplied;
        this->height = _otherSimulation.height;
        this->width = _otherSimulation.width;
        this->depth = _otherSimulation.depth;
        this->binaryMap = _otherSimulation.binaryMap;
        this->bitBlock = _otherSimulation.bitBlock;

        this->histogram = _otherSimulation.histogram;
        this->histogramList = _otherSimulation.histogramList;

        // pointers-to-method
        this->mapSimulationPointer = _otherSimulation.mapSimulationPointer;
    }

    // default destructor
    virtual ~NMR_Simulation()
    {
        (*this).clear();
        cout << "NMR_simulation object destroyed." << endl;
    }

    void reset()
    {
        if (this->walkers.size() > 0)
        {
            walkers.clear();
        }
        // free(this->bitBlock.blocks);
        // this->bitBlock.blocks = NULL;
    }

    void clear()
    {
        // RW simulation parameters
        this->numberOfPores = 0;
        this->numberOfWalkers = 0;

        // vector objects
        this->pores.clear();
        this->walkers.clear();

        // image attributes
        this->binaryMap.clear();
        this->bitBlock.clear();

        // histogram used in fast simulations
        this->histogram.clear();
        this->histogramList.clear();
    }

    // Class methods:
    // read
    void setImage(ImagePath path, uint images);
    void readImage();
    void loadRockImage();
    void loadRockImageFromList();
    void createBinaryMap(Mat &_rockImage, uint slice);
    void createBitBlockMap();

    // set walkers
    void setSimulation(double occupancy, uint64_t seed, bool use_GPU);
    void setGPU(bool _useGPU);
    void setImageOccupancy(double _occupancy);
    void setInitialSeed(uint64_t _seed, bool _flag=false);
    void setFreeDiffusionCoefficient(double _D0);
    void setBulkRelaxationTime(double _bulkTime);
    void setImageResolution(double _resolution);
    void setImageVoxelResolution();
    void setBoundaryCondition(string _bc);
    void setTimeInterval();
    void setVoxelDivision(uint _shifts);
    void applyVoxelDivision(uint _shifts);
    void setNumberOfStepsPerEcho(uint _stepsPerEcho);
    void setNumberOfStepsFromTime(double time);
    void setTimeFramework(uint _steps);
    void setTimeFramework(double _time);
    void setWalkers(void);
    void setWalkers(Point3D _point, uint _numberOfWalkers);
    void setWalkers(Point3D _point1, Point3D _point2, uint _numberOfWalkers);
    void setWalkers(uint _numberOfWalkers, bool _randomInsertion = false);
    void countPoresInBinaryMap();
    void countPoresInBitBlock();
    void countPoresInCubicSpace(Point3D _vertex1, Point3D _vertex2);
    void updatePorosity();
    void countInterfacePoreMatrix();
    void updateSVp();
    void updateNumberOfPores();
    void createPoreList();
    void createPoreList(Point3D _vertex1, Point3D _vertex2);
    void freePoreList(){ if(this->pores.size() > 0) this->pores.clear();}
    void setNumberOfWalkers(uint _numberOfWalkers = 0);
    void setWalkerSamples(uint _samples);
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

    // cost function methods
    void updateWalkersRelaxativity(vector<double> &parameters);
    void updateWalkersRelaxativity(double rho);

    typedef void (NMR_Simulation::*mapSim_ptr)();
    typedef void (NMR_Simulation::*walkSim_ptr)();

private:
    mapSim_ptr mapSimulationPointer;

    /* 
        RW collision mapping simulations
        CUDA/GPU routines generate collision histograms
    */ 
    void mapSimulation_OMP();
    void mapSimulation_CUDA_2D_histograms();
    void mapSimulation_CUDA_3D_histograms();


public:
    void mapSimulation(void)
    {
        (this->*mapSimulationPointer)();
    }
    void associateMapSimulation();


    // Class supermethod:
    void saveInfo();
    void save();
    void save(string _otherDir);

    uint pickRandomIndex(uint _maxValue);
    Pore removeRandomPore(vector<Pore> &_pores, uint _randomIndex);

    void printDetails();
    void info();
    void dummy(){ cout << "hey, I'm here dude." << endl;}

    // 'get' inline methods
    // simulation parameters
    inline string getDBPath() { return this->DBPath; }
    inline uint getSimulationSteps() { return this->simulationSteps; }
    inline uint getStepsPerEcho() { return this->stepsPerEcho; }
    inline uint getNumberOfEchoes() { return this->numberOfEchoes; }
    inline uint64_t getInitialSeed() { return this->initialSeed; }
    inline bool getSeedFlag() { return this->seedFlag; }
    inline bool getGPU() { return this->gpu_use; }
    inline string getBoundaryCondition() { return this->boundaryCondition; }

    // pore e walkers
    inline uint getNumberOfPores() { return this->numberOfPores; }
    inline double getWalkerOccupancy() { return this->walkerOccupancy; }
    inline uint getNumberOfWalkers() { return this->numberOfWalkers; }
    inline uint getWalkerSamples() { return this->walkerSamples; }
    inline vector<Pore> getPores() { return this->pores; }
    inline vector<Walker> getWalkers() { return this->walkers; }

    // physical attributes
    inline double getTimeInterval() { return this->timeInterval; }
    inline double getDiffusionCoefficient() { return this->diffusionCoefficient; }
    inline double getBulkRelaxationTime(){ return this->bulkRelaxationTime; }

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
    string createDirectoryForResults(string _root);
    void saveImageInfo(string filedir);
    void saveWalkerCollisions(string filedir);
    void saveBitBlock(string filedir);
    void saveHistogram(string filedir);
    void saveHistogramList(string filedir);
    
    void assemblyImagePath();
};

#endif