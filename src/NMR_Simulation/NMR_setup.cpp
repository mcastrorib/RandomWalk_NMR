// include OpenCV core functions
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <random>
#include <vector>
#include <string>
#include <math.h>

// include OpenMP for multicore implementation
#include <omp.h>

//include
#include "../Walker/walker.h"
#include "../BitBlock/bitBlock.h"
#include "../RNG/xorshift.h"
#include "../FileHandler/fileHandler.h"
#include "../ConsoleInput/consoleInput.h"
#include "../Laplace/tikhonov.h"
#include "../Laplace/include/nmrinv_core.h"
#include "NMR_Simulation.h"
// #include "CollisionHistogram.h"
#include "../Utils/OMPLoopEnabler.h"

using namespace cv;
using namespace std;

// Class NMR_Simulation setup methods

// Class methods:
//defaul constructor
NMR_Simulation::NMR_Simulation(string _name) : numberOfPores(0),
                                               porosity(0.0),
                                               walkerOccupancy(0.0),
                                               numberOfWalkers(0),
                                               diffusionCoefficient(DIFFUSION_COEFFICIENT),
                                               imageResolution(IMAGE_RESOLUTION),
                                               stepsPerEcho(STEPS_PER_ECHO),
                                               voxelDivision(VOXEL_DIVISIONS),
                                               voxelDivisionApplied(false),
                                               penalties(NULL),
                                               initialSeed(INITIAL_SEED),
                                               gpu_use(false),
                                               seedFlag(false)

{
    vector<Mat> binaryMap();
    vector<Pore> pores();
    vector<uint> walkersIDList();
    vector<Walker> walkers();
    vector<double> globalEnergy();
    vector<double> decayTimes();
    vector<double> T2_bins();
    vector<double> T2_input();
    vector<double> T2_simulated();
    vector<CollisionHistogram> histogramList();

    // set simulation name and directory to save results
    this->simulationName = _name;
    this->simulationDirectory = createDirectoryForResults();

    // set default time step measurement
    (*this).setImageVoxelResolution();
    (*this).setTimeInterval();    
};

// copy constructor
NMR_Simulation::NMR_Simulation(const NMR_Simulation &_otherSimulation)
{

    this->simulationName = _otherSimulation.simulationName;
    this->simulationDirectory = _otherSimulation.simulationDirectory;
    this->simulationSteps = _otherSimulation.simulationSteps;
    this->stepsPerEcho = _otherSimulation.stepsPerEcho;
    this->numberOfEchoes = _otherSimulation.numberOfEchoes;
    this->initialSeed = _otherSimulation.initialSeed;
    this->seedFlag = _otherSimulation.seedFlag;
    this->gpu_use = _otherSimulation.gpu_use;

    this->numberOfPores = _otherSimulation.numberOfPores;
    this->porosity = _otherSimulation.porosity;
    this->walkerOccupancy = _otherSimulation.walkerOccupancy;
    this->numberOfWalkers = _otherSimulation.numberOfWalkers;

    // vectors attributes copy pointers to otherImage's vectors
    // should be tested if it works or if it should be done explicitly
    this->pores = _otherSimulation.pores;
    this->walkersIDList = _otherSimulation.walkersIDList;
    this->walkers = _otherSimulation.walkers;
    this->globalEnergy = _otherSimulation.globalEnergy;
    this->decayTimes = _otherSimulation.decayTimes;
    this->T2_input = _otherSimulation.T2_input;
    this->T2_simulated = _otherSimulation.T2_simulated;

    this->timeInterval = _otherSimulation.timeInterval;
    this->diffusionCoefficient = _otherSimulation.diffusionCoefficient;

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
    this->penalties = _otherSimulation.penalties;

    // pointers-to-method
    this->mapSimulationPointer = _otherSimulation.mapSimulationPointer;
    this->walkSimulationPointer = _otherSimulation.walkSimulationPointer;
}

void NMR_Simulation::setImage(ImagePath _path, uint _images)
{
    cout << endl
         << "SETTING UP RANDOM WALK NMR SIMULATION:" << endl;
    this->imagePath = _path;
    this->numberOfImages = _images;
    this->depth = this->numberOfImages;
}

void NMR_Simulation::setSimulation(double _occupancy, uint64_t _seed, bool _use_GPU)
{ 
    this->gpu_use = _use_GPU;
    this->walkerOccupancy = _occupancy;
    this->initialSeed = _seed; 
    if (this->initialSeed != DEFAULT_SEED)
    {
        this->seedFlag = true;
    }
    else
    {
        this->seedFlag = false;
    }
}

void NMR_Simulation::setGPU(bool _useGPU)
{
    this->gpu_use = _useGPU;
}

void NMR_Simulation::setImageOccupancy(double _occupancy)
{
    this->walkerOccupancy = _occupancy;
}

void NMR_Simulation::setInitialSeed(uint64_t _seed)
{
    this->initialSeed = _seed; 
    if (this->initialSeed != DEFAULT_SEED) this->seedFlag = true;
    else this->seedFlag = false;
}

void NMR_Simulation::setFreeDiffusionCoefficient(double _bulk)
{
    this->diffusionCoefficient = _bulk;
}

void NMR_Simulation::setImageResolution(double _resolution)
{
    this->imageResolution = _resolution;
}

void NMR_Simulation::setImageVoxelResolution()
{
    this->imageVoxelResolution = this->imageResolution / (double) this->voxelDivision;
}

void NMR_Simulation::setTimeInterval()
{
    this->timeInterval = (this->imageVoxelResolution * this->imageVoxelResolution) / 
                         (6 * this->diffusionCoefficient);
}

void NMR_Simulation::setVoxelDivision(uint _shifts)
{
    this->voxelDivision = pow(2,_shifts);
    if(this->voxelDivision > 0) this->voxelDivisionApplied = true;
    else this->voxelDivisionApplied = false;
}

void NMR_Simulation::applyVoxelDivision(uint _shifts)
{
    double time = omp_get_wtime();
    cout << "applying voxel division...";

    // reset resolution scales
    uint previousDivision = (*this).getVoxelDivision();
    (*this).setVoxelDivision(_shifts);
    (*this).setImageVoxelResolution();

    // reset time framework
    double previousTime = (*this).getTimeInterval();
    (*this).setTimeInterval();
    double timeFactor = previousTime / (*this).getTimeInterval();
    (*this).setNumberOfStepsPerEcho((uint) (*this).getStepsPerEcho() * timeFactor);
    uint steps = (*this).getStepsPerEcho() * (*this).getNumberOfEchoes();
    (*this).setTimeFramework(steps);

    // if walkers exists, fill intra-voxel sites //
    if(this->walkers.size() > 0)
    {
        double shiftFactor = (*this).getVoxelDivision() / (double) previousDivision;
        uint indexExpansion = (uint) shiftFactor;
        if(indexExpansion < 1) indexExpansion = 1;
        uint shiftX, shiftY, shiftZ;
        for(uint idx = 0; idx < this->walkers.size(); idx++)
        {   
            // randomly place walker in voxel sites
            shiftX = ((uint) this->walkers[idx].initialPosition.x * shiftFactor) + (*this).pickRandomIndex(indexExpansion);
            shiftY = ((uint) this->walkers[idx].initialPosition.y * shiftFactor) + (*this).pickRandomIndex(indexExpansion);
            shiftZ = ((uint) this->walkers[idx].initialPosition.z * shiftFactor) + (*this).pickRandomIndex(indexExpansion);
            this->walkers[idx].placeWalker(shiftX, shiftY, shiftZ);

            // update collision penalty
            this->walkers[idx].computeDecreaseFactor(this->imageVoxelResolution, this->diffusionCoefficient);
        }
    }

    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}

void NMR_Simulation::setNumberOfStepsPerEcho(uint _stepsPerEcho)
{
    this->stepsPerEcho = _stepsPerEcho;
}

// param @_time needs to be in miliseconds
void NMR_Simulation::setNumberOfStepsFromTime(double _time)
{
    _time = _time;
    this->simulationSteps =  _time * (6 * this->diffusionCoefficient / 
                                     (this->imageVoxelResolution * this->imageVoxelResolution)); 
}

void NMR_Simulation::setTimeFramework(uint _steps)
{
    this->numberOfEchoes = (uint)ceil( _steps / (double)this->stepsPerEcho);
    this->simulationSteps = this->numberOfEchoes * this->stepsPerEcho;
    
    // reserve memory space for global energy vector
    if(this->globalEnergy.size() != 0) this->globalEnergy.clear();
    this->globalEnergy.reserve(this->numberOfEchoes);
    if(this->decayTimes.size() != 0) this->decayTimes.clear();
    this->decayTimes.reserve(this->numberOfEchoes);
    
    double time = 0.0;
    double delta_t = this->stepsPerEcho * this->timeInterval;
    for(int echo = 0; echo < this->numberOfEchoes; echo++)
    {   
        time += delta_t;
        this->decayTimes.push_back(time);
    }
}

// param @_time needs to be in miliseconds
void NMR_Simulation::setTimeFramework(double _time)
{
    (*this).setNumberOfStepsFromTime(_time);
    this->numberOfEchoes = (uint)ceil((double)this->simulationSteps / (double)this->stepsPerEcho);
    this->simulationSteps = this->numberOfEchoes * this->stepsPerEcho;
    
    // reserve memory space for global energy vector
    if(this->globalEnergy.size() != 0) this->globalEnergy.clear();
    this->globalEnergy.reserve(this->numberOfEchoes + 1);
    if(this->decayTimes.size() != 0) this->decayTimes.clear();
    this->decayTimes.reserve(this->numberOfEchoes + 1);
    
    // fill time array
    double time = 0.0;
    double delta_t = this->stepsPerEcho * this->timeInterval;
    this->decayTimes.push_back(time);
    for(int echo = 0; echo < this->numberOfEchoes; echo++)
    {   
        time += delta_t;
        this->decayTimes.push_back(time);
    }
}

void NMR_Simulation::readImage()
{
    (*this).loadRockImage();
    (*this).createBitBlockMap();
    (*this).countPoresInBitBlock();
}

// void NMR_Simulation::setWalkers(uint _numberOfWalkers)
// {    
//     (*this).setNumberOfWalkers(_numberOfWalkers);
//     (*this).updateWalkerOccupancy();
//     (*this),createPoreList();
//     (*this).createWalkersIDList();
//     (*this).createWalkers();
//     (*this).placeWalkersUniformly();

//     // associate rw simulation methods
//     (*this).associateMapSimulation();
//     (*this).associateWalkSimulation();    
// }

void NMR_Simulation::setWalkers(uint _numberOfWalkers, bool _randomInsertion)
{    
    (*this).setNumberOfWalkers(_numberOfWalkers);
    (*this).updateWalkerOccupancy();
    (*this).createWalkers();

    if(_randomInsertion == false)
    {
        (*this),createPoreList();
        (*this).createWalkersIDList();
        (*this).placeWalkersUniformly(); 
        (*this).freePoreList();
    } else
    {
        (*this).placeWalkersByChance();
    }

    // associate rw simulation methods
    (*this).associateMapSimulation();
    (*this).associateWalkSimulation();    
}

void NMR_Simulation::setWalkers(Point3D _point, uint _numberOfWalkers)
{    
    (*this).setNumberOfWalkers(_numberOfWalkers);
    (*this).updateWalkerOccupancy();
    (*this).createWalkers();
    (*this).createPoreList();
    (*this).placeWalkersInSamePoint(_point.x, _point.y, _point.z);

    // associate rw simulation methods
    (*this).associateMapSimulation();
    (*this).associateWalkSimulation();    
}

void NMR_Simulation::setWalkers(Point3D _point1, Point3D _point2, uint _numberOfWalkers)
{    
    (*this).setNumberOfWalkers(_numberOfWalkers);
    (*this).updateWalkerOccupancy();
    (*this).countPoresInCubicSpace(_point1, _point2);
    (*this).createPoreList(_point1, _point2);
    (*this).createWalkers();
    (*this).placeWalkersInCubicSpace(_point1, _point2);

    // associate rw simulation methods
    (*this).associateMapSimulation();
    (*this).associateWalkSimulation();    
}

// save results
void NMR_Simulation::saveInfo()
{
    if(NMR_SAVE_IMAGE_INFO)
    {   
        (*this).saveImageInfo(this->simulationDirectory);
    }    
}

void NMR_Simulation::save()
{
    double time = omp_get_wtime();
    cout << "saving results...";

    if(NMR_SAVE_DECAY) 
    {
        (*this).saveEnergyDecay(this->simulationDirectory);
    }
    
    if(NMR_SAVE_COLLISIONS)
    {
        (*this).saveWalkerCollisions(this->simulationDirectory);
    }

    if(NMR_SAVE_T2)
    {
        (*this).saveNMRT2(this->simulationDirectory);
    }    

    if(NMR_SAVE_HISTOGRAM)
    {
        (*this).saveHistogram(this->simulationDirectory);
    }    

    if(NMR_SAVE_HISTOGRAM_LIST)
    {
        (*this).saveHistogramList(this->simulationDirectory);
    }

    if(NMR_SAVE_BINIMAGE)
    {   
        (*this).saveBitBlock(this->simulationDirectory);
    }

    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}

// save results in other directory
void NMR_Simulation::save(string _otherDir)
{
    double time = omp_get_wtime();
    cout << "saving results...";
    
    if(NMR_SAVE_IMAGE_INFO)
    {   
        (*this).saveImageInfo(_otherDir);
    }

    if(NMR_SAVE_DECAY) 
    {
        (*this).saveEnergyDecay(_otherDir);
    }
    
    if(NMR_SAVE_COLLISIONS)
    {
        (*this).saveWalkerCollisions(_otherDir);
    }

    if(NMR_SAVE_T2)
    {
        (*this).saveNMRT2(_otherDir);
    }    

    if(NMR_SAVE_HISTOGRAM)
    {
        (*this).saveHistogram(_otherDir);
    }    

    if(NMR_SAVE_HISTOGRAM_LIST)
    {
        (*this).saveHistogramList(_otherDir);
    }

    if(NMR_SAVE_BINIMAGE)
    {   
        (*this).saveBitBlock(_otherDir);
    }

    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}

// apply Laplace Inversion to get T2 distribution from energy decay
void NMR_Simulation::getT2Distribution()
{

    string filename = "/NMR_decay.txt";
    laplaceInverse(this->simulationDirectory, filename);
    readT2fromFile(this->simulationDirectory);
}

void NMR_Simulation::loadRockImage()
{
    double time = omp_get_wtime();
    cout << "loading rock image ...";

    // reserve memory for binaryMap
    this->binaryMap.reserve(numberOfImages);

    // constant strings
    string currentDirectory = this->imagePath.path;
    string currentFileName = this->imagePath.filename;
    string currentExtension = this->imagePath.extension;

    // variable strings
    string currentFileID;
    string currentImagePath;

    uint firstImage = this->imagePath.fileID;
    uint digits = this->imagePath.digits;

    for (uint slice = 0; slice < this->numberOfImages; slice++)
    {
        // identifying next image to be read
        currentFileID = (*this).convertFileIDToString(firstImage + slice, digits);
        currentImagePath = currentDirectory + currentFileName + currentFileID + currentExtension;

        Mat rockImage = imread(currentImagePath, 1);

        if (!rockImage.data)
        {
            cout << "Error: No image data in file " << currentImagePath << endl;
            exit(1);
        }

        this->height = rockImage.rows;
        this->width = rockImage.cols * rockImage.channels();

        (*this).createBinaryMap(rockImage, slice);
    }

    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl;
}

void NMR_Simulation::createBinaryMap(Mat &_rockImage, uint slice)
{
    // create an "empty" image to be filled in binary map vector
    Mat emptyMap = Mat::zeros(_rockImage.rows, _rockImage.cols, CV_8UC1);
    binaryMap.push_back(emptyMap);

    int channels = _rockImage.channels();
    uchar *rockImagePixel;
    uchar *binaryMapPixel;

    for (int row = 0; row < this->height; ++row)
    {
        rockImagePixel = _rockImage.ptr<uchar>(row);
        binaryMapPixel = this->binaryMap[slice].ptr<uchar>(row);
        int mapColumn = 0;

        for (int column = 0; column < this->width; column += channels)
        {
            int currentChannel = 0;
            bool pixelIsPore = true;

            while (currentChannel < channels && pixelIsPore != false)
            {

                if (rockImagePixel[column + currentChannel] != 0)
                {
                    pixelIsPore = false;
                }

                currentChannel++;
            }

            if (pixelIsPore == false)
            {
                binaryMapPixel[mapColumn] = 255;
            }

            mapColumn++;
        }
    };
}

void NMR_Simulation::createBitBlockMap()
{
    double time = omp_get_wtime();
    cout << "creating (bit)block map...";
    this->bitBlock.createBlockMap(this->binaryMap);

    // update image info
    this->width = this->bitBlock.imageColumns;
    this->height = this->bitBlock.imageRows;
    this->depth = this->bitBlock.imageDepth;
    this->binaryMap.clear();

    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl;
}

// deprecated
void NMR_Simulation::countPoresInBinaryMap()
{
    double time = omp_get_wtime(); 
    cout << "counting ";

    for (uint slice = 0; slice < this->numberOfImages; slice++)
    { // accept only char type matrices
        CV_Assert(binaryMap[slice].depth() == CV_8U);

        uint mapWidth = (uint)binaryMap[slice].cols;
        uchar *binaryMapPixel;

        for (uint row = 0; row < height; ++row)
        {
            // psosition pointer at first element in current row
            binaryMapPixel = binaryMap[slice].ptr<uchar>(row);

            for (uint column = 0; column < mapWidth; column++)
            {
                if (binaryMapPixel[column] == 0)
                {
                    // x coordinate corresponds to column in binary map Mat structure
                    // y coordinate corresponds to row in the binary map Mat structure
                    Pore detectedPore = {column, row, slice};
                    pores.insert(pores.end(), detectedPore);
                    this->numberOfPores++;
                }
            }
        }
    }

    this->binaryMap.clear();
    (*this).updatePorosity();

    time = omp_get_wtime() - time;
    cout << this->numberOfPores << " pore voxels in rock image...Ok. (" << time << " seconds)." << endl; 
    cout << "porosity: " << this->porosity << endl;
}

void NMR_Simulation::countPoresInBitBlock()
{
    double time = omp_get_wtime(); 
    cout << "counting ";

    // consider 2 or 3 dimensions
    bool dim3 = false; 
    if(this->bitBlock.imageDepth > 1) 
        dim3 = true;

    // first, count all pores in image
    this->numberOfPores = 0;
    for(uint z = 0; z < this->bitBlock.imageDepth; z++)
    {
        for(uint y = 0; y < this->bitBlock.imageRows; y++)
        {
            for(uint x = 0; x < this->bitBlock.imageColumns; x++)
            {
                int block, bit;
                if(dim3 == true)
                {
                    block = this->bitBlock.findBlock(x, y, z);
                    bit = this->bitBlock.findBitInBlock(x, y, z);
                } else
                {
                    block = this->bitBlock.findBlock(x, y);
                    bit = this->bitBlock.findBitInBlock(x, y);                    
                }

                // now check if bit is pore or wall
                if (!this->bitBlock.checkIfBitIsWall(block, bit))
                {
                    this->numberOfPores++;
                }
            }
        }       
    }

    (*this).updatePorosity();

    time = omp_get_wtime() - time;
    cout << this->numberOfPores << " pore voxels in rock image...Ok. (" << time << " seconds)." << endl; 
    cout << "porosity: " << this->porosity << endl;
}

void NMR_Simulation::countPoresInCubicSpace(Point3D _vertex1, Point3D _vertex2)
{
    double time = omp_get_wtime(); 
    cout << "counting ";

    // consider 2 or 3 dimensions
    bool dim3 = false; 
    if(this->bitBlock.imageDepth > 1) 
        dim3 = true;

    // set cube limits
    uint x0, y0, z0;
    uint xf, yf, zf;

    // coordinate x:
    if(_vertex1.x < _vertex2.x) 
    { 
        x0 = _vertex1.x; xf = _vertex2.x;
    } else
    {
        x0 = _vertex2.x;    xf = _vertex1.x;
    }

    // coordinate y:
    if(_vertex1.y < _vertex2.y)
    {
        y0 = _vertex1.y;    yf = _vertex2.y;
    } else
    {
        y0 = _vertex1.y;    yf = _vertex2.y;
    }

    // coordinate z:
    if(_vertex1.z < _vertex2.z)
    {
        z0 = _vertex1.z;    zf = _vertex2.z;
    } else
    {
        z0 = _vertex1.z;    zf = _vertex2.z;
    }

    // apply image border restrictions
    if(x0 < 0) x0 = 0;
    if(y0 < 0) y0 = 0;
    if(z0 < 0) z0 = 0;
    if(xf > this->bitBlock.imageColumns) xf = this->bitBlock.imageColumns;
    if(yf > this->bitBlock.imageRows) yf = this->bitBlock.imageRows;
    if(zf > this->bitBlock.imageDepth) zf = this->bitBlock.imageDepth;

    // first, count all pores in image
    this->numberOfPores = 0;
    for(uint z = z0; z < zf; z++)
    {
        for(uint y = y0; y < yf; y++)
        {
            for(uint x = x0; x < xf; x++)
            {
                int block, bit;
                if(dim3 == true)
                {
                    block = this->bitBlock.findBlock(x, y, z);
                    bit = this->bitBlock.findBitInBlock(x, y, z);
                } else
                {
                    block = this->bitBlock.findBlock(x, y);
                    bit = this->bitBlock.findBitInBlock(x, y);                    
                }

                // now check if bit is pore or wall
                if (!this->bitBlock.checkIfBitIsWall(block, bit))
                {
                    this->numberOfPores++;
                }
            }
        }       
    }

    time = omp_get_wtime() - time;
    cout << this->numberOfPores << " pore voxels in cubic space...Ok. (" << time << " seconds)." << endl;  
}

void NMR_Simulation::updatePorosity()
{
    this->porosity = (double) this->numberOfPores / 
                     (double) (this->bitBlock.imageColumns * this->bitBlock.imageRows * this->bitBlock.imageDepth);
}

void NMR_Simulation::updateNumberOfPores()
{
    this->numberOfPores = (uint) (this->porosity * (double) (this->bitBlock.imageColumns * this->bitBlock.imageRows * this->bitBlock.imageDepth));
}

void NMR_Simulation::createPoreList()
{
    double time = omp_get_wtime(); 
    cout << "creating pore list...";

    // consider 2 or 3 dimensions
    bool dim3 = false; 
    if(this->bitBlock.imageDepth > 1) 
        dim3 = true;

    // second, create pore list
    (*this).updateNumberOfPores();
    if(this->pores.size() > 0) this->pores.clear();
    this->pores.reserve(this->numberOfPores);
    for(uint z = 0; z < this->bitBlock.imageDepth; z++)
    {
        for(uint y = 0; y < this->bitBlock.imageRows; y++)
        {
            for(uint x = 0; x < this->bitBlock.imageColumns; x++)
            {
                int block, bit;
                if(dim3 == true)
                {
                    block = this->bitBlock.findBlock(x, y, z);
                    bit = this->bitBlock.findBitInBlock(x, y, z);
                } else
                {
                    block = this->bitBlock.findBlock(x, y);
                    bit = this->bitBlock.findBitInBlock(x, y);                    
                }

                if (!this->bitBlock.checkIfBitIsWall(block, bit))
                {
                    Pore detectedPore = {x, y, z};
                    this->pores.push_back(detectedPore);
                }
            }
        }       
    }

    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}

void NMR_Simulation::createPoreList(Point3D _vertex1, Point3D _vertex2)
{
    double time = omp_get_wtime(); 
    cout << "creating pore list...";

    // consider 2 or 3 dimensions
    bool dim3 = false; 
    if(this->bitBlock.imageDepth > 1) 
        dim3 = true;

    // set cube limits
    uint x0, y0, z0;
    uint xf, yf, zf;

    // coordinate x:
    if(_vertex1.x < _vertex2.x) 
    { 
        x0 = _vertex1.x; xf = _vertex2.x;
    } else
    {
        x0 = _vertex2.x;    xf = _vertex1.x;
    }

    // coordinate y:
    if(_vertex1.y < _vertex2.y)
    {
        y0 = _vertex1.y;    yf = _vertex2.y;
    } else
    {
        y0 = _vertex1.y;    yf = _vertex2.y;
    }

    // coordinate z:
    if(_vertex1.z < _vertex2.z)
    {
        z0 = _vertex1.z;    zf = _vertex2.z;
    } else
    {
        z0 = _vertex1.z;    zf = _vertex2.z;
    }

    // apply image border restrictions
    if(x0 < 0) x0 = 0;
    if(y0 < 0) y0 = 0;
    if(z0 < 0) z0 = 0;
    if(xf > this->bitBlock.imageColumns) xf = this->bitBlock.imageColumns;
    if(yf > this->bitBlock.imageRows) yf = this->bitBlock.imageRows;
    if(zf > this->bitBlock.imageDepth) zf = this->bitBlock.imageDepth;

    // second, create pore list
    if(this->pores.size() > 0) this->pores.clear();
    this->pores.reserve(this->numberOfPores);
    for(uint z = z0; z < zf; z++)
    {
        for(uint y = y0; y < yf; y++)
        {
            for(uint x = x0; x < xf; x++)
            {
                int block, bit;
                if(dim3 == true)
                {
                    block = this->bitBlock.findBlock(x, y, z);
                    bit = this->bitBlock.findBitInBlock(x, y, z);
                } else
                {
                    block = this->bitBlock.findBlock(x, y);
                    bit = this->bitBlock.findBitInBlock(x, y);                    
                }

                if (!this->bitBlock.checkIfBitIsWall(block, bit))
                {
                    Pore detectedPore = {x, y, z};
                    this->pores.push_back(detectedPore);
                }
            }
        }       
    }

    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}

void NMR_Simulation::setNumberOfWalkers(uint _numberOfWalkers)
{   
    if(_numberOfWalkers)
    {   
        if(this->numberOfPores == 0) (*this).countPoresInBitBlock();
        this->numberOfWalkers = _numberOfWalkers;
    }
    else
    {
        if(this->numberOfPores == 0) (*this).countPoresInBitBlock();
        this->numberOfWalkers = (uint)(walkerOccupancy * numberOfPores);
    }
}

void NMR_Simulation::updateWalkerOccupancy()
{   
    this->walkerOccupancy = (this->numberOfWalkers / ((double) this->numberOfPores)); 
}

void NMR_Simulation::createWalkersIDList()
{
    double time = omp_get_wtime();
    cout << "creating list of " << this->numberOfWalkers << " random walkers ";

    if(this->walkersIDList.size() > 0) this->walkersIDList.clear();
    this->walkersIDList.reserve(this->numberOfWalkers);

    if(this->pores.size() == 0)
    {
        cout << endl;
        cout << "pores not listed." << endl;
        return;
    }

    if(this->walkerOccupancy < 1.0)
    {
        // create pore pool
        vector<uint> porePool;
        porePool.reserve(this->numberOfPores);
        for(uint idx = 0; idx < this->numberOfPores; idx++)
        {
            porePool.push_back(idx);
        }

        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->numberOfWalkers;
        int loop_start, loop_finish;
        cout << "using " << num_cpu_threads << " cpu threads...";

        #pragma omp parallel shared(walkersIDList) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish(); 

            for (uint i = loop_start; i < loop_finish; i++)
            {
                // ramdomly choose a pore location for the new walker
                uint nextPoreIndex = (*this).removeRandomIndexFromPool(porePool, (*this).pickRandomIndex(this->numberOfPores - i));
                #pragma omp critical
                {
                    this->walkersIDList.push_back(nextPoreIndex);
                }
            }
        }
    } 

    if(this->walkerOccupancy == 1.0)
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->numberOfWalkers;
        int loop_start, loop_finish;
        cout << "using " << num_cpu_threads << " cpu threads...";

        #pragma omp parallel shared(walkersIDList) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish();  
            
            for(uint idx = loop_start; idx < loop_finish; idx++)
            {
                #pragma omp critical
                {
                    this->walkersIDList.push_back(idx);
                }
            }
        }
    }

    // case 3
    if(this->walkerOccupancy > 1.0)
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->numberOfWalkers;
        int loop_start, loop_finish;
        cout << "using " << num_cpu_threads << " cpu threads...";

        #pragma omp parallel shared(walkersIDList) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish();  

            for (uint idx = loop_start; idx < loop_finish; idx++)
            {
                // ramdomly choose a pore location for the new walker with no restrictions
                uint nextPoreIndex = (*this).pickRandomIndex((*this).getNumberOfPores() - 1);
                #pragma omp critical
                {
                    this->walkersIDList.push_back(nextPoreIndex);
                }
            } 
        }
    }

    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}


void NMR_Simulation::createWalkers()
{
    double time = omp_get_wtime();
    cout << "creating " << this->numberOfWalkers << " walkers...";

    // define the dimensionality that walkers will be trated
    bool dim3 = false;
    if (this->bitBlock.imageDepth > 1)
    {
        dim3 = true;
    }

    // alloc memory space for vector of walkers with size = numberOfWalkers
    if(this->walkers.size() > 0) this->walkers.clear();
    this->walkers.reserve(this->numberOfWalkers);
    uint64_t tempSeed = this->initialSeed + 1;

    // create walkers
    for (uint idx = 0; idx < this->numberOfWalkers; idx++)
    {
        Walker temporaryWalker(dim3);
        this->walkers.push_back(temporaryWalker);
        this->walkers[idx].setSurfaceRelaxivity(DEFAULT_RELAXATIVITY);
        this->walkers[idx].computeDecreaseFactor(this->imageVoxelResolution, this->diffusionCoefficient);
        
        // set initial seed
        if (this->seedFlag != true)
        {
            this->walkers[idx].createRandomSeed();
        }
        else // seed was defined by user
        {
            // scramble some bits in the original seed
            tempSeed ^= tempSeed >> 12;
            tempSeed ^= tempSeed << 25;
            tempSeed ^= tempSeed >> 27;

            this->walkers[idx].setInitialSeed(tempSeed);
            this->walkers[idx].resetSeed();
        }
    }
    
    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}


void NMR_Simulation::placeWalkersByChance()
{

    double time = omp_get_wtime();
    cout << "placing walkers by chance...";

    if(numberOfPores == 0)
    {
        cout << endl;
        cout << "pores not counted." << endl;
        return;
    }

    // define the dimensionality that walkers will be trated
    bool dim3 = false;
    if (this->bitBlock.imageDepth > 1)
    {
        dim3 = true;
    }    

    uint walkersInserted = 0;
    uint errorCount = 0;
    uint erroLimit = 1000;
    uint idx = 0;
    Point3D point; 
    bool validPoint = false;   

    while(walkersInserted < this->numberOfWalkers && errorCount < erroLimit)
    {
        // randomly choose a position
        point.x = (*this).pickRandomIndex(this->bitBlock.imageColumns - 1);
        point.y = (*this).pickRandomIndex(this->bitBlock.imageRows - 1);
        point.z = (*this).pickRandomIndex(this->bitBlock.imageDepth - 1);
        if(dim3)
        {
            validPoint = walkers[idx].checkNextPosition_3D(point, this->bitBlock);        
        } else
        {
            validPoint = walkers[idx].checkNextPosition_2D(point, this->bitBlock);
        }

        if(validPoint)
        {
            this->walkers[idx].placeWalker(point.x, point.y, point.z);
            idx++;
            walkersInserted++;
            errorCount = 0;
        }
        else
        {
            errorCount++;
        }
    }
    
    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
    if(errorCount == erroLimit) cout << "could only insert " << walkersInserted << "walkers." << endl;
}


void NMR_Simulation::placeWalkersUniformly()
{
    double time = omp_get_wtime();
    cout << "placing walkers uniformly ";

    if(this->pores.size() == 0)
    {
        cout << endl;
        cout << "pores not listed." << endl;
        return;
    }

    // set omp variables for parallel loop throughout walker list
    const int num_cpu_threads = omp_get_max_threads();
    const int loop_size = this->numberOfWalkers;
    int loop_start, loop_finish;
    cout << "using " << num_cpu_threads << " cpu threads...";

    #pragma omp parallel shared(walkers, walkersIDList, pores) private(loop_start, loop_finish) 
    {
        const int thread_id = omp_get_thread_num();
        OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
        loop_start = looper.getStart();
        loop_finish = looper.getFinish();     


        if (this->walkerOccupancy != 1.0) 
        {
            for (uint idx = loop_start; idx < loop_finish; idx++)
            {
                // select a pore location from a list randomly generated to place the walker
                uint randomIndex = this->walkersIDList[idx];
                this->walkers[idx].placeWalker(this->pores[randomIndex].position_x, 
                                               this->pores[randomIndex].position_y, 
                                               this->pores[randomIndex].position_z);
            }
        }
        else
        {
            // If occupancy is 100%, just loop over pore list
            for (uint idx = loop_start; idx < loop_finish; idx++)
            {
                this->walkers[idx].placeWalker(this->pores[idx].position_x, 
                                               this->pores[idx].position_y, 
                                               this->pores[idx].position_z);
            }
        }

    }

    time = omp_get_wtime() - time;
    cout << "Ok. (" << time << " seconds)." << endl; 
}

void NMR_Simulation::placeWalkersInSamePoint(uint _x, uint _y, uint _z)
{
    double time = omp_get_wtime();
    cout << "placing walkers at point ";
    cout << "(" << _x << ", " << _y << ", " << _z << ")..."; 

    // check walkers array
    if(this->walkers.size() == 0)
    {
        cout << endl;
        cout << "no walkers to place" << endl;
        return;
    }

    // check image limits
    if(_x > this->bitBlock.imageRows || 
       _y > this->bitBlock.imageColumns || 
       _z > this->bitBlock.imageDepth)
    {
        cout << endl;
        cout << "could not place walkers: point is outside image limits." << endl;
        return;
    }

    // set dimensionality that walkers will be trated
    bool dim3 = false;
    if (this->bitBlock.imageDepth > 1)
    {
        dim3 = true;
    }

    // checar se ponto é poro
    Point3D point(_x, _y, _z);
    bool validPoint;    
    if(dim3)
    {
        validPoint = walkers[0].checkNextPosition_3D(point, this->bitBlock);        
    } else
    {
        validPoint = walkers[0].checkNextPosition_2D(point, this->bitBlock);
    }

    if(validPoint == false)
    {
        cout << endl;
        cout << "could not place walkers: point is not pore." << endl;
        return;
    } else
    {

        for(uint id = 0; id < this->walkers.size(); id++)
        {
            this->walkers[id].placeWalker(_x, _y, _z);
        }
    
        time = omp_get_wtime() - time;
        cout << "Ok. (" << time << " seconds)." << endl; 
    }
}

void NMR_Simulation::placeWalkersInCubicSpace(Point3D _vertex1, Point3D _vertex2)
{
    double time = omp_get_wtime();
    cout << "placing walkers at selected space: " << endl;
    cout << "vertex 1: "; _vertex1.printInfo();
    cout << "vertex 2: "; _vertex2.printInfo();


    // check walkers array
    if(this->walkers.size() == 0)
    {
        cout << endl;
        cout << "no walkers to place" << endl;
        return;
    }

    // create a list of pores in selected zone
    vector<uint> selectedPores;
    for(uint idx = 0; idx < this->pores.size(); idx++)
    {
        // check if pore is inside selected cube
        if(this->pores[idx].position_x >= _vertex1.x && this->pores[idx].position_x <= _vertex2.x &&
           this->pores[idx].position_y >= _vertex1.y && this->pores[idx].position_y <= _vertex2.y &&
           this->pores[idx].position_z >= _vertex1.z && this->pores[idx].position_z <= _vertex2.z)
        {
            // add pore to list
            selectedPores.push_back(idx);
        }
    }

    if(selectedPores.size() == 0)
    {
        cout << endl;
        cout << "no pores in the selected cubic space." << endl;
        return;
    }
    else
    {    
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->walkers.size();
        int loop_start, loop_finish;
        cout << "using " << num_cpu_threads << " cpu threads...";

        #pragma omp parallel shared(walkers, pores, selectedPores) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish();

            // pick random pores in selected list
            const uint listSize = selectedPores.size();
            for(uint id = loop_start; id < loop_finish; id++)
            {   
        
                uint poreID = selectedPores[(*this).pickRandomIndex(listSize)];
                this->walkers[id].placeWalker(this->pores[poreID].position_x, 
                                              this->pores[poreID].position_y, 
                                              this->pores[poreID].position_z);
            }
        }
    
        time = omp_get_wtime() - time;
        cout << "Ok. (" << time << " seconds)." << endl; 
    }
}

void NMR_Simulation::initHistogramList()
{   
    int numberOfHistograms = NMR_HISTOGRAMS;

    // check for really small simulations
    if(this->numberOfEchoes < numberOfHistograms) 
        numberOfHistograms = 1;

    int echosPerHistogram = (this->numberOfEchoes / numberOfHistograms);
    int echosInLastHistogram = this->numberOfEchoes - (numberOfHistograms * echosPerHistogram);
    if(this->histogramList.size() != 0) this->histogramList.clear();
    this->histogramList.reserve(numberOfHistograms);

    for(int idx = 0; idx < numberOfHistograms; idx++)
    {
        CollisionHistogram newHistogram;
        newHistogram.firstEcho = idx * echosPerHistogram;
        newHistogram.lastEcho = newHistogram.firstEcho + echosPerHistogram;
        this->histogramList.push_back(newHistogram);
    }

    // charge rebalance 
    this->histogramList.back().lastEcho += echosInLastHistogram;
}

void NMR_Simulation::createHistogram()
{
    // CollisionHistogram newHistogram(NMR_HISTOGRAM_SIZE);
    // newHistogram.fillHistogram(this->walkers, this->simulationSteps);
    // this->histogram = newHistogram;
    this->histogram.createBlankHistogram(NMR_HISTOGRAM_SIZE);
    int steps = this->histogramList.back().lastEcho * this->stepsPerEcho;
    this->histogram.fillHistogram(this->walkers, steps);       
}

void NMR_Simulation::createHistogram(uint histID, uint _steps)
{
    // CollisionHistogram newHistogram(NMR_HISTOGRAM_SIZE);
    // newHistogram.fillHistogram(this->walkers, _steps);
    // this->histogramList[histID].amps.assign(newHistogram.amps.begin(), newHistogram.amps.end());
    // this->histogramList[histID].bins.assign(newHistogram.bins.begin(), newHistogram.bins.end()); 

    this->histogramList[histID].createBlankHistogram(NMR_HISTOGRAM_SIZE);
    this->histogramList[histID].fillHistogram(this->walkers, _steps);       
}

// cost function methods
void NMR_Simulation::updateWalkersRelaxativity(vector<double> &sigmoid)
{
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        this->walkers[id].updateXIrate(this->simulationSteps);
        this->walkers[id].setSurfaceRelaxivity(sigmoid);
        this->walkers[id].computeDecreaseFactor(this->imageVoxelResolution, this->diffusionCoefficient);
    }
}

void NMR_Simulation::updateWalkersRelaxativity(double rho)
{
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        this->walkers[id].updateXIrate(this->simulationSteps);
        this->walkers[id].setSurfaceRelaxivity(rho);
        this->walkers[id].computeDecreaseFactor(this->imageVoxelResolution, this->diffusionCoefficient);
    }
}

void NMR_Simulation::normalizeEnergyDecay()
{
    // check if energy decay was done
    if(this->globalEnergy.size() == 0) 
    {
        cout << "no data available, could not apply inversion." << endl;
        return; 
    } 

    // normalize global energy signal
    double normalizer = 1.0 / this->globalEnergy[0];
    for(uint echo = 0; echo < this->globalEnergy.size(); echo++)
    {
        this->globalEnergy[echo] = normalizer * this->globalEnergy[echo]; 
    } 
}

// apply laplace inversion explicitly
void NMR_Simulation::applyLaplaceInversion()
{   
    // check if energy decay was done
    if(this->globalEnergy.size() == 0) 
    {
        cout << "no data available, could not apply inversion." << endl;
        return; 
    }

    // reset T2 distribution from previous simulation
    (*this).resetT2Distribution();

    NMRInverterConfig nmr_inv_config(0.1, 1e4, true, 128, -4, 2, 512, 512, 0.0);

    NMRInverter nmr_inverter;
    nmr_inverter.set_config(nmr_inv_config, this->decayTimes);
    nmr_inverter.find_best_lambda(this->globalEnergy.size(), this->globalEnergy.data());
    nmr_inverter.invert(this->globalEnergy.size(), this->globalEnergy.data());

    if(this->T2_bins.size() > 0) this->T2_bins.clear();
    if(this->T2_simulated.size() > 0) this->T2_simulated.clear();
    for(uint i = 0; i < nmr_inverter.used_t2_bins.size(); i++)
    {
        this->T2_bins.push_back(nmr_inverter.used_t2_bins[i]);
        this->T2_simulated.push_back(nmr_inverter.used_t2_amps[i]);
    }
}



// Correlation between input and simulated T2 distributions using Francisco's formula
double NMR_Simulation::correlateT2()
{
    if(this->T2_input.size() == 0 || this->T2_simulated.size() == 0)
    {
        cout << "could not measure correlation without T2 curves" << endl;
        return 0.0;
    }

    double correlation = norm(this->T2_simulated) * norm(this->T2_input);

    if (correlation != 0)
    {
        correlation = dotProduct(this->T2_simulated, this->T2_input) / correlation;
        return correlation;
    }

    return 0.0;
}

// Correlation between input and simulated T2 distributions using Least Squares Normalization
double NMR_Simulation::leastSquaresT2()
{
    if(this->T2_input.size() == 0 || this->T2_simulated.size() == 0)
    {
        cout << "could not measure correlation without T2 curves" << endl;
        return 0.0;
    }

    double residual = 0.0;
    double distance;
    for(uint i = 0; i < this->T2_input.size(); i++)
    {
        distance = this->T2_input[i] - this->T2_simulated[i];
        residual += (distance * distance);
    }

    // treating a terrible solution
    if(residual > 1.0) 
    {
        residual = 1.0;
    }

    return (1.0 - residual);
}

// associate methods
void NMR_Simulation::associateMapSimulation()
{
    if (gpu_use == true)
    {
        if (this->numberOfImages == 1)
        {
            mapSimulationPointer = &NMR_Simulation::mapSimulation_CUDA_2D_histograms;
        }
        else
        {
            mapSimulationPointer = &NMR_Simulation::mapSimulation_CUDA_3D_histograms;
        }
    }
    else
        mapSimulationPointer = &NMR_Simulation::mapSimulation_OMP;
}

void NMR_Simulation::associateWalkSimulation()
{
    if (gpu_use == true)
    {
        if (this->numberOfImages == 1)
        {
            walkSimulationPointer = &NMR_Simulation::walkSimulation_CUDA_2D;
        }
        else
        {
            walkSimulationPointer = &NMR_Simulation::walkSimulation_CUDA_3D;
        }
    }
    else
        walkSimulationPointer = &NMR_Simulation::walkSimulation_OMP;
}

uint NMR_Simulation::pickRandomIndex(uint _maxValue)
{
    int CPUfactor = 1;
    if(NMR_OPENMP) CPUfactor += omp_get_thread_num();
    std::random_device dev;
    std::mt19937 rng(dev()* CPUfactor * CPUfactor);
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, _maxValue); 

    // RNG warm up
    for(int i = 0; i < 100; i++) dist(rng);

    return (dist(rng) - 1);
}

Pore NMR_Simulation::removeRandomPore(vector<Pore> &_pores, uint _randomIndex)
{
    Pore randomPore = _pores[_randomIndex];
    std::swap(_pores[_randomIndex], _pores.back());
    _pores.pop_back();
    return randomPore;
}

uint NMR_Simulation::removeRandomIndexFromPool(vector<uint> &_pool, uint _randomIndex)
{
    uint element = _pool[_randomIndex];
    std::swap(_pool[_randomIndex], _pool.back());
    _pool.pop_back();
    return element;
}