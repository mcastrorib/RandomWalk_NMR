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

// include configuration file classes
#include "../ConfigFiles/rwnmr_config.h"
#include "../ConfigFiles/uct_config.h"

//include
#include "../Walker/walker.h"
#include "../RNG/myRNG.h"
#include "../BitBlock/bitBlock.h"
#include "../RNG/xorshift.h"
#include "../FileHandler/fileHandler.h"
#include "../Laplace/tikhonov.h"
#include "../Laplace/include/nmrinv_core.h"
#include "NMR_Simulation.h"
#include "ChordLengthHistogram.h"
#include "../Utils/OMPLoopEnabler.h"
#include "../Utils/ImagePath.h"
#include "../Utils/ProgressBar.h"
#include "../RNG/randomIndex.h"



using namespace cv;
using namespace std;

NMR_Simulation::NMR_Simulation(rwnmr_config _rwNMR_config, 
                               uct_config _uCT_config,
                               string _project_root) :   rwNMR_config(_rwNMR_config),
                                                         uCT_config(_uCT_config),
                                                         simulationSteps(0),
                                                         numberOfEchoes(0),
                                                         numberOfImages(0),
                                                         width(0),
                                                         height(0),
                                                         depth(0),
                                                         boundaryCondition("noflux"),
                                                         numberOfPores(0),
                                                         porosity(-1.0),
                                                         interfacePoreMatrix(0),
                                                         SVp(-1.0),
                                                         walkerOccupancy(-1.0),
                                                         voxelDivisionApplied(false),
                                                         penalties(NULL)
{
    // init vector objects
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
    this->DBPath = _project_root + "./db/";
    this->simulationName = this->rwNMR_config.getName();
    this->simulationDirectory = (*this).createDirectoryForResults(this->DBPath);

    // assign attributes from rwnmr config files
    (*this).setNumberOfWalkers(this->rwNMR_config.getWalkers());
    (*this).setWalkerSamples(this->rwNMR_config.getWalkerSamples());
    (*this).setFreeDiffusionCoefficient(this->rwNMR_config.getD0());
    (*this).setNumberOfStepsPerEcho(this->rwNMR_config.getStepsPerEcho());
    (*this).setGPU(this->rwNMR_config.getGPUUsage());
    if(this->rwNMR_config.getSeed() == 0)
    {
        (*this).setInitialSeed(myRNG::RNG_uint64(), true);
    } else
    {
        (*this).setInitialSeed(this->rwNMR_config.getSeed(), true);
    }   

    // assign attributes from uct config files
    (*this).setImageResolution(this->uCT_config.getResolution());

    (*this).setVoxelDivision(0);
    (*this).setImageVoxelResolution();
    if(this->uCT_config.getVoxelDivision() > 0) this->voxelDivisionApplied = true; 

    // set default time step measurement
    (*this).setTimeInterval(); 
    (*this).setBoundaryCondition(this->rwNMR_config.getBC());
}

void NMR_Simulation::setImage(ImagePath _path, uint _images)
{
    this->imagePath = _path;
    this->numberOfImages = _images;
    this->depth = this->numberOfImages;
}

void NMR_Simulation::setSimulation(double _occupancy, uint64_t _seed, bool _use_GPU)
{ 
    this->gpu_use = _use_GPU;
    this->walkerOccupancy = _occupancy;
    this->initialSeed = _seed; 
    if (this->initialSeed != this->rwNMR_config.getSeed())
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

void NMR_Simulation::setInitialSeed(uint64_t _seed, bool _flag)
{
    this->initialSeed = _seed; 
    this->seedFlag = _flag;
}

void NMR_Simulation::setBoundaryCondition(string _bc)
{
    this->boundaryCondition = _bc;
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
    if(this->voxelDivision > 1) this->voxelDivisionApplied = true;
    else this->voxelDivisionApplied = false;
}

void NMR_Simulation::applyVoxelDivision(uint _shifts)
{
    double time = omp_get_wtime();
    cout << "- applying voxel division:" << endl;

    // reset resolution scales
    uint previousDivision = (*this).getVoxelDivision();
    (*this).setVoxelDivision(_shifts);
    (*this).setImageVoxelResolution();

    // // trying to maintain steps per echo
    (*this).setTimeInterval();
    uint steps = (*this).getStepsPerEcho() * (*this).getNumberOfEchoes();
    (*this).setTimeFramework(steps);

    // if walkers exists, fill intra-voxel sites //
    if(this->walkers.size() > 0)
    {
        double shiftFactor = (*this).getVoxelDivision() / (double) previousDivision;
        uint indexExpansion = (uint) shiftFactor - 1;
        if(indexExpansion < 0) indexExpansion = 0;
        
        int shiftX, shiftY, shiftZ;
        RandomIndex rIndex(0, indexExpansion);
        ProgressBar pBar((double) this->walkers.size());
        for(uint idx = 0; idx < this->walkers.size(); idx++)
        {   
            // randomly place walker in voxel sites
            shiftX = ((int) this->walkers[idx].initialPosition.x * shiftFactor) + rIndex();
            shiftY = ((int) this->walkers[idx].initialPosition.y * shiftFactor) + rIndex();
            shiftZ = ((int) this->walkers[idx].initialPosition.z * shiftFactor) + rIndex();
            this->walkers[idx].placeWalker(shiftX, shiftY, shiftZ);

            // update collision penalty
            this->walkers[idx].computeDecreaseFactor(this->imageVoxelResolution, this->diffusionCoefficient);

            // update progress bar
            pBar.update(1);
            pBar.print();
        }
    }

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
}

void NMR_Simulation::setNumberOfStepsPerEcho(uint _stepsPerEcho)
{
    this->stepsPerEcho = _stepsPerEcho;
}

// param @_time needs to be in miliseconds
void NMR_Simulation::setNumberOfStepsFromTime(double _time)
{
    // _time = _time;
    this->simulationSteps =  round( _time * (6 * this->diffusionCoefficient / (this->imageVoxelResolution * this->imageVoxelResolution))); 
    
    // correct steps < steps per echo case (simulation becomes inaccurate but at least don't crash)
    if(this->simulationSteps < this->stepsPerEcho) this->simulationSteps = this->stepsPerEcho;
}

void NMR_Simulation::setTimeFramework(uint _steps)
{
    if(_steps < this->stepsPerEcho) _steps = this->stepsPerEcho;
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
    if(this->uCT_config.IMG_FILES.size() == this->uCT_config.SLICES)
    {
        this->numberOfImages = this->uCT_config.SLICES;
        this->depth = this->uCT_config.SLICES;
        (*this).loadRockImageFromList();
    }
    else
    {
        (*this).assemblyImagePath();
        (*this).loadRockImage();
    } 

    (*this).createBitBlockMap();
    (*this).countPoresInBitBlock();
    (*this).countInterfacePoreMatrix();

    ChordLengthHistogram chordsHistogram(this->bitBlock);
    chordsHistogram.save(this->simulationDirectory);
}

void NMR_Simulation::setWalkers(void)
{
    if(this->bitBlock.getNumberOfBlocks() > 0) // only set walkers, if image was loaded
    {
        if(this->rwNMR_config.getWalkersPlacement() == "point")
        { 
            // define coords of central point
            int center_x = (int) (*this).getImageWidth()/2;
            int center_y = (int) (*this).getImageHeight()/2;
            int center_z = (int) (*this).getImageDepth()/2;
            Point3D center(center_x, center_y, center_z);

            // now set walkers
            (*this).setWalkers(center, this->rwNMR_config.getWalkers());
        } else 
        if (this->rwNMR_config.getWalkersPlacement() == "cubic")
        {   
            // define restriction points
            int center_x = (int) ((*this).getImageWidth() - 1)/2;
            int center_y = (int) ((*this).getImageHeight() - 1)/2;
            int center_z = (int) ((*this).getImageDepth() - 1)/2;
            int deviation = (int) this->rwNMR_config.getPlacementDeviation();
            Point3D point1(center_x - deviation, center_y - deviation, center_z - deviation);
            Point3D point2(center_x + deviation, center_y + deviation, center_z + deviation);

            // now set walkers 
            (*this).setWalkers(point1, point2, this->rwNMR_config.getWalkers());
        } else
        {
            (*this).setWalkers(this->rwNMR_config.getWalkers());
        }  
        
        // apply voxel division if needed
        if(this->voxelDivisionApplied) (*this).applyVoxelDivision(this->uCT_config.getVoxelDivision());
    } else
    {
        cout << "error: image was not loaded yet" << endl;
    }
    
}

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
}

// save results
void NMR_Simulation::saveInfo()
{
    
}

void NMR_Simulation::save()
{
    double time = omp_get_wtime();
    cout << "saving results:" << endl;

    ProgressBar pBar(2.0);

    if(this->rwNMR_config.getSaveImgInfo())
    {   
        (*this).saveImageInfo(this->simulationDirectory);
    } 

    pBar.update(1);
    pBar.print();

    if(this->rwNMR_config.getSaveBinImg())
    {   
        (*this).saveBitBlock(this->simulationDirectory);
    }

    pBar.update(1);
    pBar.print();

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
}

// save results in other directory
void NMR_Simulation::save(string _otherDir)
{
    double time = omp_get_wtime();
    cout << "saving results...";

    ProgressBar pBar(2.0);
    
    if(this->rwNMR_config.getSaveImgInfo())
    {   
        (*this).saveImageInfo(_otherDir);
    }

    pBar.update(1);
    pBar.print();

    if(this->rwNMR_config.getSaveBinImg())
    {   
        (*this).saveBitBlock(_otherDir);
    }

    pBar.update(1);
    pBar.print();

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
}

void NMR_Simulation::loadRockImage()
{
    double time = omp_get_wtime();
    cout << "- loading rock image:" << endl;

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

    // create progress bar object
    ProgressBar pBar((double) (this->numberOfImages));

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

        // Update progress bar
        pBar.update(1);
        pBar.print();
        
    }

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl;
}

void NMR_Simulation::loadRockImageFromList()
{
    double time = omp_get_wtime();
    cout << "- loading rock image from list:" << endl;

    // reserve memory for binaryMap
    this->binaryMap.reserve(this->numberOfImages);

    // variable strings
    string currentImagePath;

    // create progress bar object
    ProgressBar pBar((double) this->numberOfImages);

    for (uint slice = 0; slice < this->numberOfImages; slice++)
    {
        currentImagePath = this->uCT_config.getImgFile(slice);

        Mat rockImage = imread(currentImagePath, 1);

        if (!rockImage.data)
        {
            cout << "Error: No image data in file " << currentImagePath << endl;
            exit(1);
        }

        this->height = rockImage.rows;
        this->width = rockImage.cols * rockImage.channels();

        (*this).createBinaryMap(rockImage, slice);

        // Update progress bar
        pBar.update(1);
        pBar.print();
    }

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl;
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
    cout << "- creating (bit)block map:" << endl;
    this->bitBlock.createBlockMap(this->binaryMap);

    // update image info
    this->width = this->bitBlock.imageColumns;
    this->height = this->bitBlock.imageRows;
    this->depth = this->bitBlock.imageDepth;
    this->binaryMap.clear();

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl;
}

// deprecated
void NMR_Simulation::countPoresInBinaryMap()
{
    double time = omp_get_wtime(); 
    cout << "counting ";

    for (int slice = 0; slice < this->numberOfImages; slice++)
    { // accept only char type matrices
        CV_Assert(binaryMap[slice].depth() == CV_8U);

        uint mapWidth = (uint)binaryMap[slice].cols;
        uchar *binaryMapPixel;

        for (int row = 0; row < height; ++row)
        {
            // psosition pointer at first element in current row
            binaryMapPixel = binaryMap[slice].ptr<uchar>(row);

            for (int column = 0; column < mapWidth; column++)
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
    cout << "- counting pore voxels in rock image:" << endl;

    // consider 2 or 3 dimensions
    bool dim3 = false; 
    if(this->bitBlock.imageDepth > 1) 
        dim3 = true;

    // Create progress bar object
    ProgressBar pBar((double) (this->bitBlock.imageDepth));

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

        // Update progress bar
        pBar.update(1);
        pBar.print();      
    }

    (*this).updatePorosity();

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
    cout << this->numberOfPores << " pore voxel(s) identified. " ;
    cout << "porosity: " << this->porosity << endl;
}

void NMR_Simulation::countPoresInCubicSpace(Point3D _vertex1, Point3D _vertex2)
{
    double time = omp_get_wtime(); 
    cout << "- counting pore voxels in cubic space:" << endl;

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

    // Create progress bar object
    ProgressBar pBar((double) (zf - z0));  

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
        
        // Update progress bar
        pBar.update(1);
        pBar.print();       
    }

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
    cout << this->numberOfPores << " pore voxel(s) in cubic space identified." << endl; 
    cout << "porosity: " << this->porosity << endl; 
}

void NMR_Simulation::countInterfacePoreMatrix()
{
    double time = omp_get_wtime(); 
    cout << "- counting pore-matrix interface voxels in rock image:" << endl;

    // consider 2 or 3 dimensions
    bool dim3 = false; 
    if(this->bitBlock.imageDepth > 1) 
        dim3 = true;

    // Create progress bar object
    ProgressBar pBar((double) (this->bitBlock.imageDepth));

    // first, count all pores in image
    this->interfacePoreMatrix = 0;
    for(int z = 0; z < this->bitBlock.imageDepth; z++)
    {
        // neighbor Z location
        int nextZ = (z + 1) % this->bitBlock.imageDepth;

        for(int y = 0; y < this->bitBlock.imageRows; y++)
        {
            // neighbor Y location
            int nextY = (y + 1) % this->bitBlock.imageRows;

            for(int x = 0; x < this->bitBlock.imageColumns; x++)
            {
                // neighbor X location
                int nextX = (x + 1) % this->bitBlock.imageColumns;

                int currentBlock, currentBit;
                int xBlock, xBit;
                int yBlock, yBit;
                int zBlock, zBit;              

                if(dim3 == true)
                {
                    currentBlock = this->bitBlock.findBlock(x, y, z);
                    currentBit = this->bitBlock.findBitInBlock(x, y, z);

                    xBlock = this->bitBlock.findBlock(nextX, y, z);
                    xBit = this->bitBlock.findBitInBlock(nextX, y, z);
                    
                    yBlock = this->bitBlock.findBlock(x, nextY, z);
                    yBit = this->bitBlock.findBitInBlock(x, nextY, z);
                    
                    zBlock = this->bitBlock.findBlock(x, y, nextZ);
                    zBit = this->bitBlock.findBitInBlock(x, y, nextZ);


                } else
                {
                    currentBlock = this->bitBlock.findBlock(x, y);
                    currentBit = this->bitBlock.findBitInBlock(x, y);

                    xBlock = this->bitBlock.findBlock(nextX, y);
                    xBit = this->bitBlock.findBitInBlock(nextX, y);
                    
                    yBlock = this->bitBlock.findBlock(x, nextY);
                    yBit = this->bitBlock.findBitInBlock(x, nextY);
                                                            
                }

                // Check if neighbor bit values are different
                if (this->bitBlock.checkIfBitIsWall(currentBlock, currentBit) != this->bitBlock.checkIfBitIsWall(xBlock, xBit))
                {
                    this->interfacePoreMatrix++;
                }

                if (this->bitBlock.checkIfBitIsWall(currentBlock, currentBit) != this->bitBlock.checkIfBitIsWall(yBlock, yBit))
                {
                    this->interfacePoreMatrix++;
                }

                if (dim3 == true and this->bitBlock.checkIfBitIsWall(currentBlock, currentBit) != this->bitBlock.checkIfBitIsWall(yBlock, yBit))
                {
                    this->interfacePoreMatrix++;
                }
            }
        } 

        // Update progress bar
        pBar.update(1);
        pBar.print();      
    }

    (*this).updateSVp();

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
    cout << this->interfacePoreMatrix << " interface pore-matrix voxel(s) identified. " ;
    cout << "S/Vp: " << this->SVp << endl;
}

void NMR_Simulation::updateSVp()
{
    double voxelFacialArea = (*this).getImageVoxelResolution() * (*this).getImageVoxelResolution();
    double voxelVolume = voxelFacialArea * (*this).getImageVoxelResolution();
    this->SVp = ((double) this->interfacePoreMatrix * voxelFacialArea) / ((double) this->numberOfPores * voxelVolume);
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
    cout << "- creating pore list:" << endl;

    // consider 2 or 3 dimensions
    bool dim3 = false; 
    if(this->bitBlock.imageDepth > 1) 
        dim3 = true;

    // second, create pore list
    (*this).updateNumberOfPores();
    if(this->pores.size() > 0) this->pores.clear();
    this->pores.reserve(this->numberOfPores);

    // Create progress bar object
    ProgressBar pBar((double) (this->bitBlock.imageDepth));   

    for(int z = 0; z < this->bitBlock.imageDepth; z++)
    {
        for(int y = 0; y < this->bitBlock.imageRows; y++)
        {
            for(int x = 0; x < this->bitBlock.imageColumns; x++)
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

        // Update progress bar
        pBar.update(1);
        pBar.print();         
    }

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
}

void NMR_Simulation::createPoreList(Point3D _vertex1, Point3D _vertex2)
{
    double time = omp_get_wtime(); 
    cout << "- creating pore list:" << endl;

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

    // Create progress bar object
    ProgressBar pBar((double) (zf-z0));
    for(int z = z0; z < zf; z++)
    {
        for(int y = y0; y < yf; y++)
        {
            for(int x = x0; x < xf; x++)
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

        // Update progress bar
        pBar.update(1);
        pBar.print();          
    }

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
}

void NMR_Simulation::setNumberOfWalkers(uint _numberOfWalkers)
{   
    this->numberOfWalkers = _numberOfWalkers;
}

void NMR_Simulation::setWalkerSamples(uint _samples)
{   
    this->walkerSamples = _samples;
}

void NMR_Simulation::updateWalkerOccupancy()
{   
    this->walkerOccupancy = (this->numberOfWalkers / ((double) this->numberOfPores)); 
}

void NMR_Simulation::createWalkersIDList()
{
    double time = omp_get_wtime();
    cout << "- creating list of " << this->numberOfWalkers << " random walkers ";

    if(this->walkersIDList.size() > 0) this->walkersIDList.clear();
    this->walkersIDList.reserve(this->numberOfWalkers);

    if(this->pores.size() == 0)
    {
        cout << endl;
        cout << "pores not listed." << endl;
        return;
    }

    /*
        case 1 - set exactly one random walker at each pore voxel in image
    */
    if(this->walkerOccupancy == 1.0)
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->numberOfWalkers;
        int loop_start, loop_finish;
        cout << "using " << num_cpu_threads << " cpu threads:" << endl;

        // create Progress Bar object
        ProgressBar pBar((double) num_cpu_threads);
        pBar.print();

        #pragma omp parallel shared(walkersIDList, pBar) private(loop_start, loop_finish) 
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

            #pragma omp critical
            {
                pBar.update(1);
                pBar.print();
            }
        }
    }

    /* 
        case 3 - for each walker, choose a pore voxel randomly 
    */
    // if(this->walkerOccupancy > 1.0)
    if(this->walkerOccupancy != 1.0)
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->numberOfWalkers;
        int loop_start, loop_finish;
        cout << "using " << num_cpu_threads << " cpu threads:" << endl;

        // create Progress Bar object
        ProgressBar pBar((double) num_cpu_threads);
        pBar.print();

        const uint listSize = (*this).getNumberOfPores();
        vector<RandomIndex> generators;
        for(uint i = 0; i < num_cpu_threads; i++)
        {
            RandomIndex rnew(0, listSize - 1, i);
            generators.emplace_back(rnew);
        }

        #pragma omp parallel shared(walkersIDList, generators, pBar) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish();              

            for (uint idx = loop_start; idx < loop_finish; idx++)
            {
                // ramdomly choose a pore location for the new walker with no restrictions
                uint nextPoreIndex = generators[thread_id]();
                #pragma omp critical
                {
                    this->walkersIDList.push_back(nextPoreIndex);
                }
            } 

            #pragma omp critical
            {
                pBar.update(1);
                pBar.print();
            }
        }
    }

    /* 
    case 3: 
        for each walker, choose a pore voxel randomly but never allow two walkers in same voxel
        - it needs to be revisited, a bug was found in the random picking procedure - try using 
        RandomIndex class instead. 
        For now, all the cases where occupancy != 1.0 will be branched to case 2
    */
    bool debuged = false; 
    if(this->walkerOccupancy < 1.0 and debuged == true)
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
        cout << "using " << num_cpu_threads << " cpu threads:" << endl;

        // create Progress Bar object
        ProgressBar pBar((double) num_cpu_threads);
        pBar.print();

        #pragma omp parallel shared(walkersIDList, pBar) private(loop_start, loop_finish) 
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

            #pragma omp critical
            {
                pBar.update(1);
                pBar.print();
            }
        }
    }

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
}


void NMR_Simulation::createWalkers()
{
    double time = omp_get_wtime();
    cout << "- creating " << this->numberOfWalkers << " walkers:" << endl;

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
    vector<double> rho;
    rho = this->rwNMR_config.getRho();

    uint walkerPacks = 10;
    uint packSize = this->numberOfWalkers / walkerPacks;
    uint lastPackSize = this->numberOfWalkers % walkerPacks;

    // create progress bar object
    ProgressBar pBar(10);
    uint idx = 0;
    uint count = 0;
    for (uint pack = 0; pack < (walkerPacks - 1); pack++)
    {
        for (uint i = 0; i < packSize; i++)
        {
            idx = pack * packSize + i;
            Walker temporaryWalker(dim3);
            this->walkers.push_back(temporaryWalker);
            if(this->rwNMR_config.getRhoType() == "uniform") this->walkers[idx].setSurfaceRelaxivity(rho[0]);
            else if(this->rwNMR_config.getRhoType() == "sigmoid") this->walkers[idx].setSurfaceRelaxivity(rho);
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

            count++;
        }

        // Update progress bar
        pBar.update(1);
        pBar.print();
    }

    for (uint i = 0; i < (packSize + lastPackSize); i++)
    {
        idx = (walkerPacks - 1) * packSize + i;
        Walker temporaryWalker(dim3);
        this->walkers.push_back(temporaryWalker);
        if(this->rwNMR_config.getRhoType() == "uniform") this->walkers[idx].setSurfaceRelaxivity(rho[0]);
        else if(this->rwNMR_config.getRhoType() == "sigmoid") this->walkers[idx].setSurfaceRelaxivity(rho);
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

        count++;
    }

    // Update progress bar
    pBar.update(1);
    pBar.print();
    
    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
}


void NMR_Simulation::placeWalkersByChance()
{

    double time = omp_get_wtime();
    cout << "- placing walkers by chance...";

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
    
    RandomIndex columnRandomIndex(0, this->bitBlock.imageColumns - 1);
    RandomIndex rowRandomIndex(0, this->bitBlock.imageRows - 1);
    RandomIndex depthRandomIndex(0, this->bitBlock.imageDepth - 1);
    
    while(walkersInserted < this->numberOfWalkers && errorCount < erroLimit)
    {
        // randomly choose a position
        point.x = columnRandomIndex();
        point.y = rowRandomIndex();
        point.z = depthRandomIndex();
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
    if(errorCount == erroLimit) cout << "could only insert " << endl;
}


void NMR_Simulation::placeWalkersUniformly()
{
    double time = omp_get_wtime();
    cout << "- placing walkers uniformly ";

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
    cout << "using " << num_cpu_threads << " cpu threads:" << endl;

    ProgressBar pBar((double) num_cpu_threads);
    pBar.print();

    #pragma omp parallel shared(walkers, walkersIDList, pores, pBar) private(loop_start, loop_finish) 
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

        #pragma omp critical
        {
            // update progress bar
            pBar.update(1);
            pBar.print();
        }

    }

    time = omp_get_wtime() - time;
    cout << " in " << time << " seconds." << endl; 
}

void NMR_Simulation::placeWalkersInSamePoint(uint _x, uint _y, uint _z)
{
    double time = omp_get_wtime();
    cout << "- placing walkers at point ";
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
    cout << "- placing walkers at selected space: " << endl;
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
        cout << "using " << num_cpu_threads << " cpu threads:" << endl;

        const uint listSize = selectedPores.size();
        vector<RandomIndex> generators;
        for(uint i = 0; i < num_cpu_threads; i++)
        {
            RandomIndex rnew(0, listSize - 1, i);
            generators.emplace_back(rnew);
        }

        // Create Progress Bar object
        ProgressBar pBar((double) num_cpu_threads);
        pBar.print();

        #pragma omp parallel shared(walkers, pores, selectedPores, generators, pBar) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish();

            // pick random pores in selected list
            const uint listSize = selectedPores.size();
            for(uint id = loop_start; id < loop_finish; id++)
            {   
        
                uint poreID = selectedPores[generators[thread_id]()];
                this->walkers[id].placeWalker(this->pores[poreID].position_x, 
                                              this->pores[poreID].position_y, 
                                              this->pores[poreID].position_z);
            }

            // update progress bar
            #pragma omp critical
            {
                pBar.update(1);
                pBar.print();
            }
        }
    
        time = omp_get_wtime() - time;
        cout << " in " << time << " seconds." << endl; 
    }
}

void NMR_Simulation::initHistogramList()
{   
    int numberOfHistograms = this->rwNMR_config.getHistograms();

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
    this->histogram.createBlankHistogram(this->rwNMR_config.getHistogramSize());
    int steps = this->histogramList.back().lastEcho * this->stepsPerEcho;
    this->histogram.fillHistogram(this->walkers, steps);       
}

void NMR_Simulation::createHistogram(uint histID, uint _steps)
{
    // CollisionHistogram newHistogram(NMR_HISTOGRAM_SIZE);
    // newHistogram.fillHistogram(this->walkers, _steps);
    // this->histogramList[histID].amps.assign(newHistogram.amps.begin(), newHistogram.amps.end());
    // this->histogramList[histID].bins.assign(newHistogram.bins.begin(), newHistogram.bins.end()); 

    this->histogramList[histID].createBlankHistogram(this->rwNMR_config.getHistogramSize());
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
        cout << "no data available, could not apply normalization." << endl;
        return; 
    } 

    // normalize global energy signal
    double normalizer = 1.0 / this->globalEnergy[0];
    for(uint echo = 0; echo < this->globalEnergy.size(); echo++)
    {
        this->globalEnergy[echo] = normalizer * this->globalEnergy[echo]; 
    } 
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
            if((*this).getBoundaryCondition() == "periodic") 
                mapSimulationPointer = &NMR_Simulation::mapSimulation_CUDA_3D_histograms_periodic;
            else
                mapSimulationPointer = &NMR_Simulation::mapSimulation_CUDA_3D_histograms;
        }
    }
    else
        mapSimulationPointer = &NMR_Simulation::mapSimulation_OMP;
}

uint NMR_Simulation::pickRandomIndex(uint _maxValue)
{
    int CPUfactor = 1;
    if(this->rwNMR_config.getOpenMPUsage()) CPUfactor += omp_get_thread_num();
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

void NMR_Simulation::assemblyImagePath()
{
    // User Input
     ImagePath input;
     input.images = this->uCT_config.getSlices();
     input.path = this->uCT_config.getDirPath();
     input.filename = this->uCT_config.getFilename();
     input.fileID = this->uCT_config.getFirstIdx();
     input.digits = this->uCT_config.getDigits();
     input.extension = this->uCT_config.getExtension();
     input.updateCompletePath();

     (*this).setImage(input, input.images);
}

string NMR_Simulation::createDirectoryForResults(string _path)
{
    createDirectory(_path, this->simulationName);
    return (_path + this->simulationName);
}

void NMR_Simulation::saveImageInfo(string filedir)
{
    string filename = filedir + "/ImageInfo.txt";

    fileHandler external_file(filename);
    external_file.writeImageInfo(this->imagePath.completePath, 
                                 this->bitBlock.imageColumns, 
                                 this->bitBlock.imageRows, 
                                 this->bitBlock.imageDepth, 
                                 this->imageVoxelResolution,
                                 this->porosity,
                                 this->SVp);
}

void NMR_Simulation::saveEnergyDecay(string filePath)
{

    string fileName = filePath + "/NMR_decay.txt";

    fileHandler external_file(fileName);
    external_file.writeEnergyDecay(this->globalEnergy, this->decayTimes);
}

void NMR_Simulation::saveWalkerCollisions(string filePath)
{

    string fileName = filePath + "/NMR_collisions.txt";

    fileHandler external_file(fileName);
    external_file.writeIndividualCollisions(this->walkers, this->simulationSteps);
}

void NMR_Simulation::saveBitBlock(string filePath)
{

    string fileName = filePath + "/NMR_binImage.txt";
    this->bitBlock.saveBitBlockArray(fileName);
}

void NMR_Simulation::saveHistogram(string filePath)
{
    string filename = filePath + "/NMR_histogram.txt";
    fileHandler external_file(filename);
    external_file.writeHistogram(this->histogram.bins, this->histogram.amps);
}

void NMR_Simulation::saveHistogramList(string filePath)
{
    string filename = filePath + "/NMR_histogramEvolution.txt";
    fileHandler hfile(filename);
    if(this->rwNMR_config.getHistograms() > 0)
    {
        for(int hst_ID = 0; hst_ID < this->histogramList.size(); hst_ID++)
        {
            hfile.writeHistogramFromList(this->histogramList[hst_ID].bins, this->histogramList[hst_ID].amps, hst_ID);
        }
    }
}

void NMR_Simulation::info()
{
    (*this).printDetails();
}

void NMR_Simulation::printDetails()
{ 
    // print input details
    cout << "------------------------------------------------------" << endl;
    cout << ">>> NMR SIMULATION 3D PARAMETERS: " << this->simulationName << endl;
    cout << "------------------------------------------------------" << endl;
    cout << "Data path: " << this->DBPath + this->simulationName << endl;
    cout << "Image path: " << this->imagePath.completePath << endl;
    cout << "Image resolution (um/voxel): " << this->imageVoxelResolution << endl;
    cout << "Diffusion coefficient (um^2/ms): " << this->diffusionCoefficient << endl;
    cout << "Number of images: " << this->numberOfImages << endl;
    cout << "Number of steps in simulation: " << this->simulationSteps << endl;
    cout << "Decay time (ms): " << this->simulationSteps * this->timeInterval << endl;
    cout << "Walkers pore occupancy in simulation: " << this->walkerOccupancy * 100.0 << "%" << endl;
    

    cout << "Initial seed: ";
    if (this->seedFlag)
    {
        cout << this->initialSeed << endl;
    }
    else
    {
        cout << this->initialSeed << "\tdefined by user" << endl;
    }

    cout << "GPU usage: ";
    if (this->gpu_use)
    {
        cout << "ON" << endl;
    }
    else
    {
        cout << "OFF" << endl;
    }

    cout << "BC: " << (*this).getBoundaryCondition() << endl;

    cout << "------------------------------------------------------" << endl;
    cout << endl;
}

void NMR_Simulation::createPenaltiesVector(vector<double> &_sigmoid)
{
    // initialize penalties array
    if(this->penalties == NULL)
        this->penalties = new double[this->histogram.size];
    
    Walker toy;
    double artificial_xirate;
    double artificial_steps = (double) this->stepsPerEcho;
    for(int idx = 0; idx < this->histogram.size; idx++)
    {   
        artificial_xirate = this->histogram.bins[idx];
        toy.setXIrate(artificial_xirate);
        toy.setSurfaceRelaxivity(_sigmoid);
        toy.computeDecreaseFactor(this->imageVoxelResolution, this->diffusionCoefficient);
        this->penalties[idx] = pow(toy.getDecreaseFactor(), (artificial_xirate * artificial_steps));

        // debug
        // cout << "delta[" << artificial_xirate << "] = " << penalties[idx] << endl;
    }
}

void NMR_Simulation::createPenaltiesVector(double rho)
{
    // initialize penalties array
    penalties = new double[this->histogram.size];
    Walker toy;
    double artificial_xirate;
    double artificial_steps = (double) this->stepsPerEcho;
    for(int idx = 0; idx < this->histogram.size; idx++)
    {   
        artificial_xirate = this->histogram.bins[idx];
        toy.setXIrate(artificial_xirate);
        toy.setSurfaceRelaxivity(rho);
        toy.computeDecreaseFactor(this->imageVoxelResolution, this->diffusionCoefficient);
        penalties[idx] = pow(toy.getDecreaseFactor(), (artificial_xirate * artificial_steps));

        // debug
        // cout << "delta[" << artificial_xirate << "] = " << penalties[idx] << endl;
    }
}

// mapping simulation using bitblock data structure
void NMR_Simulation::mapSimulation_OMP()
{
    double begin_time = omp_get_wtime();
    cout << "initializing mapping simulation... ";
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        this->walkers[id].resetPosition();
        this->walkers[id].resetSeed();
        this->walkers[id].resetCollisions();
        this->walkers[id].resetTCollisions();
    }

    // initialize list of collision histograms
    (*this).initHistogramList(); 

    // loop throughout list
    for(int hst_ID = 0; hst_ID < this->histogramList.size(); hst_ID++)
    {
        int eBegin = this->histogramList[hst_ID].firstEcho;
        int eEnd = this->histogramList[hst_ID].lastEcho;
        for (uint id = 0; id < this->numberOfWalkers; id++)
        {
            for(uint echo = eBegin; echo < eEnd; echo++)
            {
                for (uint step = 0; step < this->stepsPerEcho; step++)
                {
                    this->walkers[id].map(bitBlock);
                }
            }
        }

        int steps = this->stepsPerEcho * (eEnd - eBegin);
        (*this).createHistogram(hst_ID, steps);

        for (uint id = 0; id < this->numberOfWalkers; id++)
        {
            this->walkers[id].tCollisions += this->walkers[id].collisions;
            this->walkers[id].resetCollisions();
        }
    }

    // recover walkers collisions from total sum and create a global histogram
    for (uint id = 0; id < this->numberOfWalkers; id++)
    {
        this->walkers[id].collisions = this->walkers[id].tCollisions;   
    }
    (*this).createHistogram();

    cout << "Completed.";
    double finish_time = omp_get_wtime();
    printElapsedTime(begin_time, finish_time);
}