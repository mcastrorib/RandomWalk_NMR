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

// include OpenMP for multicore implementation
#include <omp.h>

//include
#include "../Walker/walker.h"
#include "../BitBlock/bitBlock.h"
#include "../RNG/xorshift.h"
#include "../FileHandler/fileHandler.h"
#include "../ConsoleInput/consoleInput.h"
#include "NMR_Simulation.h"

using namespace std;
using namespace cv;

string NMR_Simulation::createDirectoryForResults()
{
    string path = this->rwNMR_config.getDBPath();
    createDirectory(path, this->simulationName);
    return (path + this->simulationName);
}

void NMR_Simulation::saveImageInfo(string filedir)
{
    string filename = filedir + "/NMR_imageInfo.txt";

    fileHandler external_file(filename);
    external_file.writeImageInfo(this->imagePath.completePath, 
                                 this->bitBlock.imageColumns, 
                                 this->bitBlock.imageRows, 
                                 this->bitBlock.imageDepth, 
                                 this->imageVoxelResolution);
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

void NMR_Simulation::saveNMRT2(string filePath)
{
    string filename = filePath + "/NMR_T2.txt";
    fileHandler external_file(filename);
    external_file.writeNMRT2(this->T2_bins, this->T2_simulated);
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

void NMR_Simulation::readT2fromFile(string filePath)
{

    string fileName = filePath + "/NMR_T2.txt";

    fileHandler external_file(fileName);
    external_file.readT2Distribution(fileName, this->T2_simulated);
}

void NMR_Simulation::readInputT2()
{
    cout << "reading input T2 distribution from file...";

    fileHandler external_file(this->inputT2File);
    external_file.readT2Distribution(this->inputT2File, this->T2_input);
    cout << "Ok." << endl;
}

void NMR_Simulation::info()
{
    (*this).printInputDetails();
}

void NMR_Simulation::printInputDetails()
{ 
    // print input details
    cout << "------------------------------------------------------" << endl;
    cout << ">>> NMR SIMULATION 3D PARAMETERS: " << this->simulationName << endl;
    cout << "------------------------------------------------------" << endl;
    cout << "Data path: " << this->rwNMR_config.getDBPath() + this->simulationName << endl;
    cout << "Image path: " << this->imagePath.completePath << endl;
    cout << "Image resolution (um/voxel): " << this->imageVoxelResolution << endl;
    cout << "Diffusion coefficient (um^2/ms): " << this->diffusionCoefficient << endl;
    cout << "Number of images: " << this->numberOfImages << endl;
    cout << "Number of steps in simulation: " << this->simulationSteps << endl;
    cout << "Decay time (ms): " << this->simulationSteps * this->timeInterval << endl;
    cout << "Walkers pore occupancy in simulation: " << this->walkerOccupancy * 100.0 << "%" << endl;
    cout << "Input T2 path: " << this->inputT2File << endl;

    cout << "Initial seed: ";
    if (this->seedFlag)
    {
        cout << this->initialSeed << endl;
    }
    else
    {
        cout << "not defined by user" << endl;
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

    cout << "------------------------------------------------------" << endl;
    cout << endl;
}