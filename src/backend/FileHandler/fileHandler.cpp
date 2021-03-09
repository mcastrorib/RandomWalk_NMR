// include OpenCV core functions
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

// include C++ standard libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <math.h>
#include <stdint.h>

#include "../Walker/walker.h"
#include "../NMR_Simulation/NMR_Simulation.h"

#include "fileHandler.h"

using namespace std;
using namespace cv;

void fileHandler::writeInFile(NMR_Simulation _inputData)
{
    cout << "writing data in file...";

    ofstream fileObject;
    fileObject.open(this->filename, ios::app);

    fileObject.write((char *)&_inputData, sizeof(_inputData));

    cout << "Ok" << endl;
    fileObject.close();
}

void fileHandler::readFromFile(NMR_Simulation _outputData)
{
    cout << "reading data from file...";

    ifstream fileObject;
    fileObject.open(this->filename, ios::in);

    fileObject.read((char *)&_outputData, sizeof(_outputData));

    cout << "Ok" << endl;
    fileObject.close();
}

void fileHandler::writeImageInfo(string _imagePath, 
                                 int _imageColumns, 
                                 int _imageRows, 
                                 int _imageDepth, 
                                 double _imageResolution, 
                                 double _imagePorosity,
                                 double _SVp)
{
    // file object init
    ofstream fileObject;

    // open file
    fileObject.open(this->filename, ios::out);
    if (fileObject.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    // write info 
    fileObject << "path: " << _imagePath << endl;
    fileObject << "width(x): " << _imageColumns << endl;
    fileObject << "height(y): " << _imageRows << endl;
    fileObject << "depth(z): " << _imageDepth << endl;
    fileObject << "resolution: " << _imageResolution << endl;
    fileObject << "porosity: " << _imagePorosity << endl;
    fileObject << "SVp: " << _SVp << endl;

    // close file
    fileObject.close();
}

void fileHandler::writeIndividualCollisions(vector<Walker> &_walkers, uint _rwsteps)
{
    ofstream fileObject;
    fileObject.open(this->filename, ios::out);
    if (fileObject.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    fileObject << "WalkerID" << ", ";
    fileObject << "PositionXi, ";
    fileObject << "PositionYi, ";
    fileObject << "PositionZi, ";
    fileObject << "PositionXf, ";
    fileObject << "PositionYf, ";
    fileObject << "PositionZf, ";
    fileObject << "Collisions, ";
    fileObject << "XIRate, ";
    fileObject << "RNGSeed" << endl;

    // double xirate_factor = 1.0 / (double) _rwsteps;
    double xirate;

    for (uint index = 0; index < _walkers.size(); index++)
    {
        // xirate = walkers[index].getCollisions() * xirate_factor;
        // xirate = _walkers[index].getXIrate();
        // cout << xirate << endl;

        fileObject << index << ", ";
        fileObject << _walkers[index].getInitialPositionX() << ", ";
        fileObject << _walkers[index].getInitialPositionY() << ", ";
        fileObject << _walkers[index].getInitialPositionZ() << ", ";
        fileObject << _walkers[index].getPositionX() << ", ";
        fileObject << _walkers[index].getPositionY() << ", ";
        fileObject << _walkers[index].getPositionZ() << ", ";        
        fileObject << _walkers[index].getCollisions() << ", ";
        fileObject << _walkers[index].getXIrate() << ", ";
        fileObject << _walkers[index].getInitialSeed() << endl;
    }

    fileObject.close();
}

void fileHandler::writeEnergyDecay(vector<double> &_globalEnergy, vector<double> &_timeDecay)
{
    ofstream fileObject;
    fileObject.open(this->filename, ios::out);
    if (fileObject.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    // write initial state, sample 100% magnetized v
    //fileObject << time << " " << initialEnergy << endl;

    uint size = _timeDecay.size();
    int precision = std::numeric_limits<double>::max_digits10;
    for (uint index = 0; index < size; index++)
    {
        fileObject << setprecision(precision) << _timeDecay[index] << " " << _globalEnergy[index] << endl;
    }

    fileObject.close();
}

void fileHandler::writeBitBlockObject2D(int _numberOfBlocks,
                                        int _blockRows,
                                        int _blockColumns,
                                        int _imageRows,
                                        int _imageColumns,
                                        uint64_t *_blocks)
{
    ofstream fileObject;
    fileObject.open(this->filename, ios::out);
    if (fileObject.fail())
    {
        cout << "Could not open file from disc" << endl;
    }

    // write properties
    fileObject << "blocks, ";
    fileObject << "bRows, ";
    fileObject << "bColumns, ";
    fileObject << "imgRows, ";
    fileObject << "imgColumns, " << endl;
    fileObject << _numberOfBlocks << ", ";
    fileObject << _blockRows << ", ";
    fileObject << _blockColumns << ", ";
    fileObject << _imageRows << ", ";
    fileObject << _imageColumns << endl;

    fileObject << endl;
    fileObject << "blockID, ";
    fileObject << "blockData" << endl;

    for (int index = 0; index < _numberOfBlocks; index++)
    {
        fileObject << index << ", ";
        fileObject << _blocks[index] << endl;
    }

    fileObject.close();
}

void fileHandler::writeBitBlockObject3D(int _numberOfBlocks,
                                        int _blockRows,
                                        int _blockColumns,
                                        int _blockDepth,
                                        int _imageRows,
                                        int _imageColumns,
                                        int _imageDepth,
                                        uint64_t *_blocks)
{
    ofstream fileObject;
    fileObject.open(this->filename, ios::out);
    if (fileObject.fail())
    {
        cout << "Could not open file from disc" << endl;
    }

    // write properties
    fileObject << "blocks, ";
    fileObject << "bRows, ";
    fileObject << "bColumns, ";
    fileObject << "bDepth, ";
    fileObject << "imgRows, ";
    fileObject << "imgColumns, ";
    fileObject << "imgDepth, " << endl;
    fileObject << _numberOfBlocks << ", ";
    fileObject << _blockRows << ", ";
    fileObject << _blockColumns << ", ";
    fileObject << _blockDepth << ", ";
    fileObject << _imageRows << ", ";
    fileObject << _imageColumns << ", ";
    fileObject << _imageDepth << endl;

    fileObject << endl;
    fileObject << "blockID, ";
    fileObject << "blockData" << endl;

    for (int index = 0; index < _numberOfBlocks; index++)
    {
        fileObject << index << ", ";
        fileObject << _blocks[index] << endl;
    }

    fileObject.close();
}

void fileHandler::writeNMRT2(vector<double> &_bins, vector<double> &_amps)
{
    const size_t num_points = _bins.size();
    ofstream in(this->filename, std::ios::out);
    if (in)
    {
        // in << x_label << "\t" << y_label << endl;
        for (int i = 0; i < num_points; i++)
        {
            in << _bins[i] << "\t" << _amps[i] << endl;
        }
        return;
    }
    throw(errno);
}

void fileHandler::writeHistogram(vector<double> &_bins, vector<double> &_amps)
{
    ofstream in(this->filename, std::ios::out);
 
    const size_t num_points = _bins.size();
    if (in)
    {
        // in << x_label << "\t" << y_label << endl;
        for (int i = 0; i < num_points; i++)
        {
            in << _bins[i] << "\t" << _amps[i] << endl;
        }
        return;
    }
    throw(errno);
}

void fileHandler::writeHistogramFromList(vector<double> &_bins, vector<double> &_amps, int _id)
{
    
    ofstream in;
    if(_id == 0) in.open(this->filename, std::ios::out);
    else in.open(this->filename, std::ios::app);

    in << "histogram [" << _id << "]" << endl;
    const size_t num_points = _bins.size();
    if (in)
    {
        // in << x_label << "\t" << y_label << endl;
        for (int i = 0; i < num_points; i++)
        {
            in << _bins[i] << "\t" << _amps[i] << endl;
        }
        in << endl;
        return;
    }
    throw(errno);
}

void fileHandler::readT2Distribution(string fileName, vector<double> &T2_distribution)
{
    ifstream fileObject;
    fileObject.open(this->filename, ios::in);
    if (fileObject.fail())
    {
        cout << "Could not open file from disc" << endl;
    }

    double valueT2;
    string foo0, foo1;
    if (fileObject)
    {

        while (fileObject >> foo0 >> foo1)
        {

            stringstream value(foo1);
            value >> valueT2;
            T2_distribution.insert(T2_distribution.end(), valueT2);
        }
        return;
    }
    throw(errno);
}