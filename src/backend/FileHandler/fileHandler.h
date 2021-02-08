#ifndef FILE_HANDLER_H
#define FILE_HANDLER_H

#include "../NMR_Simulation/NMR_Simulation.h"
#include "../Walker/walker.h"

using namespace std;

class fileHandler
{
public:
    string filename;

    fileHandler(string _filename) : filename(_filename){};

    void writeInFile(NMR_Simulation _inputData);
    void readFromFile(NMR_Simulation _outputData);

    void writeImageInfo(string _imagePath, int _imageRows, int _imageColumns, int _imageDepth, double _imageResolution, double _imagePorosity);

    void writeIndividualCollisions(vector<Walker> _walkers, uint _iterations);

    void writeEnergyDecay(vector<double> _globalEnergy, vector<double> _timeDecay);

    void writeBitBlockObject2D(int _numberOfBlocks,
                               int _blockRows,
                               int _blockColumns,
                               int _imageRows,
                               int _imageColumns,
                               uint64_t *_blocks);

    void writeBitBlockObject3D(int _numberOfBlocks,
                               int _blockRows,
                               int _blockColumns,
                               int _blockDepth,
                               int _imageRows,
                               int _imageColumns,
                               int _imageDepth,
                               uint64_t *_blocks);

    void writeNMRT2(vector<double> &_bins, vector<double> &_amps);

    void writeHistogram(vector<double> &_bins, vector<double> &_amps);

    void writeHistogramFromList(vector<double> &_bins, vector<double> &_amps, int _id);

    void readT2Distribution(string fileName, vector<double> &T2_distribution);
};

// some small functions to handle bit data

// small function to print int64_t in bits
void showbits(uint64_t x);

// small function to print int64_t in an 8 by 8 bitblock
void showblock8x8(uint64_t x);

// small function to print int64_t in an 4 by 4 by 4 cubic bitblock
void showblock4x4x4(uint64_t x);

// small function to print elapsed time
void printElapsedTime(double start, double finish);

// small function to create a directory in filesystem
void createDirectory(string path, string name);

// some basic linear functions
double euclideanDistance(vector<double> &vec1, vector<double> &vec2);
double norm(vector<double> &vec);
double dotProduct(vector<double> &vec1, vector<double> &vec2);
vector<double> linspace(double start, double end, uint points = 10);
vector<double> logspace(double exp_start, double exp_end, uint points = 10, double base = 10.0);


#endif
