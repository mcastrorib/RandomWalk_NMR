// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>

// include OpenMP for multicore implementation
#include <omp.h>

//include
#include "ChordLengthHistogram.h"
#include "NMR_Simulation.h"
#include "../BitBlock/bitBlock.h"

using namespace std;

ChordLengthHistogram::ChordLengthHistogram():size(0),
										 	 gap(1),
										 	 bitBlock(NULL)
{
	vector<double> ampsX();
	vector<double> ampsY();
	vector<double> ampsZ();
	vector<double> bins();
	vector<uint> chordsX();
	vector<uint> chordsY();
	vector<uint> chordsZ();

}

ChordLengthHistogram::ChordLengthHistogram(BitBlock &_bitBlock):size(0),
														 	    gap(1),
														 	    bitBlock(&_bitBlock)
{
	vector<double> ampsX();
	vector<double> ampsY();
	vector<double> ampsZ();
	vector<double> bins();
	vector<uint> chordsX();
	vector<uint> chordsY();
	vector<uint> chordsZ();

	(*this).readImageInfo();
	(*this).applyChords();
	(*this).createBlankHistogram();
	(*this).fillHistogram();
}

ChordLengthHistogram::ChordLengthHistogram(const ChordLengthHistogram &_otherHistogram)
{
	this->size = _otherHistogram.size;
	this->gap = _otherHistogram.gap;
	this->bitBlock = _otherHistogram.bitBlock;
	this->chordsX = _otherHistogram.chordsX;
	this->chordsY = _otherHistogram.chordsY;
	this->chordsZ = _otherHistogram.chordsZ;
	this->ampsX = _otherHistogram.ampsX;
	this->ampsY = _otherHistogram.ampsY;
	this->ampsZ = _otherHistogram.ampsZ;
	this->bins = _otherHistogram.bins;
}

void ChordLengthHistogram::run()
{
	(*this).readImageInfo();
	(*this).applyChords();
	(*this).createBlankHistogram();
	(*this).fillHistogram();
}

void ChordLengthHistogram::readImageInfo()
{
	int maxDim = this->bitBlock->imageRows;
	if(maxDim < this->bitBlock->imageColumns) maxDim = this->bitBlock->imageColumns;
	if(maxDim < this->bitBlock->imageDepth) maxDim = this->bitBlock->imageDepth;
	(*this).setSize(maxDim);
}

void ChordLengthHistogram::applyChords()
{
	cout << endl << "*** applying chords length estimation ***" << endl;
	double time = omp_get_wtime();
	
    if(this->bitBlock->imageDepth == 1) (*this).applyChords2D();
    else (*this).applyChords3D();

    time = omp_get_wtime() - time;
	cout << "in " << time << " seconds." << endl;
}

void ChordLengthHistogram::applyChords2D()
{
	int dimX = this->bitBlock->imageColumns;
	int dimY = this->bitBlock->imageRows;

}

void ChordLengthHistogram::applyChords3D()
{
	int dimX = this->bitBlock->imageColumns;
	int dimY = this->bitBlock->imageRows;
	int dimZ = this->bitBlock->imageDepth;

	int currentBlock, currentBit, currentPhase, currentLength;
	int firstBlock, firstBit;

	for(int z = 0; z < dimZ; z++)
    {
        for(int y = 0; y < dimY; y++)
        {
        	currentLength = 0;
        	vector<int> newChords;
            for(int x = 0; x < dimX; x++)
            {                                            
                currentBlock = this->bitBlock->findBlock(x, y, z);
                currentBit = this->bitBlock->findBitInBlock(x, y, z);           

                if (this->bitBlock->checkIfBitIsWall(currentBlock, currentBit))
                {
                    currentPhase = 0;
                } else
                {
                	currentPhase = 1;
                }                

                if(currentPhase == 1) currentLength++;
                else if(currentLength > 0) 
                {
                	newChords.push_back(currentLength);               
                	currentLength = 0;
                } 
            }

            if(currentLength > 0) newChords.push_back(currentLength);

            // check first and last chords 
            firstBlock = this->bitBlock->findBlock(0, y, z);
            firstBit = this->bitBlock->findBitInBlock(0, y, z);
            if(!this->bitBlock->checkIfBitIsWall(firstBlock, firstBit) and currentPhase == 1 and newChords.size() > 1)
            {
            	int lastChordLength = newChords.back();
            	newChords.pop_back();
            	newChords[0] = newChords[0] + lastChordLength;
            }    

            // add chords to chordsX list
            this->chordsX.insert(this->chordsX.end(), newChords.begin(), newChords.end());
        }
    }

    for(int z = 0; z < dimZ; z++)
    {
        for(int x = 0; x < dimX; x++)
        {
        	currentLength = 0;
        	vector<int> newChords;
            for(int y = 0; y < dimY; y++)
            {                                            
                currentBlock = this->bitBlock->findBlock(x, y, z);
                currentBit = this->bitBlock->findBitInBlock(x, y, z);           

                if (this->bitBlock->checkIfBitIsWall(currentBlock, currentBit))
                {
                    currentPhase = 0;
                } else
                {
                	currentPhase = 1;
                }                

                if(currentPhase == 1) currentLength++;
                else if(currentLength > 0) 
                {
                	newChords.push_back(currentLength);               
                	currentLength = 0;
                } 
            }

            if(currentLength > 0) newChords.push_back(currentLength);

            // check first and last chords 
            firstBlock = this->bitBlock->findBlock(x, 0, z);
            firstBit = this->bitBlock->findBitInBlock(x, 0, z);
            if(!this->bitBlock->checkIfBitIsWall(firstBlock, firstBit) and currentPhase == 1 and newChords.size() > 1)
            {
            	int lastChordLength = newChords.back();
            	newChords.pop_back();
            	newChords[0] = newChords[0] + lastChordLength;
            }    

            // add chords to chordsX list
            this->chordsY.insert(this->chordsY.end(), newChords.begin(), newChords.end());
        }
    }

    for(int x = 0; x < dimX; x++)
    {
        for(int y = 0; y < dimY; y++)
        {
        	currentLength = 0;
        	vector<int> newChords;
            for(int z = 0; z < dimZ; z++)
            {                                            
                currentBlock = this->bitBlock->findBlock(x, y, z);
                currentBit = this->bitBlock->findBitInBlock(x, y, z);           

                if (this->bitBlock->checkIfBitIsWall(currentBlock, currentBit))
                {
                    currentPhase = 0;
                } else
                {
                	currentPhase = 1;
                }                

                if(currentPhase == 1) currentLength++;
                else if(currentLength > 0) 
                {
                	newChords.push_back(currentLength);               
                	currentLength = 0;
                } 
            }

            if(currentLength > 0) newChords.push_back(currentLength);

            // check first and last chords 
            firstBlock = this->bitBlock->findBlock(x, y, 0);
            firstBit = this->bitBlock->findBitInBlock(x, y, 0);
            if(!this->bitBlock->checkIfBitIsWall(firstBlock, firstBit) and currentPhase == 1 and newChords.size() > 1)
            {
            	int lastChordLength = newChords.back();
            	newChords.pop_back();
            	newChords[0] = newChords[0] + lastChordLength;
            }    

            // add chords to chordsX list
            this->chordsZ.insert(this->chordsZ.end(), newChords.begin(), newChords.end());
        }
    }
}

void ChordLengthHistogram::createBlankHistogram()
{	
	vector<double> newVector(this->size, 0.0);
	this->ampsX.assign(newVector.begin(), newVector.end());
	this->ampsY.assign(newVector.begin(), newVector.end());
	this->ampsZ.assign(newVector.begin(), newVector.end());

	this->bins.reserve(this->size);
}

void ChordLengthHistogram::fillHistogram()
{
	(*this).createBinsVector();
	(*this).createAmpsVector();
}

void ChordLengthHistogram::createBinsVector()
{
	for(uint i = 0; i < this->size; i++) 
		this->bins.push_back(i+1);
}

void ChordLengthHistogram::createAmpsVector()
{
	// fill chords X
	for(uint idx = 0; idx < this->chordsX.size(); idx++) 
		this->ampsX[chordsX[idx] - 1] += 1.0;
	
	// fill chords Y
	for(uint idx = 0; idx < this->chordsY.size(); idx++) 
		this->ampsY[chordsY[idx] - 1] += 1.0;
	
	// fill chords Z
	for(uint idx = 0; idx < this->chordsZ.size(); idx++) 
		this->ampsZ[chordsZ[idx] - 1] += 1.0;

	cout << "ampsX = {";
    for(int i = 0; i < this->ampsX.size(); i++)
    {
    	cout << ampsX[i] << " ";
    }
    cout << "}" << endl;

    cout << "ampsY = {";
    for(int i = 0; i < this->ampsX.size(); i++)
    {
    	cout << ampsY[i] << " ";
    }
    cout << "}" << endl;

    cout << "ampsZ = {";
    for(int i = 0; i < this->ampsX.size(); i++)
    {
    	cout << ampsZ[i] << " ";
    }
    cout << "}" << endl;

	
}