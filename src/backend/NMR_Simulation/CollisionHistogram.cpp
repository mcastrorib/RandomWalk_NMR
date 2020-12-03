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
#include "CollisionHistogram.h"
#include "NMR_Simulation.h"
#include "../Walker/walker.h"


using namespace std;

CollisionHistogram::CollisionHistogram():size(0),
										 gap(0.0),
										 firstEcho(0),
										 lastEcho(0)
{
	vector<double> amps();
	vector<double> bins();
}

CollisionHistogram::CollisionHistogram(int _size):size(_size),
												  gap(0.0),
												  firstEcho(0),
												  lastEcho(0)
{	
	// Initialize stl vectors
	vector<double> amps();
	vector<double> bins();

	if(this->size != 0)
		(*this).createBlankHistogram(this->size);

}

CollisionHistogram::CollisionHistogram(const CollisionHistogram &_otherHistogram)
{
	this->size = _otherHistogram.size;
	this->gap = _otherHistogram.gap;
	this->amps = _otherHistogram.amps;
	this->bins = _otherHistogram.bins;
	this->firstEcho = _otherHistogram.firstEcho;
	this->lastEcho = _otherHistogram.lastEcho;
}

void CollisionHistogram::createBlankHistogram(int _size)
{
	double gap = 1.0 / ((double) _size);
	(*this).setSize(_size);
	(*this).setGap(gap);
	this->amps.reserve(this->size);
	this->bins.reserve(this->size);
}

void CollisionHistogram::fillHistogram(vector<Walker> &_walkers, uint _numberOfSteps)
{
	(*this).createBinsVector(_walkers);
	(*this).createAmpsVector(_walkers, _numberOfSteps);
}

void CollisionHistogram::createBinsVector(vector<Walker> &_walkers)
{	
	double offset;
	double meanBin = (0.5) * this->gap;
	for(int idx = 0; idx < this->size; idx++)
	{	
		offset = idx * this->gap;
		this->bins.push_back(offset + meanBin);
	}
}

void CollisionHistogram::createAmpsVector(vector<Walker> &_walkers, uint _numberOfSteps)
{	
	// initialize amps vector entries	
	for(int id = 0; id < this->size; id++)
	{
		this->amps.push_back(0.0);
	}


	// compute histogram
	int histogramIndex;
	double xi_rate;
	double steps = (double) _numberOfSteps;
	for(uint id = 0; id < _walkers.size(); id++)
	{
		xi_rate = _walkers[id].collisions / steps;
		histogramIndex = floor(xi_rate / this->gap);
		if(histogramIndex >= this->size) histogramIndex--;
		this->amps[histogramIndex] += 1.0;
	}

	// normalize histogram, i.e, histogram values sum 1.0
	double numberOfWalkers = (double) _walkers.size();
	for(int id = 0; id < this->size; id++)
	{
		this->amps[id] = this->amps[id] / numberOfWalkers;
	}	
}

