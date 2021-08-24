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
										 scale("linear"),
										 gap(0.0),
										 firstEcho(0),
										 lastEcho(0)
{
	vector<double> amps();
	vector<double> bins();
}

CollisionHistogram::CollisionHistogram(int _size, string _scale):size(_size),
														 		 scale(_scale),
																 gap(0.0),
																 firstEcho(0),
																 lastEcho(0)
{	
	// Initialize stl vectors
	vector<double> amps();
	vector<double> bins();

	if(this->size != 0)
		(*this).createBlankHistogram(this->size, this->scale);

}

CollisionHistogram::CollisionHistogram(const CollisionHistogram &_otherHistogram)
{
	this->size = _otherHistogram.size;
	this->scale = _otherHistogram.scale;
	this->gap = _otherHistogram.gap;
	this->amps = _otherHistogram.amps;
	this->bins = _otherHistogram.bins;
	this->firstEcho = _otherHistogram.firstEcho;
	this->lastEcho = _otherHistogram.lastEcho;
}

void CollisionHistogram::createBlankHistogram(int _size, string _scale)
{
	(*this).setScale(_scale);
	double gap = 1.0 / ((double) _size);
	(*this).setSize(_size);
	(*this).setGap(gap);
	this->amps.reserve(this->size);
	this->bins.reserve(this->size);
}

void CollisionHistogram::fillHistogram(vector<Walker> &_walkers, uint _numberOfSteps)
{
	if(this->scale == "log")
	{
		cout << "Histogram with log-scale was chosen :)" << endl;
		(*this).createBinsLogVector(_walkers);
		(*this).createAmpsLogVector(_walkers, _numberOfSteps);
	} else
	{
		(*this).createBinsLinearVector(_walkers);
		(*this).createAmpsLinearVector(_walkers, _numberOfSteps);
	}
}

void CollisionHistogram::createBinsLinearVector(vector<Walker> &_walkers)
{	
	double offset;
	double meanBin = (0.5) * this->gap;
	for(int idx = 0; idx < this->size; idx++)
	{	
		offset = idx * this->gap;
		this->bins.push_back(offset + meanBin);
	}
}

void CollisionHistogram::createAmpsLinearVector(vector<Walker> &_walkers, uint _numberOfSteps)
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

void CollisionHistogram::createBinsLogVector(vector<Walker> &_walkers)
{	
}

void CollisionHistogram::createAmpsLogVector(vector<Walker> &_walkers, uint _numberOfSteps)
{	
}

