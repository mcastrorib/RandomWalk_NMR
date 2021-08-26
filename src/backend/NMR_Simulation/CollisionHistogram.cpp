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
	(*this).setGap(gap);

	if(this->scale == "log")
	{
		// reserve additional spot for 0 collisions in log scale
		(*this).setSize(_size + 1); 
	} else
	{
		(*this).setSize(_size);
	}

	this->amps.reserve(this->size);
	this->bins.reserve(this->size);
}

void CollisionHistogram::fillHistogram(vector<Walker> &_walkers, uint _numberOfSteps)
{
	if(this->scale == "log")
	{
		(*this).createBinsLogVector(_walkers, _numberOfSteps);
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

void CollisionHistogram::createBinsLogVector(vector<Walker> &_walkers, uint _numberOfSteps)
{	
	// find min rate to generate bins accordingly
	double steps = (double) _numberOfSteps;
	uint first_idx = 0;
	while(first_idx < _walkers.size() and _walkers[first_idx].collisions == 0)
	{
		first_idx++;	
	}

	if(first_idx < _walkers.size())
	{
		double min_rate = _walkers[first_idx].collisions / steps;
		double xi_rate;
		for(uint id = first_idx; id < _walkers.size(); id++)
		{
			if(_walkers[id].collisions != 0)
			{
				xi_rate = _walkers[id].collisions / steps;
				if(xi_rate < min_rate)
				{
					min_rate = xi_rate;
				}
			}
		}


		// create vector of logspaced values 
		double logmin_rate = log10(min_rate);
		vector<double> logbins = (*this).logspace(round(logmin_rate), 0.0, this->size - 1);

		// first entry used for control no-collision cases
		this->bins.push_back(round(logmin_rate) - 1.0);

		// other entries based on logspaced vector
		for(uint idx = 1; idx < this->size; idx++)
		{
			this->bins.push_back(logbins[idx - 1]);
		}

	} else
	{
		// dealing with free diffusion (no collision at all)
		for(uint idx = 0; idx < this->size; idx++)
		{
			this->bins.push_back(0.0);
		}
	}
}

void CollisionHistogram::createAmpsLogVector(vector<Walker> &_walkers, uint _numberOfSteps)
{	
	// 1st: initialize entries
	for(uint idx = 0; idx < this->size; idx++)
	{
		this->amps.push_back(0.0);
	}

	
	// compute histogram
	int histogramIndex;
	double xi_rate;
	double logGap = log10(this->bins[2]) - log10(this->bins[1]);
	double min_val = log10(this->bins[1]);
	double steps = (double) _numberOfSteps;
	uint leaks = 0;

	for(uint id = 0; id < _walkers.size(); id++)
	{
		if(_walkers[id].collisions == 0)
		{
			this->amps[0] += 1.0;
		} else
		{
			xi_rate = (_walkers[id].collisions / steps);
			histogramIndex = floor( round( (log10(xi_rate) - min_val) / logGap ) );
			histogramIndex += 1;
			if(histogramIndex > 0 /* and histogramIndex < this->size */)
			{
				this->amps[histogramIndex] += 1.0;				
			} else
			{
				this->amps[1] += 1.0;		
			}
		}
	}

	// normalize histogram, i.e, histogram values sum 1.0
	double numberOfWalkers = (double) _walkers.size();
	for(int id = 0; id < this->size; id++)
	{
		this->amps[id] = this->amps[id] / numberOfWalkers;
	}
}

// Returns a vector<double> linearly space from @start to @end with @points
vector<double> CollisionHistogram::linspace(double start, double end, uint points)
{
    vector<double> vec(points);
    double step = (end - start) / ((double) points - 1.0);
    
    for(int idx = 0; idx < points; idx++)
    {
        double x_i = start + step * idx;
        vec[idx] = x_i;
    }

    return vec;
}

// Returns a vector<double> logarithmly space from 10^@exp_start to 10^@end with @points
vector<double> CollisionHistogram::logspace(double exp_start, double exp_end, uint points, double base)
{
    vector<double> vec(points);
    double step = (exp_end - exp_start) / ((double) points - 1.0);
    
    for(int idx = 0; idx < points; idx++)
    {
        double x_i = exp_start + step * idx;
        vec[idx] = pow(base, x_i);
    }

    return vec;
}