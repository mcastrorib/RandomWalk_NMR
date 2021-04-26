// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

// include OpenMP for multicore implementation
#include <omp.h>

// Class header
#include "LeastSquareAdjust.h"

LeastSquareAdjust::LeastSquareAdjust(vector<double> &_x, vector<double> &_y): X(_x), 
															Y(_y),
															meanX(0.0),
															meanY(0.0),
															A(0.0),
															B(0.0),
															solved(false)
{
	this->begin = 0;
	this->end = this->X.size();
	this->points = this->end + 1;
	this->threshold = numeric_limits<double>::max();
}

void LeastSquareAdjust::setX(vector<double> &_x)
{
	this->X = _x;
	(*this).setAsUnsolved();
}

void LeastSquareAdjust::setY(vector<double> &_y)
{
	this->Y = _y;
	(*this).setAsUnsolved();
}

void LeastSquareAdjust::setThreshold(double _threshold)
{
	this->threshold = _threshold;
	(*this).setLimits();
}

void LeastSquareAdjust::setPoints(int _points)
{
	this->points = _points;
	if(_points > 0 and _points < this->end)
	{
		(*this).setThreshold(this->X[_points - 1]);
	}
}

void LeastSquareAdjust::setLimits()
{
	int idx = this->begin;
	bool limitExceeded = false;

	while(idx < this->end && limitExceeded == false)
	{
		if(this->threshold <= fabs(this->X[idx])) 
			limitExceeded = true;
		
		idx++;
	}

	if(limitExceeded) this->end = idx;
}

void LeastSquareAdjust::solve()
{
	this->meanX = computeMean(this->X);
	this->meanY = computeMean(this->Y);
	(*this).computeB();
	(*this).computeA();
	(*this).setAsSolved();
}    

double LeastSquareAdjust::computeMean(vector<double> &_vector)
{
	double sum = 0.0;
	double size = (double) _vector.size();
	// double size = (double) this->points;

	for(uint idx = this->begin; idx < this->end; idx++)
	{
		sum += _vector[idx];
	}

	return (sum/size);
}

void LeastSquareAdjust::computeB()
{
	// get B dividend
	double dividend = 0.0;
	for(uint idx = this->begin; idx < this->end; idx++)
	{
		dividend += this->X[idx] * (this->Y[idx] - this->meanY);
	}

	// get B divisor
	double divisor = 0.0;
	for(uint idx = this->begin; idx < this->end; idx++)
	{
		divisor += this->X[idx] * (this->X[idx] - this->meanX);
	}

	this->B = (dividend/divisor);
}

void LeastSquareAdjust::computeA()
{
	this->A = this->meanY - (this->B * this->meanX);
}

void LeastSquareAdjust::setAsSolved()
{
	this->solved = true;
}

void LeastSquareAdjust::setAsUnsolved()
{
	this->solved = false;
}
