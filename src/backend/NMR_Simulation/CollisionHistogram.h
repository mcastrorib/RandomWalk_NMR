#ifndef COLLISION_HISTOGRAM_H_
#define COLLISION_HISTOGRAM_H_

// include string stream manipulation functions
#include <sstream>
#include <iomanip>

#include <vector>

// include OpenMP for multicore implementation
#include <omp.h>

#include "NMR_defs.h"
#include "../Walker/walker.h"

using namespace std;

class CollisionHistogram
{
public:	
	int size;
	string scale;
	double gap;
	vector<double> amps;
	vector<double> bins;
	int firstEcho;
	int lastEcho;
	

	CollisionHistogram();
	CollisionHistogram(int _size, string _scale);
	CollisionHistogram(const CollisionHistogram &_otherHistogram);
	virtual ~CollisionHistogram()
	{
		// cout << "erasing histogram" << endl;
	}

	void clear()
	{
		this->size = 0;
		this->gap = 0;
		this->amps.clear();
		this->bins.clear();
	}

	void createBlankHistogram(int _size, string scale);
	void fillHistogram(vector<Walker> &_walkers, uint _numberOfSteps);
	void createBinsLinearVector(vector<Walker> &_walkers);
	void createAmpsLinearVector(vector<Walker> &_walkers, uint _numberOfSteps);
	void createBinsLogVector(vector<Walker> &_walkers);
	void createAmpsLogVector(vector<Walker> &_walkers, uint _numberOfSteps);
	void setSize(int _size) { this->size = _size; }
	void setScale(string _scale) { this->scale = _scale; }
	void setGap(double _gap) { this->gap = _gap; }
	void setFirstEcho(int _firstEcho) { this->firstEcho = _firstEcho;}
	void setLastEcho(int _lastEcho) { this->lastEcho = _lastEcho;}
};

#endif