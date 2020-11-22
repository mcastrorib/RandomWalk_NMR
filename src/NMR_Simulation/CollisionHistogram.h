#ifndef NMR_HISTOGRAM_H_
#define NMR_HISTOGRAM_H_

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
	double gap;
	vector<double> amps;
	vector<double> bins;
	int firstEcho;
	int lastEcho;
	

	CollisionHistogram();
	CollisionHistogram(int _size);
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

	void createBlankHistogram(int _size);
	void fillHistogram(vector<Walker> &_walkers, uint _numberOfSteps);
	void createBinsVector(vector<Walker> &_walkers);
	void createAmpsVector(vector<Walker> &_walkers, uint _numberOfSteps);
	void setSize(int _size) { this->size = _size; }
	void setGap(double _gap) { this->gap = _gap; }
	void setFirstEcho(int _firstEcho) { this->firstEcho = _firstEcho;}
	void setLastEcho(int _lastEcho) { this->lastEcho = _lastEcho;}
};

#endif