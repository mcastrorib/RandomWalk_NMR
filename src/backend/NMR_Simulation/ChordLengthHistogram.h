#ifndef CHORD_LENGTH_HISTOGRAM_H_
#define CHORD_LENGTH_HISTOGRAM_H_

// include string stream manipulation functions
#include <sstream>
#include <iomanip>
#include <vector>
#include "NMR_defs.h"
#include "../BitBlock/bitBlock.h"

using namespace std;

class ChordLengthHistogram
{
public:	
	int size;
	int gap;
	BitBlock *bitBlock;
	vector<uint> chordsX;
	vector<uint> chordsZ;
	vector<uint> chordsY;
	vector<double> ampsX;
	vector<double> ampsY;
	vector<double> ampsZ;	
	vector<double> bins;
	

	ChordLengthHistogram();	
	ChordLengthHistogram(BitBlock &_bitBlock);
	ChordLengthHistogram(const ChordLengthHistogram &_otherHistogram);
	virtual ~ChordLengthHistogram()
	{
		if(this->bitBlock != NULL) 
		{
			// delete[] this->bitBlock;
        	this->bitBlock = NULL;
		}
		// cout << "erasing histogram" << endl;
	}

	void clear()
	{
		this->size = 0;
		this->gap = 1;
		this->ampsX.clear();
		this->ampsY.clear();
		this->ampsZ.clear();
		this->bins.clear();
	}

	void run();
	void readImageInfo();
	void applyChords();
	void applyChords2D();
	void applyChords3D();
	void createBlankHistogram();
	void createAmpsVector();
	void createBinsVector();
	void fillHistogram();
	
	void setSize(int _size) { this->size = _size; }
	void setGap(double _gap) { this->gap = _gap; }
	void setBitBlock(BitBlock &_bitBlock) { this->bitBlock = &_bitBlock; }
	
};

#endif