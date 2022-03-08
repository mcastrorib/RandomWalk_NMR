#ifndef INTERNAL_FIELD_H_
#define INTERNAL_FIELD_H_

#include <string>
#include <vector>
#include <omp.h>
#include "NMR_defs.h"
#include "../BitBlock/bitBlock.h"

using namespace std;

class InternalField
{
public:	
	double *data;
	
	InternalField(string _file);
	InternalField(BitBlock &_bitblock, double _poreVal, double _matVal);
	InternalField(const InternalField &_other);
	virtual ~InternalField()
	{
		if(this->data != NULL)
        {
            delete[] this->data;
            this->data = NULL;
        }
	}

	int getDimX() { return this->dimX; }
	int getDimY() { return this->dimY; }
	int getDimZ() { return this->dimZ; }
	long getSize() { return (*this).getDimX() * (*this).getDimY() * (*this).getDimZ(); }
	double *getData() { return this->data; }
	double getData(int x, int y, int z);

private:
	int dimX;
    int dimY;
    int dimZ;
    uint rowScale;
	uint depthScale;
	
	uint getRowScale() { return this->rowScale; }
	uint getDepthScale() { return this->depthScale; }
	long getIndex(int x, int y, int z) { return ( x + (y * (*this).getRowScale()) + (z * (*this).getDepthScale()) ); }
	
	void setDimX(int _x);
	void setDimY(int _y);
	void setDimZ(int _z);
	void setLinearRowScale();
	void setLinearDepthScale();
	void setDims(int _x, int _y, int _z);
	void allocDataArray();
	void fillDataArray(BitBlock &_bitblock, double _poreVal, double _matVal);
	void fillData(long _index, double _data);
	void readDataFromFile(string _file);
	void show();	
};

#endif