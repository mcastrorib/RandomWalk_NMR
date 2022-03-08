// include C++ standard libraries
#include <iostream>
#include <iomanip>      // std::setprecision
#include <sstream>
#include <fstream>
#include <string>

// include OpenMP for multicore implementation
#include <omp.h>

//include 
#include "NMR_defs.h"
#include "InternalField.h"

using namespace std;

InternalField::InternalField(BitBlock &_bitblock, double _poreValue, double _matValue) : dimX(0), 
																						 dimY(0), 
																						 dimZ(0),
																						 rowScale(0),
																						 depthScale(0), 
																						 data(NULL)
{
	(*this).setDims(_bitblock.imageColumns, _bitblock.imageRows, _bitblock.imageDepth);
	(*this).allocDataArray();
	(*this).fillDataArray(_bitblock, _poreValue, _matValue);
}

InternalField::InternalField(string _file) : dimX(0), 
											 dimY(0), 
											 dimZ(0), 
											 rowScale(0), 
											 depthScale(0), 
											 data(NULL)
{
	cout << "import internal magnetic field map to be implemented." << endl;
}

InternalField::InternalField(const InternalField &_other)
{
	this->dimX = _other.dimX;
	this->dimY = _other.dimY;
	this->dimZ = _other.dimZ;
	this->data = _other.data;
}

void InternalField::setDimX(int _x)
{
	if(_x > 0)	this->dimX = _x;
}

void InternalField::setDimY(int _y)
{
	if(_y > 0)	this->dimY = _y;
}

void InternalField::setDimZ(int _z)
{
	if(_z > 0)	this->dimZ = _z;
}

void InternalField::setLinearRowScale()
{
	this->rowScale = (*this).getDimX(); 	
}

void InternalField::setLinearDepthScale()
{
	this->depthScale = (*this).getDimX() * (*this).getDimY(); 
}

void InternalField::setDims(int _x, int _y, int _z)
{
	(*this).setDimX(_x);
	(*this).setDimY(_y);
	(*this).setDimZ(_z);
	(*this).setLinearRowScale();
	(*this).setLinearDepthScale();
}

void InternalField::allocDataArray()
{
	int size = (*this).getSize();
	this->data = new double[size];
}

void InternalField::fillDataArray(BitBlock &_bitblock, double _poreValue, double _matValue)
{
	double newValue;
	long currentIndex;
	
	for(int z = 0; z < (*this).getDimZ(); z++)
	{
		for(int y = 0; y < (*this).getDimY(); y++)
		{
			for(int x = 0; x < (*this).getDimX(); x++)
			{
				
				int block = _bitblock.findBlock(x, y, z);
				int bit = _bitblock.findBitInBlock(x, y, z);
				if(_bitblock.checkIfBitIsWall(block, bit))
				{
					newValue = _matValue;
				} else
				{
					newValue = _poreValue;
				}
				
				currentIndex = (*this).getIndex(x, y, z);
				(*this).fillData(currentIndex, newValue);
			}
		}
	}

	// (*this).show();
}


void InternalField::fillData(long _index, double _value)
{
	this->data[_index] = _value;
}

void InternalField::show()
{
	cout << "Internal field: " << endl;
	for(int z = 0; z < (*this).getDimZ(); z++)
	{
		cout << endl << "z = " << z << ":" << endl << endl;
		for(int y = 0; y < (*this).getDimY(); y++)
		{
			for(int x = 0; x < (*this).getDimX(); x++)
			{
				
				cout << std::setprecision(2) << (*this).getData(x,y,z) << " ";
			}

			cout << endl;
		}
	}	
	cout << endl << endl;
}

double InternalField::getData(int _x, int _y, int _z)
{
	long index = (*this).getIndex(_x, _y, _z);
	return this->data[index];			
}