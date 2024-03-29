// include C++ standard libraries
#include <iostream>
#include <iomanip>      // std::setprecision
#include <sstream>
#include <fstream>
#include <stdint.h>
#include <string>

// include OpenMP for multicore implementation
#include <omp.h>

//include 
#include "NMR_defs.h"
#include "InternalField.h"

using namespace std;

InternalField::InternalField(BitBlock &_bitblock, double _resolution, double _gradient, int _direction) : dimX(0), 
																											 dimY(0), 
																											 dimZ(0), 
																											 rowScale(0), 
																											 depthScale(0), 
																											 data(NULL)
{
	(*this).setDims(_bitblock.imageColumns, _bitblock.imageRows, _bitblock.imageDepth);
	(*this).allocDataArray();
	(*this).fillDataArray(_bitblock, _resolution, _gradient, _direction);
	// (*this).show();
}

InternalField::InternalField(BitBlock &_bitblock, string _file) : dimX(0), 
																  dimY(0), 
																  dimZ(0), 
																  rowScale(0), 
																  depthScale(0), 
																  data(NULL)
{
	(*this).setDims(_bitblock.imageColumns, _bitblock.imageRows, _bitblock.imageDepth);
	(*this).allocDataArray();
	(*this).readDataFromFile(_file);
	// (*this).show();
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

void InternalField::fillDataArray(BitBlock &_bitblock, double _resolution, double _gValue, int _gDirection)
{
	double dGx = 0;
	double dGy = 0;
	double dGz = 0;
	if(_gDirection == 0) dGx = _gValue * (1.0e-6) * _resolution;
	if(_gDirection == 1) dGy = _gValue * (1.0e-6) * _resolution;
	if(_gDirection == 2) dGz = _gValue * (1.0e-6) * _resolution;
	
	double initialValue = 0.0;
	double currentValue;
	double newValue;
	long currentIndex;
	
	currentValue = initialValue;
	for(int z = 0; z < (*this).getDimZ(); z++)
	{
		if(_gDirection == 2)
			currentValue = z*dGz;

		for(int y = 0; y < (*this).getDimY(); y++)
		{
			if(_gDirection == 1) 
				currentValue = y*dGy;		
			
			for(int x = 0; x < (*this).getDimX(); x++)
			{
				if(_gDirection == 0) 
					currentValue = x*dGx;
		
				int block = _bitblock.findBlock(x, y, z);
				int bit = _bitblock.findBitInBlock(x, y, z);
				if(_bitblock.checkIfBitIsWall(block, bit))
					newValue = 0.0;
				else
					newValue = currentValue;
								
				currentIndex = (*this).getIndex(x, y, z);
				(*this).fillData(currentIndex, newValue);
			}
		}
	}
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
				
				cout << std::setprecision(3) << (*this).getData(x,y,z) << " ";
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

void InternalField::readDataFromFile(string _file)
{
	cout << "- reading internal field from file: " << endl << _file << endl;
	ifstream datafile(_file, ios::in | ios::binary);
	if(!datafile) {
	  cout << "Cannot open file!" << endl;
	  exit(1);
	}

	// check data file size
	datafile.seekg(0, datafile.end);
    long fsize = datafile.tellg() / sizeof(double);
    long fieldVolume = (*this).getDimZ()*(*this).getDimY()*(*this).getDimX();
    if(fsize != fieldVolume)
	{
	  cout << "Error: dims from file don't match!" << endl;
	  cout << fsize << " vs. " << fieldVolume << endl;
      exit(1);
	}

    // If sizes match
	datafile.seekg(0, datafile.beg);
	datafile.read((char *)this->data, fsize*sizeof(double));

	// double inData;
	// long index;
	// for(int z = 0; z < (*this).getDimZ(); z++)
	// {
	// 	for(int y = 0; y < (*this).getDimY(); y++)
	// 	{		
	// 		for(int x = 0; x < (*this).getDimX(); x++)
	// 		{					
	// 			index = (*this).getIndex(x, y, z);
	// 			datafile.read((char *) &inData, sizeof(double));				
	// 			(*this).fillData(index, inData);
	// 		}
	// 	}
	// }

	datafile.close();

	if(!datafile.good()) 
	{
	  cout << "Error occurred at reading time!" << endl;
	  exit(1);
	}
}