#ifndef VEC3_H_
#define VEC3_H_

// include C++ standard libraries
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

class Vector3D
{
public:
    // pore position
    double x;
    double y;
    double z;

private:
    double _norm;

public:

    // Pore methods:
    // default constructors
    Vector3D();
    Vector3D(double _x, double _y, double _z);

    //copy constructors
    Vector3D(const Vector3D &_otherVec3);

    // default destructor
    virtual ~Vector3D()
    {
        // cout << "Vector3D object destroyed succesfully" << endl;
    }

    // set methods
    void setX(double _x) { this->x = _x; }
    void setY(double _y) { this->y = _y; }
    void setZ(double _z) { this->z = _z; }
    void setNorm()
    {
        this->_norm = sqrt((*this).getX()*(*this).getX() + 
                           (*this).getY()*(*this).getY() + 
                           (*this).getZ()*(*this).getZ());
    }

    // get methods
    double getX() const { return this->x; }
    double getY() const { return this->y; }
    double getZ() const { return this->z; }
    double getNorm()  
    { 
        return this->_norm; 
    }

    void printInfo();
    string getInfo();

    // LA methods
    void addVector(Vector3D _vec);
    void subtractVector(Vector3D _vec);
    double dotProduct(Vector3D _vec);
    void crossProduct(Vector3D _vec);

};

#endif