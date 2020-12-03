// include C++ standard libraries
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// include header file
#include "Vector3D.h"

using namespace std;

// Vector3D methods
// default constructor
Vector3D::Vector3D() : x(0.0), y(0.0), z(0.0)
{
    (*this).setNorm();
}

Vector3D::Vector3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z)
{
    (*this).setNorm();
}

// copy constructor
Vector3D::Vector3D(const Vector3D &_otherVec3)
{
    // image attributes
    (*this).setX(_otherVec3.getX());
    (*this).setY(_otherVec3.getY());
    (*this).setZ(_otherVec3.getZ());
    (*this).setNorm();
}

void Vector3D::printInfo()
{
    cout << "{" << (*this).getX();
    cout << ", " << (*this).getY();
    cout << ", " << (*this).getZ();
    cout << "}" << endl;
}

string Vector3D::getInfo()
{
    string info;
    info = "{";
    info = info + std::to_string(this->x) + ", ";
    info = info + std::to_string(this->y) + ", ";
    info = info + std::to_string(this->z) + "}\n";

    return info;
}

// LA methods

void Vector3D::addVector(Vector3D _vec)
{
    (*this).setX((*this).getX() + _vec.getX());
    (*this).setY((*this).getY() + _vec.getY());
    (*this).setZ((*this).getZ() + _vec.getZ());
    (*this).setNorm();

    return;
}

void Vector3D::subtractVector(Vector3D _vec)
{

    (*this).setX((*this).getX() - _vec.getX());
    (*this).setY((*this).getY() - _vec.getY());
    (*this).setZ((*this).getZ() - _vec.getZ());
    (*this).setNorm();
    
    return;
}

double Vector3D::dotProduct(Vector3D _vec)
{
    return ((*this).getX() * _vec.getX() + 
            (*this).getY() * _vec.getY() + 
            (*this).getZ() * _vec.getZ());
}

void Vector3D::crossProduct(Vector3D _vec)
{
    Vector3D tempVec;
    tempVec.setX((*this).getY()*_vec.getZ() - (*this).getZ()*_vec.getY());
    tempVec.setY((*this).getZ()*_vec.getX() - (*this).getX()*_vec.getZ());
    tempVec.setZ((*this).getX()*_vec.getY() - (*this).getY()*_vec.getX());

    (*this).setX(tempVec.getX());
    (*this).setY(tempVec.getY());
    (*this).setZ(tempVec.getZ());

    (*this).setNorm();
    return;
}
