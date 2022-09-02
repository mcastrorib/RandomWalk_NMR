// include OpenCV core functions
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>

// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

// include RNG lib
#include <random>

//include
#include "walker.h"
#include "../BitBlock/bitBlock.h"
#include "../RNG/xorshift.h"
#include "../NMR_Simulation/NMR_Simulation.h"

using namespace std;
using namespace cv;

// Class Point3D
// Class methods
Point3D::Point3D() : x(0), y(0), z(0) {}
Point3D::Point3D(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}

// Class Pore3D
// Class methods

// Class Walker3D
// Class methods

// default constructors
Walker::Walker() : position_x(0),
                   position_y(0),
                   position_z(0),
                   nextDirection(None),
                   surfaceRelaxivity(WALKER_DEFAULT_RHO),
                   energy(WALKER_INITIAL_ENERGY),
                   decreaseFactor(1.0),
                   collisions(0),
                   tCollisions(0),
                   xi_rate(0.0),
                   initialSeed(0),
                   currentSeed(0)
{
    initialPosition.x = position_x;
    initialPosition.y = position_y;
    initialPosition.z = position_z;

    // Define methods 'map' and 'walk' according to problem's dimension
    associateMap(true);
    associateWalk(true);
}

// default constructor with dimensions defined
Walker::Walker(bool _3rdDim) : position_x(0),
                               position_y(0),
                               position_z(0),
                               nextDirection(None),
                               surfaceRelaxivity(WALKER_DEFAULT_RHO),
                               energy(WALKER_INITIAL_ENERGY),
                               decreaseFactor(1.0),
                               collisions(0),
                               tCollisions(0),
                               xi_rate(0.0),
                               initialSeed(0),
                               currentSeed(0)
{
    initialPosition.x = position_x;
    initialPosition.y = position_y;
    initialPosition.z = position_z;

    // Define methods 'map' and 'walk' according to walker dimensionality
    (*this).associateMap(_3rdDim);
    (*this).associateWalk(_3rdDim);
}

// default constructor with positions defined
Walker::Walker(int _x, int _y, int _z, bool _3rdDim) : position_x(_x),
                                                       position_y(_y),
                                                       position_z(_z),
                                                       nextDirection(None),
                                                       surfaceRelaxivity(WALKER_DEFAULT_RHO),
                                                       energy(WALKER_INITIAL_ENERGY),
                                                       decreaseFactor(1.0),
                                                       collisions(0),
                                                       tCollisions(0),
                                                       initialSeed(0),
                                                       currentSeed(0)
{
    initialPosition.x = position_x;
    initialPosition.y = position_y;
    initialPosition.z = position_z;

    // Define methods 'map' and 'walk' according to walker's dimensionality
    (*this).associateMap(_3rdDim);
    (*this).associateWalk(_3rdDim);
}

// copy constructor
Walker::Walker(const Walker &_walker)
{
    this->initialPosition = _walker.initialPosition;
    this->position_x = _walker.position_x;
    this->position_y = _walker.position_y;
    this->position_z = _walker.position_z;
    this->nextDirection = _walker.nextDirection;

    this->initialSeed = _walker.initialSeed;
    this->currentSeed = _walker.currentSeed;

    this->surfaceRelaxivity = _walker.surfaceRelaxivity;
    this->decreaseFactor = _walker.decreaseFactor;
    this->collisions = _walker.collisions;
    this->tCollisions = _walker.tCollisions;
    this->energy = _walker.energy;
    this->xi_rate = _walker.xi_rate;

    this->mapPointer = _walker.mapPointer;
    this->walkPointer = _walker.walkPointer;
}

void Walker::associateMap(bool _3rdDim)
{
    if (_3rdDim == true)
    {
        mapPointer = &Walker::map_3D;
    }
    else
    {
        mapPointer = &Walker::map_2D;
    }
}

void Walker::associateWalk(bool _3rdDim)
{
    if (_3rdDim == true)
    {
        walkPointer = &Walker::walk_3D;
    }
    else
    {
        walkPointer = &Walker::walk_2D;
    }
}

void Walker::map(BitBlock &_bitBlock)
{
    (this->*mapPointer)(_bitBlock);
}

void Walker::walk(BitBlock &_bitBlock)
{
    (this->*walkPointer)(_bitBlock);
}

void Walker::map_2D(BitBlock &_bitBlock)
{
    computeNextDirection_2D();
    checkBorder_2D(_bitBlock);
    Point3D nextPosition = computeNextPosition_2D();

    //check if next position is pore wall
    if (checkNextPosition_2D(nextPosition, _bitBlock))
    {
        moveWalker(nextPosition);
    }
    else
    {
        // walker crashes with wall and "comes back" to the same position
        this->collisions++;
    }
}

void Walker::walk_2D(BitBlock &_bitBlock)
{
    computeNextDirection_2D();
    checkBorder_2D(_bitBlock);
    Point3D nextPosition = computeNextPosition_2D();

    if (checkNextPosition_2D(nextPosition, _bitBlock))
    {
        moveWalker(nextPosition);
    }
    else
    {
        // walker chocks with wall and comes back to the same position
        // walker loses energy due to this collision
        this->energy = this->energy * decreaseFactor;
    }
}

void Walker::map_3D(BitBlock &_bitBlock)
{
    computeNextDirection_3D();
    checkBorder_3D(_bitBlock);
    Point3D nextPosition = computeNextPosition_3D();

    //check if next position is pore wall
    if (checkNextPosition_3D(nextPosition, _bitBlock))
    {
        moveWalker(nextPosition);
    }
    else
    {
        // walker crashes with wall and "comes back" to the same position
        this->collisions++;
    }
}

void Walker::walk_3D(BitBlock &_bitBlock)
{
    computeNextDirection_3D();
    checkBorder_3D(_bitBlock);
    Point3D nextPosition = computeNextPosition_3D();

    if (checkNextPosition_3D(nextPosition, _bitBlock))
    {
        moveWalker(nextPosition);
    }
    else
    {
        // walker chocks with wall and comes back to the same position
        // walker loses energy due to this collision
        this->energy = this->energy * this->decreaseFactor;
    }
}

// generate seed for walker's random movement
void Walker::createRandomSeed()
{
    std::mt19937_64 myRNG;
    std::random_device device;
    myRNG.seed(device());
    std::uniform_int_distribution<uint64_t> uint64_dist;

    uint64_t garbageSeed;

    for (uint i = 0; i < 100; i++)
    {
        // assign a random seed to the simulation
        garbageSeed = uint64_dist(myRNG);
    }

    // assign a random seed to the simulation
    this->initialSeed = uint64_dist(myRNG) + 1;
    this->currentSeed = this->initialSeed;
}

void Walker::setRandomSeed(uint64_t _seed)
{
    this->initialSeed = _seed;
    this->currentSeed = this->initialSeed;
}