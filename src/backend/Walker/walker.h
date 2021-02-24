#ifndef WALKER_H_
#define WALKER_H_

#include <iostream>
#include <vector>
#include <math.h>
#include "walker_defs.h"
#include "../BitBlock/bitBlock.h"
#include "../RNG/xorshift.h"

inline void showblock(uint64_t x)
{
    int lineCounter = 0;

    for (int i = 0; i < (sizeof(uint64_t) * 8); i++)
    {
        (x & (1ull << i) ? cout << "1" : cout << "0");
        cout << "  ";

        if (lineCounter == 7)
        {
            cout << endl;
            lineCounter = -1;
        }

        lineCounter++;
    }
    cout << endl;
}

using namespace std;
using namespace cv;

class Pore
{
public:
    // Class attributes:
    int position_x, position_y, position_z;
};

class Point3D
{
public:
    int x, y, z;

    // methods
    Point3D();
    Point3D(int _x, int _y, int _z);

    inline bool isPore(vector<Mat> &_binaryMap)
    {
        uchar *mapPixel = _binaryMap[this->z].ptr<uchar>(this->y);

        if (mapPixel[this->x] == 0)
            return true;
        else
            return false;
    };

    inline bool isPore(Mat &_binaryMap)
    {
        uchar *mapPixel = _binaryMap.ptr<uchar>(this->y);

        if (mapPixel[this->x] == 0)
            return true;
        else
            return false;
    };

    inline void printInfo()
    {
        cout << "{" << this->x << ", " << this->y << ", " << this->z << "}" << endl;
    };
};

class Walker
{
public:
    // Object attributes:
    Point3D initialPosition;
    int position_x, position_y, position_z;
    direction nextDirection;

    // RNG seeds
    uint64_t initialSeed;
    uint64_t currentSeed;

    // physical properties
    double surfaceRelaxivity;
    double decreaseFactor;
    uint collisions;
    uint tCollisions;
    double xi_rate;
    double energy;

    // default constructor
    Walker();
    Walker(bool _3rdDim);
    Walker(int _x, int _y, int _z, bool _3rdDim);
    // copy constructor
    Walker(const Walker &_walker);
    // virtual destructor
    virtual ~Walker(){};

    //Class methods:
    void createRandomSeed();
    void setInitialSeed(uint64_t _seed){ this->initialSeed = _seed; }

    // supermethods
    typedef void (Walker::*map_ptr)(BitBlock &);
    typedef void (Walker::*walk_ptr)(BitBlock &);

private:
    map_ptr mapPointer;
    walk_ptr walkPointer;

    // 2D
    void map_2D(BitBlock &_bitBlock);
    void walk_2D(BitBlock &_bitBlock);

    // 3D
    void map_3D(BitBlock &_bitBlock);
    void walk_3D(BitBlock &_bitBlock);

public:
    void map(BitBlock &_bitBlock);
    void walk(BitBlock &_bitBlock);
    void associateMap(bool _3rdDim);
    void associateWalk(bool _3rdDim);

    // Inline methods
    inline void resetPosition()
    {
        this->position_x = initialPosition.x;
        this->position_y = initialPosition.y;
        this->position_z = initialPosition.z;
    };

    inline void resetCollisions()
    {
        this->collisions = 0;
    };

    inline void resetTCollisions()
    {
        this->tCollisions = 0;
    };

    inline void resetEnergy()
    {
        this->energy = WALKER_INITIAL_ENERGY;
    };

    inline void resetSeed()
    {
        this->currentSeed = this->initialSeed;
    };

    void setXIrate(double _xi_rate) { this->xi_rate = _xi_rate; }

    void updateXIrate(uint _numberOfSteps)
    {
        double steps = (double) _numberOfSteps;
        if(LOG_XIRATE) 
        {
            this->xi_rate = log10((double) this->collisions) / log10(steps);
        } else 
        {
            this->xi_rate =  this->collisions / steps;
        }
    }

    void setSurfaceRelaxivity(vector<double> &parameters)
    {

        // Set 1st Sigmoid Curve
        double K1 = parameters[0];
        double A1 = parameters[1];
        double eps1 = parameters[2];
        double B1 = parameters[3];

        double shapeFunction1;
        shapeFunction1 = A1 + ((K1 - A1) / (1 + exp((-1) * B1 * (this->xi_rate - eps1))));

        // Set 2nd Sigmoid Curve
        double K2 = parameters[4];
        double A2 = parameters[5];
        double eps2 = parameters[6];
        double B2 = parameters[7];

        double shapeFunction2;
        shapeFunction2 = A2 + ((K2 - A2) / (1 + exp((-1) * B2 * (this->xi_rate - eps2))));

        this->surfaceRelaxivity = shapeFunction1 + shapeFunction2;
    }

    void setSurfaceRelaxivity(double rho)
    {
        this->surfaceRelaxivity = rho;
    }

    inline void computeDecreaseFactor(double _walkerStepLength,
                                      double _diffusionCoefficient)
    {
        // A factor: see ref. Bergman 1995
        double Afactor = 2.0 / 3.0;
        this->decreaseFactor = 1.0 - Afactor * ((this->surfaceRelaxivity * _walkerStepLength) / (_diffusionCoefficient * 1e03));
    };

    // generate a random number using xorshift64
    inline uint64_t generateRandomNumber(uint64_t _seed)
    {
        xorshift64_state xor_state;
        xor_state.a = _seed;
        return xorShift64(&xor_state);
    }

    // 2D movements
    inline void computeNextDirection_2D()
    {
        // generate a random number using xorshift library
        uint64_t rand = generateRandomNumber(this->currentSeed);

        // update current seed for next move
        this->currentSeed = rand;

        // set direction based on the random number
        rand = rand & (4 - 1);
        this->nextDirection = (direction)(rand + 1);
    };

    inline Point3D computeNextPosition_2D()
    {
        Point3D nextPosition = {this->position_x, this->position_y, this->position_z};
        switch (nextDirection)
        {
        case North:
            nextPosition.y = nextPosition.y - 1;
            break;

        case West:
            nextPosition.x = nextPosition.x - 1;
            break;

        case South:
            nextPosition.y = nextPosition.y + 1;
            break;

        case East:
            nextPosition.x = nextPosition.x + 1;
            break;
        }

        return nextPosition;
    };

    inline void checkBorder_2D(Mat &_binaryMap)
    {
        switch (this->nextDirection)
        {
        case North:
            if (this->position_y == 0)
            {
                nextDirection = South;
            }
            break;

        case South:

            if (this->position_y == _binaryMap.rows - 1)
            {
                nextDirection = North;
            }
            break;

        case West:
            if (this->position_x == 0)
            {
                nextDirection = East;
            }
            break;

        case East:
            if (this->position_x == _binaryMap.cols - 1)
            {
                nextDirection = West;
            }
            break;
        }
    };

    inline void checkBorder_2D(const BitBlock &_bitBlock)
    {
        switch (this->nextDirection)
        {
        case North:
            if (this->position_y == 0)
            {
                nextDirection = South;
            }
            break;

        case South:

            if (this->position_y == _bitBlock.imageRows - 1)
            {
                nextDirection = North;
            }
            break;

        case West:
            if (this->position_x == 0)
            {
                nextDirection = East;
            }
            break;

        case East:
            if (this->position_x == _bitBlock.imageColumns - 1)
            {
                nextDirection = West;
            }
            break;
        }
    };

    inline bool checkNextPosition_2D(Point3D _nextPosition, Mat &_binaryMap)
    {
        return (_nextPosition.isPore(_binaryMap));
    };

    inline bool checkNextPosition_2D(Point3D _nextPosition, BitBlock &_bitBlock)
    {
        int next_x = _nextPosition.x;
        int next_y = _nextPosition.y;
        int nextBlock = _bitBlock.findBlock(next_x, next_y);
        int nextBit = _bitBlock.findBitInBlock(next_x, next_y);

        return (!_bitBlock.checkIfBitIsWall(nextBlock, nextBit));
    };

    ////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////// 3D movements
    //////////////////////////////
    inline void computeNextDirection_3D()
    {
        // generate a random number using xorshift library
        uint64_t rand = generateRandomNumber(this->currentSeed);

        // update current seed for next move
        this->currentSeed = rand;

        // set direction based on the random number
        //rand = rand % 6;
        rand = mod6(rand);
        this->nextDirection = (direction)(rand + 1);
    };

    inline Point3D computeNextPosition_3D()
    {
        Point3D nextPosition = {this->position_x, this->position_y, this->position_z};
        switch (nextDirection)
        {
        case North:
            nextPosition.y = nextPosition.y - 1;
            break;

        case West:
            nextPosition.x = nextPosition.x - 1;
            break;

        case South:
            nextPosition.y = nextPosition.y + 1;
            break;

        case East:
            nextPosition.x = nextPosition.x + 1;
            break;

        case Up:
            nextPosition.z = nextPosition.z + 1;
            break;

        case Down:
            nextPosition.z = nextPosition.z - 1;
            break;
        }

        return nextPosition;
    };

    inline void checkBorder_3D(vector<Mat> &_binaryMap)
    {
        uint slice = this->position_z;

        switch (this->nextDirection)
        {
        case North:
            if (this->position_y == 0)
            {
                nextDirection = South;
            }
            break;

        case South:

            if (this->position_y == _binaryMap[slice].rows - 1)
            {
                nextDirection = North;
            }
            break;

        case West:
            if (this->position_x == 0)
            {
                nextDirection = East;
            }
            break;

        case East:
            if (this->position_x == _binaryMap[slice].cols - 1)
            {
                nextDirection = West;
            }
            break;

        case Up:
            if (this->position_z == _binaryMap.size() - 1)
            {
                nextDirection = Down;
            }
            break;

        case Down:
            if (this->position_z == 0)
            {
                nextDirection = Up;
            }
            break;
        }
    };

    inline void checkBorder_3D(const BitBlock &_bitBlock)
    {
        switch (this->nextDirection)
        {
        case North:
            if (this->position_y == 0)
            {
                nextDirection = South;
            }
            break;

        case South:

            if (this->position_y == _bitBlock.imageRows - 1)
            {
                nextDirection = North;
            }
            break;

        case West:
            if (this->position_x == 0)
            {
                nextDirection = East;
            }
            break;

        case East:
            if (this->position_x == _bitBlock.imageColumns - 1)
            {
                nextDirection = West;
            }
            break;

        case Up:
            if (this->position_z == _bitBlock.imageDepth - 1)
            {
                nextDirection = Down;
            }
            break;

        case Down:
            if (this->position_z == 0)
            {
                nextDirection = Up;
            }
            break;
        }
    };

    inline bool checkNextPosition_3D(Point3D _nextPosition, vector<Mat> &_binaryMap)
    {
        return (_nextPosition.isPore(_binaryMap));
    };

    inline bool checkNextPosition_3D(Point3D _nextPosition, BitBlock &_bitBlock)
    {
        int next_x = _nextPosition.x;
        int next_y = _nextPosition.y;
        int next_z = _nextPosition.z;
        int nextBlock = _bitBlock.findBlock(next_x, next_y, next_z);
        int nextBit = _bitBlock.findBitInBlock(next_x, next_y, next_z);

        return (!_bitBlock.checkIfBitIsWall(nextBlock, nextBit));
    };

    inline void moveWalker(Point3D _nextPosition)
    {
        this->position_x = _nextPosition.x;
        this->position_y = _nextPosition.y;
        this->position_z = _nextPosition.z;
    };

    inline void placeWalker(uint x0 = 0, uint y0 = 0, uint z0 = 0)
    {
        this->initialPosition.x = x0;
        this->initialPosition.y = y0;
        this->initialPosition.z = z0;
        (*this).resetPosition();
    }

    inline void printPosition()
    {
        cout << "{" << position_x << ", " << position_y << ", " << position_z << "}" << endl;
    };

    inline void printPosition(Point3D position)
    {
        cout << "{" << position.x << ", " << position.y << ", " << position.z << "}" << endl;
    };

    // 'get' inline methods
    // coordinate and orientation
    inline Point3D getInitialPosition() { return this->initialPosition; }
    inline int getInitialPositionX() { return this->initialPosition.x; }
    inline int getInitialPositionY() { return this->initialPosition.y; }
    inline int getInitialPositionZ() { return this->initialPosition.z; }
    inline int getPositionX() { return this->position_x; }
    inline int getPositionY() { return this->position_y; }
    inline int getPositionZ() { return this->position_z; }
    inline direction getNextDirection() { return this->nextDirection; }

    // RNG seeds
    inline uint64_t getInitialSeed() { return this->initialSeed; }
    inline uint64_t getCurrentSeed() { return this->currentSeed; }

    // physical attributes
    inline double getSurfaceRelaxivity() { return this->surfaceRelaxivity; }
    inline double getDecreaseFactor() { return this->decreaseFactor; }
    inline uint getCollisions() { return this->collisions; }
    inline double getEnergy() { return this->energy; }

private:
    inline uint64_t xorShift64(struct xorshift64_state *state)
    {
        uint64_t x = state->a;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return state->a = x;
    }

    inline uint64_t mod6(uint64_t a)
    {
        while (a > 11)
        {
            int s = 0; /* accumulator for the sum of the digits */
            while (a != 0)
            {
                s = s + (a & 7);
                a = (a >> 2) & -2;
            }
            a = s;
        }
        /* note, at this point: a < 12 */
        if (a > 5)
            a = a - 6;
        return a;
    }
};

#endif