#ifndef BITBLOCK_H_
#define BITBLOCK_H_

#include <stdint.h>

// include OpenCV core functions
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>

// define 2D block properties
#define ROWSPERBLOCK2D 8
#define COLUMNSPERBLOCK2D 8
#define IDX2C(i, j, ld) (((i) * (ld)) + (j))

// define 3D block properties
#define ROWSPERBLOCK3D 4
#define COLUMNSPERBLOCK3D 4
#define DEPTHPERBLOCK3D 4
#define IDX2C_3D(i, j, k, lx, ly) ((k * (lx * ly)) + ((j) * (lx)) + (i))

using namespace std;
using namespace cv;

class BitBlock
{
public:
    uint64_t *blocks;
    int numberOfBlocks;
    int imageRows;
    int imageColumns;
    int imageDepth;
    int blockRows;
    int blockColumns;
    int blockDepth;

    // methods:

    // class constructor
    BitBlock();

    // class constructor by copy
    BitBlock(const BitBlock &_bitBlock);

    // class destructor
    virtual ~BitBlock()
    {
        delete[] this->blocks;
        this->blocks = NULL;
    };

    void clear()
    {
        this->numberOfBlocks = 0;
        this->imageRows = 0;
        this->imageColumns = 0;
        this->imageDepth = 0;
        this->blockRows = 0;
        this->blockColumns = 0;
        this->blockDepth = 0;
        if(this->blocks != NULL)
        {
            delete[] this->blocks;
            this->blocks = NULL;
        }
    }

    void createBlockMap(vector<Mat> &_binaryMap);
    void setBlockMapDimensions_2D(Mat &_binaryMap);
    void createBitBlocksArray_2D(Mat &_binaryMap);
    void saveBitBlockArray_2D(string filename);
    void setBlockMapDimensions_3D(vector<Mat> &_binaryMap);
    void createBitBlocksArray_3D(vector<Mat> &_binaryMap);
    void saveBitBlockArray_3D(string filename);

    // inline methods
    inline void saveBitBlockArray(string filename)
    {
        if (this->imageDepth == 1)
        {
            saveBitBlockArray_2D(filename);
        }
        else
        {
            saveBitBlockArray_3D(filename);
        }
    }

    // 2D block
    inline int findBlock(int _position_x, int _position_y)
    {
        // "x >> 3" is like "x / 8" in bitwise operation
        int block_x = _position_x >> 3;
        int block_y = _position_y >> 3;
        int blockIndex = block_x + block_y * this->blockColumns;

        return blockIndex;
    }

    inline int findBitInBlock(int _position_x, int _position_y)
    {
        // "x & (n - 1)" is lise "x % n" in bitwise operation
        int bit_x = _position_x & (COLUMNSPERBLOCK2D - 1);
        int bit_y = _position_y & (ROWSPERBLOCK2D - 1);
        // "x << 3" is like "x * 8" in bitwise operation
        int bitIndex = bit_x + (bit_y << 3);

        return bitIndex;
    }

    // 3D block
    inline int findBlock(int _position_x, int _position_y, int _position_z)
    {
        // "x >> 3" is like "x / 8" in bitwise operation
        int block_x = _position_x >> 2;
        int block_y = _position_y >> 2;
        int block_z = _position_z >> 2;
        int blockIndex = block_x + block_y * this->blockColumns + (block_z * (this->blockColumns * this->blockRows));

        return blockIndex;
    }

    inline int findBitInBlock(int _position_x, int _position_y, int _position_z)
    {
        // "x & (n - 1)" is "x % n" in bitwise operation
        int bit_x = _position_x & (COLUMNSPERBLOCK3D - 1);
        int bit_y = _position_y & (ROWSPERBLOCK3D - 1);
        int bit_z = _position_z & (DEPTHPERBLOCK3D - 1);
        // "x << 3" is like "x * 8" in bitwise operation
        int bitIndex = bit_x + (bit_y << 2) + ((bit_z << 2) << 2);

        return bitIndex;
    }

    inline bool checkIfBitIsWall(int _blockIndex, int _bitIndex)
    {
        return ((this->blocks[_blockIndex] >> _bitIndex) & 1ull);
    }

    // inline 'get' methods
    // attributes
    inline int getNumberOfBlocks() { return this->numberOfBlocks; }
    inline int getImageRows() { return this->imageRows; }
    inline int getImageColumns() { return this->imageColumns; }
    inline int getImageDepth() { return this->imageDepth; }
    inline int getBlockRows() { return this->blockRows; }
    inline int getBlockColumns() { return this->blockColumns; }
    inline int getBlockDepth() { return this->blockDepth; }

    // bitblock array
    inline uint64_t *getBitBlockArray() { return this->blocks; }
    inline uint64_t getBitBlock(int id) { return this->blocks[id]; }
};

#endif