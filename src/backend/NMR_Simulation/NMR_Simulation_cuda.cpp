#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <cstdint>
#include <random>
#include <vector>
#include <string>

// include C standard library for memory allocation using pointers
#include <stdlib.h>

// include OpenMP for multicore implementation
#include <omp.h>
#include "../Utils/OMPLoopEnabler.h"
#include "../Utils/myAllocator.h"

//include
#include "../Walker/walker.h"
#include "../RNG/xorshift.h"
#include "NMR_Simulation.h"
#include "NMR_Simulation_cuda.h"

// GPU kernel for Mapping simulation - a.k.a. walker's collision count
// in this kernel, each thread will behave as a solitary walker
__global__ void map_2D( int *walker_px,
                        int *walker_py,
                        uint *collisions,
                        uint64_t *seed,
                        const uint64_t *bitBlock,
                        const int bitBlockColumns,
                        const uint numberOfWalkers,
                        const uint numberOfSteps,
                        const int map_columns,
                        const int map_rows,
                        const uint shift_convert)
    {

    // identify thread's walker
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Unique read from device global memory
    int position_x, position_y;
    int thread_collisions;
    uint64_t local_seed; 

    // thread variables for future movements
    int next_x, next_y;
    direction nextDirection = None;

    if (index < numberOfWalkers)
    {
        // Unique read from device global memory
        position_x = walker_px[index];
        position_y = walker_py[index];
        thread_collisions = 0;
        local_seed = seed[index];

        for (int step = 0; step < numberOfSteps; step++)
        {
            nextDirection = computeNextDirection_2D(local_seed);

            nextDirection = checkBorder_2D( position_x,
                                            position_y,
                                            nextDirection,
                                            map_columns,
                                            map_rows);

            computeNextPosition_2D( position_x,
                                    position_y,
                                    nextDirection,
                                    next_x,
                                    next_y);

            if (checkNextPosition_2D(next_x, next_y, bitBlock, bitBlockColumns))
            {
                // position is valid
                position_x = next_x;
                position_y = next_y;
            }
            else
            {
                // position is pore wall
                thread_collisions++;
            }
        }

        // device global memory update
        collisions[index] = thread_collisions;
        walker_px[index] = position_x;
        walker_py[index] = position_y;
        seed[index] = local_seed;
    }
}

// function to call GPU kernel to execute
// walker's "map" method in Graphics Processing Unit
void NMR_Simulation::mapSimulation_CUDA_2D()
{
    cout << "initializing mapping simulation in GPU... ";
    // reset walkers
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        this->walkers[id].resetPosition();
        this->walkers[id].resetSeed();
        this->walkers[id].resetCollisions();
    }

    // initialize histograms
    (*this).initHistogramList();

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values
    int bitBlockColumns = this->bitBlock.blockColumns;
    int numberOfBitBlocks = this->bitBlock.numberOfBlocks;
    uint numberOfWalkers = this->numberOfWalkers;
    uint numberOfSteps = this->simulationSteps;
    int map_columns = this->bitBlock.imageColumns;
    int map_rows = this->bitBlock.imageRows;
    uint shiftConverter = log2(this->voxelDivision);

    // Copy bitBlock2D data from host to device (only once)
    // assign pointer to bitBlock datastructure
    uint64_t *bitBlock;
    bitBlock = this->bitBlock.blocks;

    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // pointers used in array conversion and their host memory allocation
    myAllocator arrayFactory;
    int *walker_px = arrayFactory.getIntArray(numberOfWalkers);
    int *walker_py = arrayFactory.getIntArray(numberOfWalkers);
    uint *collisions = arrayFactory.getUIntArray(numberOfWalkers);
    uint64_t *seed = arrayFactory.getUInt64Array(numberOfWalkers);

    // Host data copy
    // copy original walkers' data to temporary host arrays
    for (uint i = 0; i < numberOfWalkers; i++)
    {
        walker_px[i] = this->walkers[i].initialPosition.x;
        walker_py[i] = this->walkers[i].initialPosition.y;
        collisions[i] = 0;
        seed[i] = this->walkers[i].initialSeed;
    }

    // Device memory allocation
    // Declaration of device data arrays
    int *d_walker_px;
    int *d_walker_py;
    uint *d_collisions;
    uint64_t *d_seed;

    // alloc memory in device for data arrays
    cudaMalloc((void **)&d_walker_px, numberOfWalkers * sizeof(int));
    cudaMalloc((void **)&d_walker_py, numberOfWalkers * sizeof(int));
    cudaMalloc((void **)&d_collisions, numberOfWalkers * sizeof(uint));
    cudaMalloc((void **)&d_seed, numberOfWalkers * sizeof(uint64_t));

    // Device data copy
    // copy host data to device
    cudaMemcpy(d_walker_px, walker_px, numberOfWalkers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_walker_py, walker_py, numberOfWalkers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_collisions, collisions, numberOfWalkers * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed, seed, numberOfWalkers * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Launch kernel for GPU computation
    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    int threadsPerBlock = this->rwNMR_config.getThreadsPerBlock();
    int blocksPerKernel = (int)ceil(double(numberOfWalkers) / double(threadsPerBlock));

    // kernel "map" launch
    map_2D<<<threadsPerBlock, blocksPerKernel>>>(d_walker_px,
                                                 d_walker_py,
                                                 d_collisions,
                                                 d_seed,
                                                 d_bitBlock,
                                                 bitBlockColumns,
                                                 numberOfWalkers,
                                                 numberOfSteps,
                                                 map_columns,
                                                 map_rows,
                                                 shiftConverter);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Host data copy
    // copy device data to host
    cudaMemcpy(collisions, d_collisions, numberOfWalkers * sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(walker_px, d_walker_px, numberOfWalkers * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(walker_py, d_walker_py, numberOfWalkers * sizeof(int), cudaMemcpyDeviceToHost);


    // copy collisions host data to class members
    for (uint id = 0; id < numberOfWalkers; id++)
    {
        this->walkers[id].collisions = collisions[id];
        this->walkers[id].position_x = walker_px[id];
        this->walkers[id].position_y = walker_py[id];
    }

    // create collision histogram
    (*this).createHistogram();

    // free pointers in host
    free(walker_px);
    free(walker_py);
    free(collisions);
    free(seed);

    // and direct them to NULL
    walker_px = NULL;
    walker_py = NULL;
    collisions = NULL;
    seed = NULL;

    // also direct the bitBlock pointer created in this context
    // (original data is kept safe)
    bitBlock = NULL;

    // free device global memory
    cudaFree(d_walker_px);
    cudaFree(d_walker_py);
    cudaFree(d_collisions);
    cudaFree(d_seed);
    cudaFree(d_bitBlock);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Completed.\telapsed time: " << elapsedTime * 1.0e-3 << endl;
}

void NMR_Simulation::mapSimulation_CUDA_2D_histograms()
{
    cout << "initializing mapping simulation in GPU... ";
    // reset walkers
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        this->walkers[id].resetPosition();
        this->walkers[id].resetSeed();
        this->walkers[id].resetCollisions();
        this->walkers[id].resetTCollisions();
    }

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values
    int bitBlockColumns = this->bitBlock.blockColumns;
    int numberOfBitBlocks = this->bitBlock.numberOfBlocks;
    uint numberOfWalkers = this->numberOfWalkers;
    int map_columns = this->bitBlock.imageColumns;
    int map_rows = this->bitBlock.imageRows;
    uint shiftConverter = log2(this->voxelDivision);

    // Launch kernel for GPU computation
    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    int threadsPerBlock = this->rwNMR_config.getThreadsPerBlock();
    int blocksPerKernel = (int)ceil(double(numberOfWalkers) / double(threadsPerBlock));

    // Copy bitBlock2D data from host to device (only once)
    // assign pointer to bitBlock datastructure
    uint64_t *bitBlock;
    bitBlock = this->bitBlock.blocks;

    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // pointers used in array conversion and their host memory allocation
    myAllocator arrayFactory;
    int *walker_px = arrayFactory.getIntArray(numberOfWalkers);
    int *walker_py = arrayFactory.getIntArray(numberOfWalkers);
    uint *collisions = arrayFactory.getUIntArray(numberOfWalkers);
    uint64_t *seed = arrayFactory.getUInt64Array(numberOfWalkers);

    // Device memory allocation
    // Declaration of device data arrays
    int *d_walker_px;
    int *d_walker_py;
    uint *d_collisions;
    uint64_t *d_seed;

    // alloc memory in device for data arrays
    cudaMalloc((void **)&d_walker_px, numberOfWalkers * sizeof(int));
    cudaMalloc((void **)&d_walker_py, numberOfWalkers * sizeof(int));
    cudaMalloc((void **)&d_collisions, numberOfWalkers * sizeof(uint));
    cudaMalloc((void **)&d_seed, numberOfWalkers * sizeof(uint64_t));

    // initialize histograms
    (*this).initHistogramList();

    // loop throughout histogram list
    for(int hst_ID = 0; hst_ID < this->histogramList.size(); hst_ID++)
    {
        // set steps for each histogram
        uint eBegin = this->histogramList[hst_ID].firstEcho;
        uint eEnd = this->histogramList[hst_ID].lastEcho;
        uint steps = this->stepsPerEcho * (eEnd - eBegin);

        // Host data copy
        // copy original walkers' data to temporary host arrays
        for (uint id = 0; id < numberOfWalkers; id++)
        {
            walker_px[id] = this->walkers[id].position_x;
            walker_py[id] = this->walkers[id].position_y;
            collisions[id] = 0;
            seed[id] = this->walkers[id].currentSeed;
        }

        // Device data copy
        // copy host data to device
        cudaMemcpy(d_collisions, collisions, numberOfWalkers * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_px, walker_px, numberOfWalkers * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, walker_py, numberOfWalkers * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, seed, numberOfWalkers * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // kernel "map" launch
        map_2D<<<threadsPerBlock, blocksPerKernel>>>(d_walker_px,
                                                    d_walker_py,
                                                    d_collisions,
                                                    d_seed,
                                                    d_bitBlock,
                                                    bitBlockColumns,
                                                    numberOfWalkers,
                                                    steps,
                                                    map_columns,
                                                    map_rows,
                                                    shiftConverter);
        cudaDeviceSynchronize();

        // Host data copy
        // copy device data to host
        cudaMemcpy(collisions, d_collisions, numberOfWalkers * sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(walker_px, d_walker_px, numberOfWalkers * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(walker_py, d_walker_py, numberOfWalkers * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(seed, d_seed, numberOfWalkers * sizeof(uint64_t), cudaMemcpyDeviceToHost);


        // copy collisions host data to class members
        for (uint id = 0; id < numberOfWalkers; id++)
        {
            this->walkers[id].collisions = collisions[id];
            this->walkers[id].position_x = walker_px[id];
            this->walkers[id].position_y = walker_py[id];
            this->walkers[id].currentSeed = seed[id];
        }

        // create histogram
        (*this).createHistogram(hst_ID, steps);

        // reset collision count, but keep summation in alternative count
        for (uint id = 0; id < numberOfWalkers; id++)
        {
            this->walkers[id].tCollisions += this->walkers[id].collisions;
            this->walkers[id].resetCollisions();
        }
    }
    // histogram loop is finished

    // recover walkers collisions from total sum and create a global histogram
    for (uint id = 0; id < this->numberOfWalkers; id++)
    {
        this->walkers[id].collisions = this->walkers[id].tCollisions;   
    }

    // create collision histogram
    (*this).createHistogram();

    // free pointers in host
    free(walker_px);
    free(walker_py);
    free(collisions);
    free(seed);

    // and direct them to NULL
    walker_px = NULL;
    walker_py = NULL;
    collisions = NULL;
    seed = NULL;

    // also direct the bitBlock pointer created in this context
    // (original data is kept safe)
    bitBlock = NULL;

    // free device global memory
    cudaFree(d_walker_px);
    cudaFree(d_walker_py);
    cudaFree(d_collisions);
    cudaFree(d_seed);
    cudaFree(d_bitBlock);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Completed.\telapsed time: " << elapsedTime * 1.0e-3 << endl;
}

/////////////////////////////////////////////////////////////////////
//////////////////////// DEVICE FUNCTIONS ///////////////////////////
/////////////////////////////////////////////////////////////////////

__device__ direction computeNextDirection_2D(uint64_t &seed)
{
    // generate random number using xorshift algorithm
    xorshift64_state xor_state;
    xor_state.a = seed;
    seed = xorShift64_2D(&xor_state);

    // set direction based on the random number
    direction nextDirection = (direction)((seed % 4) + 1);
    return nextDirection;
}

__device__ uint64_t xorShift64_2D(struct xorshift64_state *state)
{
    uint64_t x = state->a;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return state->a = x;
}

__device__ direction checkBorder_2D(int walker_px,
                                    int walker_py,
                                    direction &nextDirection,
                                    const int map_columns,
                                    const int map_rows)
{
    switch (nextDirection)
    {
    case North:
        if (walker_py == 0)
        {
            nextDirection = South;
        }
        break;

    case South:

        if (walker_py == map_rows - 1)
        {
            nextDirection = North;
        }
        break;

    case West:
        if (walker_px == 0)
        {
            nextDirection = East;
        }
        break;

    case East:
        if (walker_px == map_columns - 1)
        {
            nextDirection = West;
        }
        break;
    }

    return nextDirection;
}

__device__ void computeNextPosition_2D( int &walker_px,
                                        int &walker_py,
                                        direction nextDirection,
                                        int &next_x,
                                        int &next_y)
{
    next_x = walker_px;
    next_y = walker_py;

    switch (nextDirection)
    {
    case North:
        next_y = next_y - 1;
        break;

    case West:
        next_x = next_x - 1;
        break;

    case South:
        next_y = next_y + 1;
        break;

    case East:
        next_x = next_x + 1;
        break;
    }
}

__device__ bool checkNextPosition_2D(int next_x,
                                     int next_y,
                                     const uint64_t *bitBlock,
                                     const int bitBlockColumns)
{
    int blockIndex = findBlockIndex_2D(next_x, next_y, bitBlockColumns);
    int nextBit = findBitIndex_2D(next_x, next_y);
    uint64_t nextBlock = bitBlock[blockIndex];

    return (!checkIfBlockBitIsWall_2D(nextBlock, nextBit));
};

__device__ int findBlockIndex_2D(int next_x, int next_y, int bitBlockColumns)
{
    // "x >> 3" is like "x / 8" in bitwise operation
    int block_x = next_x >> 3;
    int block_y = next_y >> 3;
    int blockIndex = block_x + block_y * bitBlockColumns;

    return blockIndex;
}

__device__ int findBitIndex_2D(int next_x, int next_y)
{
    // "x & (n - 1)" is lise "x % n" in bitwise operation
    int bit_x = next_x & (COLUMNSPERBLOCK2D - 1);
    int bit_y = next_y & (ROWSPERBLOCK2D - 1);
    // "x << 3" is like "x * 8" in bitwise operation
    int bitIndex = bit_x + (bit_y << 3);

    return bitIndex;
}

__device__ bool checkIfBlockBitIsWall_2D(uint64_t nextBlock, int nextBit)
{
    return ((nextBlock >> nextBit) & 1ull);
}

__device__ uint convertLocalToGlobal_2D(uint _localPos, uint _shiftConverter)
{
    return (_localPos >> _shiftConverter);
}

// --------------
// -- 3D

// GPU kernel for Mapping simulation - a.k.a. walker's collision count
// in this kernel, each thread will behave as an unique walker
__global__ void map_3D(int *walker_px,
                       int *walker_py,
                       int *walker_pz,
                       uint *collisions,
                       uint64_t *seed,
                       const uint64_t *bitBlock,
                       const uint bitBlockColumns,
                       const uint bitBlockRows,
                       const uint numberOfWalkers,
                       const uint numberOfSteps,
                       const uint map_columns,
                       const uint map_rows,
                       const uint map_depth,
                       const uint shift_convert)
{

    // identify thread's walker
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Unique read from device global memory
    int position_x, position_y, position_z;
    uint thread_collisions;
    uint64_t local_seed;

    // thread variables for future movements
    int next_x, next_y, next_z;
    direction nextDirection = None;

    if (index < numberOfWalkers)
    {
        // Unique read from device global memory
        position_x = walker_px[index];
        position_y = walker_py[index];
        position_z = walker_pz[index];
        thread_collisions = collisions[index];
        local_seed = seed[index];

        for (int step = 0; step < numberOfSteps; step++)
        {
            nextDirection = computeNextDirection_3D(local_seed);

            nextDirection = checkBorder_3D(convertLocalToGlobal_3D(position_x, shift_convert),
                                           convertLocalToGlobal_3D(position_y, shift_convert),
                                           convertLocalToGlobal_3D(position_z, shift_convert),
                                           nextDirection,
                                           map_columns,
                                           map_rows,
                                           map_depth);

            computeNextPosition_3D(position_x,
                                   position_y,
                                   position_z,
                                   nextDirection,
                                   next_x,
                                   next_y,
                                   next_z);

            if (checkNextPosition_3D(convertLocalToGlobal_3D(next_x, shift_convert),
                                     convertLocalToGlobal_3D(next_y, shift_convert),
                                     convertLocalToGlobal_3D(next_z, shift_convert),
                                     bitBlock, 
                                     bitBlockColumns, 
                                     bitBlockRows))
            {
                // position is pore space
                position_x = next_x;
                position_y = next_y;
                position_z = next_z;
            }
            else
            {
                // position is pore wall
                thread_collisions++;
            }
        }

        // device global memory update
        collisions[index] = thread_collisions;
        walker_px[index] = position_x;
        walker_py[index] = position_y;
        walker_pz[index] = position_z;
        seed[index] = local_seed;
    }
}

__global__ void map_3D_periodic(int *walker_px,
                                int *walker_py,
                                int *walker_pz,
                                uint *collisions,
                                uint64_t *seed,
                                const uint64_t *bitBlock,
                                const uint bitBlockColumns,
                                const uint bitBlockRows,
                                const uint numberOfWalkers,
                                const uint numberOfSteps,
                                const uint map_columns,
                                const uint map_rows,
                                const uint map_depth,
                                const uint shift_convert)
{

    // identify thread's walker
    int walkerId = threadIdx.x + blockIdx.x * blockDim.x;

    // Local variables for unique read from device global memory
    int localPosX, localPosY, localPosZ;
    int imgPosX, imgPosY, imgPosZ;
    uint localCollisions;
    uint64_t localSeed;

    // thread variables for future movements
    int localNextX, localNextY, localNextZ;
    // int imgNextX, imgNextY, imgNextZ;
    direction nextDirection = None;

    // now begin the "walk" procedure de facto
    if (walkerId < numberOfWalkers)
    {
        // Local variables for unique read from device global memory
        localPosX = walker_px[walkerId];
        localPosY = walker_py[walkerId];
        localPosZ = walker_pz[walkerId];
        localCollisions = collisions[walkerId];
        localSeed = seed[walkerId];
        
        for(int step = 0; step < numberOfSteps; step++)
        {
            
            nextDirection = computeNextDirection_3D(localSeed);            
        
            computeNextPosition_3D(localPosX,
                                   localPosY,
                                   localPosZ,
                                   nextDirection,
                                   localNextX,
                                   localNextY,
                                   localNextZ);

            // update img position
            imgPosX = convertLocalToGlobal_3D(localNextX, shift_convert) % map_columns;
            if(imgPosX < 0) imgPosX += map_columns;

            imgPosY = convertLocalToGlobal_3D(localNextY, shift_convert) % map_rows;
            if(imgPosY < 0) imgPosY += map_rows;

            imgPosZ = convertLocalToGlobal_3D(localNextZ, shift_convert) % map_depth;
            if(imgPosZ < 0) imgPosZ += map_depth;

            // printf("%d, %d, %d \n", imgPosX, imgPosY, imgPosZ);

            if (checkNextPosition_3D(imgPosX, 
                                     imgPosY, 
                                     imgPosZ, 
                                     bitBlock, 
                                     bitBlockColumns, 
                                     bitBlockRows))
            {
                // update real position
                localPosX = localNextX;
                localPosY = localNextY;
                localPosZ = localNextZ;                
            }
            else
            {
                // walker hits wall and comes back to the same position
                // collisions count is incremented
                localCollisions++;
            }
        }

        // position and seed device global memory update
        // must be done for each kernel
        walker_px[walkerId] = localPosX;
        walker_py[walkerId] = localPosY;
        walker_pz[walkerId] = localPosZ;
        collisions[walkerId] = localCollisions;
        seed[walkerId] = localSeed;
    }
}


// function to call GPU kernel to execute
// walker's "map" method in Graphics Processing Unit
void NMR_Simulation::mapSimulation_CUDA_3D()
{
    cout << "initializing mapping simulation 3D in GPU... ";
    // reset walkers
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        this->walkers[id].resetPosition();
        this->walkers[id].resetSeed();
        this->walkers[id].resetCollisions();
    }

    // initialize histograms
    (*this).initHistogramList();

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values
    uint bitBlockColumns = this->bitBlock.blockColumns;
    uint bitBlockRows = this->bitBlock.blockRows;
    uint numberOfBitBlocks = this->bitBlock.numberOfBlocks;
    uint numberOfWalkers = this->numberOfWalkers;
    uint numberOfSteps = this->simulationSteps;
    uint map_columns = this->bitBlock.imageColumns;
    uint map_rows = this->bitBlock.imageRows;
    uint map_depth = this->bitBlock.imageDepth;
    uint shiftConverter = log2(this->voxelDivision);

    // create a steps bucket
    uint stepsLimit =this->rwNMR_config.getMaxRWSteps();
    uint stepsSize = numberOfSteps/stepsLimit;
    vector<uint> stepsList;
    for(uint idx = 0; idx < stepsSize; idx++)
    {
        stepsList.push_back(stepsLimit);
    }
    // charge rebalance
    if((numberOfSteps % stepsLimit) > 0)
    {
        stepsSize++;
        stepsList.push_back(numberOfSteps%stepsLimit);
    }

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    uint threadsPerBlock = this->rwNMR_config.getThreadsPerBlock();
    uint blocksPerKernel = this->rwNMR_config.getBlocks();
    uint walkersPerKernel = threadsPerBlock * blocksPerKernel;
    if (numberOfWalkers < walkersPerKernel)
    {
        blocksPerKernel = (int)ceil((double)(numberOfWalkers) / (double)(threadsPerBlock));
        walkersPerKernel = threadsPerBlock * blocksPerKernel;
    }
    uint numberOfWalkerPacks = (numberOfWalkers / walkersPerKernel) + 1;
    uint lastWalkerPackSize = numberOfWalkers % walkersPerKernel;

    // bitBlock3D host to device copy
    // assign pointer to bitBlock datastructure
    uint64_t *bitBlock;
    bitBlock = this->bitBlock.blocks;

    // copy host bitblock data to temporary host arrays
    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Device memory allocation
    // pointers used in array conversion and their host memory allocation
    myAllocator arrayFactory;
    int *walker_px = arrayFactory.getIntArray(walkersPerKernel);
    int *walker_py = arrayFactory.getIntArray(walkersPerKernel);
    int *walker_pz = arrayFactory.getIntArray(walkersPerKernel);
    uint *collisions = arrayFactory.getUIntArray(walkersPerKernel);
    uint64_t *seed = arrayFactory.getUInt64Array(walkersPerKernel);

    // Device memory allocation
    // Declaration of device data arrays
    int *d_walker_px;
    int *d_walker_py;
    int *d_walker_pz;
    uint *d_collisions;
    uint64_t *d_seed;

    // alloc memory in device for data arrays
    cudaMalloc((void **)&d_walker_px, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_py, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_pz, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_collisions, walkersPerKernel * sizeof(uint));
    cudaMalloc((void **)&d_seed, walkersPerKernel * sizeof(uint64_t));

    for (uint packId = 0; packId < (numberOfWalkerPacks - 1); packId++)
    {
        // set offset in walkers vector
        uint packOffset = packId * walkersPerKernel;

        // Host data copy
        // copy original walkers' data to temporary host arrays
// #pragma omp parallel for
        for (uint i = 0; i < walkersPerKernel; i++)
        {
            walker_px[i] = this->walkers[i + packOffset].initialPosition.x;
            walker_py[i] = this->walkers[i + packOffset].initialPosition.y;
            walker_pz[i] = this->walkers[i + packOffset].initialPosition.z;
            collisions[i] = 0;
            seed[i] = this->walkers[i + packOffset].initialSeed;
        }

        // Device data copy
        // copy host data to device
        cudaMemcpy(d_walker_px, walker_px, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, walker_py, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_collisions, collisions, walkersPerKernel * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyHostToDevice);

        //////////////////////////////////////////////////////////////////////
        // Launch kernel for GPU computation
        // kernel "map" launch
        for(uint sIdx = 0; sIdx < stepsList.size(); sIdx++)
        {
            map_3D<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                         d_walker_py,
                                                         d_walker_pz,
                                                         d_collisions,
                                                         d_seed,
                                                         d_bitBlock,
                                                         bitBlockColumns,
                                                         bitBlockRows,
                                                         walkersPerKernel,
                                                         stepsList[sIdx],
                                                         map_columns,
                                                         map_rows,
                                                         map_depth,
                                                         shiftConverter);
            cudaDeviceSynchronize();
        }

        // Host data copy
        // copy device data to host
        cudaMemcpy(collisions, d_collisions, walkersPerKernel * sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(walker_px, d_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(walker_py, d_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(walker_pz, d_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        

        // copy collisions host data to class members
// #pragma omp parallel for
        for (uint id = 0; id < walkersPerKernel; id++)
        {
            this->walkers[id + packOffset].collisions = collisions[id];
            this->walkers[id + packOffset].position_x = walker_px[id];
            this->walkers[id + packOffset].position_y = walker_py[id];
            this->walkers[id + packOffset].position_z = walker_pz[id];

        }
    }

    if (lastWalkerPackSize > 0)
    { // last pack is done explicitly
        // set offset in walkers vector
        uint packOffset = (numberOfWalkerPacks - 1) * walkersPerKernel;

        // Host data copy
        // copy original walkers' data to temporary host arrays
// #pragma omp parallel for
        for (uint i = 0; i < lastWalkerPackSize; i++)
        {
            walker_px[i] = this->walkers[i + packOffset].initialPosition.x;
            walker_py[i] = this->walkers[i + packOffset].initialPosition.y;
            walker_pz[i] = this->walkers[i + packOffset].initialPosition.z;
            collisions[i] = 0;
            seed[i] = this->walkers[i + packOffset].initialSeed;
        }

        // Device data copy
        // copy host data to device
        cudaMemcpy(d_walker_px, walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_collisions, collisions, lastWalkerPackSize * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyHostToDevice);

        //////////////////////////////////////////////////////////////////////
        // Launch kernel for GPU computation
        // kernel "map" launch
        for(uint sIdx = 0; sIdx < stepsList.size(); sIdx++)
        {
            map_3D<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                         d_walker_py,
                                                         d_walker_pz,
                                                         d_collisions,
                                                         d_seed,
                                                         d_bitBlock,
                                                         bitBlockColumns,
                                                         bitBlockRows,
                                                         walkersPerKernel,
                                                         stepsList[sIdx],
                                                         map_columns,
                                                         map_rows,
                                                         map_depth,
                                                         shiftConverter);
            cudaDeviceSynchronize();
        }

        // Host data copy
        // copy device data to host
        cudaMemcpy(collisions, d_collisions, lastWalkerPackSize * sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(walker_px, d_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(walker_py, d_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(walker_pz, d_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        

        // copy collisions host data to class members
// #pragma omp parallel for
        for (uint id = 0; id < lastWalkerPackSize; id++)
        {
            this->walkers[id + packOffset].collisions = collisions[id];
            this->walkers[id + packOffset].position_x = walker_px[id];
            this->walkers[id + packOffset].position_y = walker_py[id];
            this->walkers[id + packOffset].position_z = walker_pz[id];

        }
    }
    // procedure is completed

    // create collision histogram
    (*this).createHistogram();

    // free pointers in host
    free(walker_px);
    free(walker_py);
    free(walker_pz);
    free(collisions);
    free(seed);

    // and direct them to NULL
    walker_px = NULL;
    walker_py = NULL;
    walker_pz = NULL;
    collisions = NULL;
    seed = NULL;

    // also direct the bitBlock pointer created in this context
    // (original data is kept safe)
    bitBlock = NULL;

    // free device global memory
    cudaFree(d_walker_px);
    cudaFree(d_walker_py);
    cudaFree(d_walker_pz);
    cudaFree(d_collisions);
    cudaFree(d_seed);
    cudaFree(d_bitBlock);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Completed.\telapsed time: " << elapsedTime * 1.0e-3 << endl;
}

// function to call GPU kernel to execute
// walker's "walk" method in Graphics Processing Unit
void NMR_Simulation::mapSimulation_CUDA_3D_histograms()
{
    cout << "initializing mapping simulation 3D in GPU... ";
    // reset walkers
    if(this->rwNMR_config.getOpenMPUsage())
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->walkers.size();
        int loop_start, loop_finish;

        #pragma omp parallel shared(walkers) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish(); 

            for (uint id = loop_start; id < loop_finish; id++)
            {
                this->walkers[id].resetPosition();
                this->walkers[id].resetSeed();
                this->walkers[id].resetCollisions();
                this->walkers[id].resetTCollisions();
            }
        }
    } else
    {
        for (uint id = 0; id < this->walkers.size(); id++)
        {
            this->walkers[id].resetPosition();
            this->walkers[id].resetSeed();
            this->walkers[id].resetCollisions();
            this->walkers[id].resetTCollisions();
        }
    }

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values
    uint bitBlockColumns = this->bitBlock.blockColumns;
    uint bitBlockRows = this->bitBlock.blockRows;
    uint numberOfBitBlocks = this->bitBlock.numberOfBlocks;
    uint numberOfWalkers = this->numberOfWalkers;
    uint map_columns = this->bitBlock.imageColumns;
    uint map_rows = this->bitBlock.imageRows;
    uint map_depth = this->bitBlock.imageDepth;
    uint shiftConverter = log2(this->voxelDivision);

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    uint threadsPerBlock = this->rwNMR_config.getThreadsPerBlock();
    uint blocksPerKernel = this->rwNMR_config.getBlocks();
    uint walkersPerKernel = threadsPerBlock * blocksPerKernel;
    if (numberOfWalkers < walkersPerKernel)
    {
        blocksPerKernel = (int)ceil((double)(numberOfWalkers) / (double)(threadsPerBlock));
        walkersPerKernel = threadsPerBlock * blocksPerKernel;
    }
    uint numberOfWalkerPacks = (numberOfWalkers / walkersPerKernel) + 1;
    uint lastWalkerPackSize = numberOfWalkers % walkersPerKernel;

    // bitBlock3D host to device copy
    // assign pointer to bitBlock datastructure
    uint64_t *bitBlock;
    bitBlock = this->bitBlock.blocks;

    // copy host bitblock data to temporary host arrays
    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Device memory allocation
    // pointers used in array conversion and their host memory allocation
    myAllocator arrayFactory;
    int *walker_px = arrayFactory.getIntArray(walkersPerKernel);
    int *walker_py = arrayFactory.getIntArray(walkersPerKernel);
    int *walker_pz = arrayFactory.getIntArray(walkersPerKernel);
    uint *collisions = arrayFactory.getUIntArray(walkersPerKernel);
    uint64_t *seed = arrayFactory.getUInt64Array(walkersPerKernel);

    // Device memory allocation
    // Declaration of device data arrays
    int *d_walker_px;
    int *d_walker_py;
    int *d_walker_pz;
    uint *d_collisions;
    uint64_t *d_seed;

    // alloc memory in device for data arrays
    cudaMalloc((void **)&d_walker_px, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_py, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_pz, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_collisions, walkersPerKernel * sizeof(uint));
    cudaMalloc((void **)&d_seed, walkersPerKernel * sizeof(uint64_t));

    // initialize histograms
    (*this).initHistogramList();

    // loop throughout histogram list
    for(int hst_ID = 0; hst_ID < this->histogramList.size(); hst_ID++)
    {
        // set steps for each histogram
        uint eBegin = this->histogramList[hst_ID].firstEcho;
        uint eEnd = this->histogramList[hst_ID].lastEcho;
        uint steps = this->stepsPerEcho * (eEnd - eBegin);

        // create a steps bucket
        uint stepsLimit = this->rwNMR_config.getMaxRWSteps();
        uint stepsSize = steps/stepsLimit;
        vector<uint> stepsList;
        for(uint idx = 0; idx < stepsSize; idx++)
        {
            stepsList.push_back(stepsLimit);
        }
        // charge rebalance
        if((steps % stepsLimit) > 0)
        {
            stepsSize++;
            stepsList.push_back(steps%stepsLimit);
        } 

        for (uint packId = 0; packId < (numberOfWalkerPacks - 1); packId++)
        {
            // set offset in walkers vector
            uint packOffset = packId * walkersPerKernel;
    
            // Host data copy
            // copy original walkers' data to temporary host arrays
            if(this->rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = walkersPerKernel;
                int loop_start, loop_finish;

                #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, collisions, seed, walkers) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        walker_px[id] = this->walkers[id + packOffset].position_x;
                        walker_py[id] = this->walkers[id + packOffset].position_y;
                        walker_pz[id] = this->walkers[id + packOffset].position_z;
                        collisions[id] = 0;
                        seed[id] = this->walkers[id + packOffset].currentSeed;
                    }
                }
            } else
            {
                for (uint id = 0; id < walkersPerKernel; id++)
                {
                    walker_px[id] = this->walkers[id + packOffset].position_x;
                    walker_py[id] = this->walkers[id + packOffset].position_y;
                    walker_pz[id] = this->walkers[id + packOffset].position_z;
                    collisions[id] = 0;
                    seed[id] = this->walkers[id + packOffset].currentSeed;
                }
            }
    
            // Device data copy
            // copy host data to device
            cudaMemcpy(d_walker_px, walker_px, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_walker_py, walker_py, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_walker_pz, walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_collisions, collisions, walkersPerKernel * sizeof(uint), cudaMemcpyHostToDevice);
            cudaMemcpy(d_seed, seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
            //////////////////////////////////////////////////////////////////////
            // Launch kernel for GPU computation
            // kernel "map" launch
            for(uint sIdx = 0; sIdx < stepsList.size(); sIdx++)
            {
                map_3D<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                             d_walker_py,
                                                             d_walker_pz,
                                                             d_collisions,
                                                             d_seed,
                                                             d_bitBlock,
                                                             bitBlockColumns,
                                                             bitBlockRows,
                                                             walkersPerKernel,
                                                             stepsList[sIdx],
                                                             map_columns,
                                                             map_rows,
                                                             map_depth,
                                                             shiftConverter);
                cudaDeviceSynchronize();
            }

            // Host data copy
            // copy device data to host
            cudaMemcpy(collisions, d_collisions, walkersPerKernel * sizeof(uint), cudaMemcpyDeviceToHost);            
            cudaMemcpy(walker_px, d_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_py, d_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_pz, d_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(seed, d_seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
            // copy collisions host data to class members
            if(this->rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = walkersPerKernel;
                int loop_start, loop_finish;

                #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, collisions, seed, walkers) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        this->walkers[id + packOffset].collisions = collisions[id];
                        this->walkers[id + packOffset].position_x = walker_px[id];
                        this->walkers[id + packOffset].position_y = walker_py[id];
                        this->walkers[id + packOffset].position_z = walker_pz[id]; 
                        this->walkers[id + packOffset].currentSeed = seed[id];
                    }
                }
            } else
            {
                
                for (uint id = 0; id < walkersPerKernel; id++)
                {
                    this->walkers[id + packOffset].collisions = collisions[id];
                    this->walkers[id + packOffset].position_x = walker_px[id];
                    this->walkers[id + packOffset].position_y = walker_py[id];
                    this->walkers[id + packOffset].position_z = walker_pz[id]; 
                    this->walkers[id + packOffset].currentSeed = seed[id]; 
                }
            }
    
            
        }
    
        if (lastWalkerPackSize > 0)
        { 
            // last pack is done explicitly
            // set offset in walkers vector
            uint packOffset = (numberOfWalkerPacks - 1) * walkersPerKernel;
    
            // Host data copy
            // copy original walkers' data to temporary host arrays
            if(this->rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = lastWalkerPackSize;
                int loop_start, loop_finish;

                #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, collisions, seed, walkers) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        walker_px[id] = this->walkers[id + packOffset].position_x;
                        walker_py[id] = this->walkers[id + packOffset].position_y;
                        walker_pz[id] = this->walkers[id + packOffset].position_z;
                        collisions[id] = 0;
                        seed[id] = this->walkers[id + packOffset].currentSeed;
                    }
                }
            } else
            {
                for (uint id = 0; id < lastWalkerPackSize; id++)
                {
                    walker_px[id] = this->walkers[id + packOffset].position_x;
                    walker_py[id] = this->walkers[id + packOffset].position_y;
                    walker_pz[id] = this->walkers[id + packOffset].position_z;
                    collisions[id] = 0;
                    seed[id] = this->walkers[id + packOffset].currentSeed;
                }
            }
    
            // Device data copy
            // copy host data to device
            cudaMemcpy(d_walker_px, walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_walker_py, walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_walker_pz, walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_collisions, collisions, lastWalkerPackSize * sizeof(uint), cudaMemcpyHostToDevice);
            cudaMemcpy(d_seed, seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
            //////////////////////////////////////////////////////////////////////
            // Launch kernel for GPU computation
            // kernel "map" launch
            for(uint sIdx = 0; sIdx < stepsList.size(); sIdx++)
            {
                map_3D<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                             d_walker_py,
                                                             d_walker_pz,
                                                             d_collisions,
                                                             d_seed,
                                                             d_bitBlock,
                                                             bitBlockColumns,
                                                             bitBlockRows,
                                                             lastWalkerPackSize,
                                                             stepsList[sIdx],
                                                             map_columns,
                                                             map_rows,
                                                             map_depth,
                                                             shiftConverter);
                cudaDeviceSynchronize();
            }
    
            // Host data copy
            // copy device data to host
            cudaMemcpy(collisions, d_collisions, lastWalkerPackSize * sizeof(uint), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_px, d_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_py, d_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_pz, d_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(seed, d_seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
    
            // copy collisions host data to class members
            if(this->rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = lastWalkerPackSize;
                int loop_start, loop_finish;

                #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, collisions, seed, walkers) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        this->walkers[id + packOffset].collisions = collisions[id];
                        this->walkers[id + packOffset].position_x = walker_px[id];
                        this->walkers[id + packOffset].position_y = walker_py[id];
                        this->walkers[id + packOffset].position_z = walker_pz[id]; 
                        this->walkers[id + packOffset].currentSeed = seed[id];
                    }
                }
            } else
            {
                
                for (uint id = 0; id < lastWalkerPackSize; id++)
                {
                    this->walkers[id + packOffset].collisions = collisions[id];
                    this->walkers[id + packOffset].position_x = walker_px[id];
                    this->walkers[id + packOffset].position_y = walker_py[id];
                    this->walkers[id + packOffset].position_z = walker_pz[id]; 
                    this->walkers[id + packOffset].currentSeed = seed[id]; 
                }
            }
        }

        // create histogram
        (*this).createHistogram(hst_ID, steps);

        // reset collision count, but keep summation in alternative count
        if(this->rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = this->numberOfWalkers;
            int loop_start, loop_finish;

            #pragma omp parallel shared(walkers) private(loop_start, loop_finish) 
            {
                const int thread_id = omp_get_thread_num();
                OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                loop_start = looper.getStart();
                loop_finish = looper.getFinish(); 

                for (uint id = loop_start; id < loop_finish; id++)
                {
                    this->walkers[id].tCollisions += this->walkers[id].collisions;
                    this->walkers[id].resetCollisions();
                }
            }
        } else
        {
            for (uint id = 0; id < this->numberOfWalkers; id++)
            {
                this->walkers[id].tCollisions += this->walkers[id].collisions;
                this->walkers[id].resetCollisions();
            }
        }
    }
    // histogram loop is finished

    // recover walkers collisions from total sum and create a global histogram
    if(this->rwNMR_config.getOpenMPUsage())
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->numberOfWalkers;
        int loop_start, loop_finish;

        #pragma omp parallel shared(walkers) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish(); 

            for (uint id = loop_start; id < loop_finish; id++)
            {
                this->walkers[id].collisions = this->walkers[id].tCollisions;
            }
        }

    } else
    {
        for (uint id = 0; id < this->numberOfWalkers; id++)
        {
            this->walkers[id].collisions = this->walkers[id].tCollisions;   
        }
    }

    // create collision histogram
    (*this).createHistogram();

    // free pointers in host
    free(walker_px);
    free(walker_py);
    free(walker_pz);
    free(collisions);
    free(seed);

    // and direct them to NULL
    walker_px = NULL;
    walker_py = NULL;
    walker_pz = NULL;
    collisions = NULL;
    seed = NULL;

    // also direct the bitBlock pointer created in this context
    // (original data is kept safe)
    bitBlock = NULL;

    // free device global memory
    cudaFree(d_walker_px);
    cudaFree(d_walker_py);
    cudaFree(d_walker_pz);
    cudaFree(d_collisions);
    cudaFree(d_seed);
    cudaFree(d_bitBlock);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Completed.\telapsed time: " << elapsedTime * 1.0e-3 << endl;
}

// function to call GPU kernel to execute
// walker's "walk" method in Graphics Processing Unit
void NMR_Simulation::mapSimulation_CUDA_3D_histograms_periodic()
{
    cout << "initializing mapping simulation 3D in GPU (bc = periodic)... ";
    // reset walkers
    if(this->rwNMR_config.getOpenMPUsage())
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->walkers.size();
        int loop_start, loop_finish;

        #pragma omp parallel shared(walkers) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish(); 

            for (uint id = loop_start; id < loop_finish; id++)
            {
                this->walkers[id].resetPosition();
                this->walkers[id].resetSeed();
                this->walkers[id].resetCollisions();
                this->walkers[id].resetTCollisions();
            }
        }
    } else
    {
        for (uint id = 0; id < this->walkers.size(); id++)
        {
            this->walkers[id].resetPosition();
            this->walkers[id].resetSeed();
            this->walkers[id].resetCollisions();
            this->walkers[id].resetTCollisions();
        }
    }

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values
    uint bitBlockColumns = this->bitBlock.blockColumns;
    uint bitBlockRows = this->bitBlock.blockRows;
    uint numberOfBitBlocks = this->bitBlock.numberOfBlocks;
    uint numberOfWalkers = this->numberOfWalkers;
    uint map_columns = this->bitBlock.imageColumns;
    uint map_rows = this->bitBlock.imageRows;
    uint map_depth = this->bitBlock.imageDepth;
    uint shiftConverter = log2(this->voxelDivision);

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    uint threadsPerBlock = this->rwNMR_config.getThreadsPerBlock();
    uint blocksPerKernel = this->rwNMR_config.getBlocks();
    uint walkersPerKernel = threadsPerBlock * blocksPerKernel;
    if (numberOfWalkers < walkersPerKernel)
    {
        blocksPerKernel = (int)ceil((double)(numberOfWalkers) / (double)(threadsPerBlock));
        walkersPerKernel = threadsPerBlock * blocksPerKernel;
    }
    uint numberOfWalkerPacks = (numberOfWalkers / walkersPerKernel) + 1;
    uint lastWalkerPackSize = numberOfWalkers % walkersPerKernel;

    // bitBlock3D host to device copy
    // assign pointer to bitBlock datastructure
    uint64_t *bitBlock;
    bitBlock = this->bitBlock.blocks;

    // copy host bitblock data to temporary host arrays
    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Device memory allocation
    // pointers used in array conversion and their host memory allocation
    myAllocator arrayFactory;
    int *walker_px = arrayFactory.getIntArray(walkersPerKernel);
    int *walker_py = arrayFactory.getIntArray(walkersPerKernel);
    int *walker_pz = arrayFactory.getIntArray(walkersPerKernel);
    uint *collisions = arrayFactory.getUIntArray(walkersPerKernel);
    uint64_t *seed = arrayFactory.getUInt64Array(walkersPerKernel);

    // Device memory allocation
    // Declaration of device data arrays
    int *d_walker_px;
    int *d_walker_py;
    int *d_walker_pz;
    uint *d_collisions;
    uint64_t *d_seed;

    // alloc memory in device for data arrays
    cudaMalloc((void **)&d_walker_px, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_py, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_pz, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_collisions, walkersPerKernel * sizeof(uint));
    cudaMalloc((void **)&d_seed, walkersPerKernel * sizeof(uint64_t));

    // initialize histograms
    (*this).initHistogramList();

    // loop throughout histogram list
    for(int hst_ID = 0; hst_ID < this->histogramList.size(); hst_ID++)
    {
        // set steps for each histogram
        uint eBegin = this->histogramList[hst_ID].firstEcho;
        uint eEnd = this->histogramList[hst_ID].lastEcho;
        uint steps = this->stepsPerEcho * (eEnd - eBegin);

        // create a steps bucket
        uint stepsLimit = this->rwNMR_config.getMaxRWSteps();
        uint stepsSize = steps/stepsLimit;
        vector<uint> stepsList;
        for(uint idx = 0; idx < stepsSize; idx++)
        {
            stepsList.push_back(stepsLimit);
        }
        // charge rebalance
        if((steps % stepsLimit) > 0)
        {
            stepsSize++;
            stepsList.push_back(steps%stepsLimit);
        } 

        for (uint packId = 0; packId < (numberOfWalkerPacks - 1); packId++)
        {
            // set offset in walkers vector
            uint packOffset = packId * walkersPerKernel;
    
            // Host data copy
            // copy original walkers' data to temporary host arrays
            if(this->rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = walkersPerKernel;
                int loop_start, loop_finish;

                #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, collisions, seed, walkers) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        walker_px[id] = this->walkers[id + packOffset].position_x;
                        walker_py[id] = this->walkers[id + packOffset].position_y;
                        walker_pz[id] = this->walkers[id + packOffset].position_z;
                        collisions[id] = 0;
                        seed[id] = this->walkers[id + packOffset].currentSeed;
                    }
                }
            } else
            {
                for (uint id = 0; id < walkersPerKernel; id++)
                {
                    walker_px[id] = this->walkers[id + packOffset].position_x;
                    walker_py[id] = this->walkers[id + packOffset].position_y;
                    walker_pz[id] = this->walkers[id + packOffset].position_z;
                    collisions[id] = 0;
                    seed[id] = this->walkers[id + packOffset].currentSeed;
                }
            }
    
            // Device data copy
            // copy host data to device
            cudaMemcpy(d_walker_px, walker_px, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_walker_py, walker_py, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_walker_pz, walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_collisions, collisions, walkersPerKernel * sizeof(uint), cudaMemcpyHostToDevice);
            cudaMemcpy(d_seed, seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
            //////////////////////////////////////////////////////////////////////
            // Launch kernel for GPU computation
            // kernel "map" launch
            for(uint sIdx = 0; sIdx < stepsList.size(); sIdx++)
            {
                map_3D_periodic<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                                      d_walker_py,
                                                                      d_walker_pz,
                                                                      d_collisions,
                                                                      d_seed,
                                                                      d_bitBlock,
                                                                      bitBlockColumns,
                                                                      bitBlockRows,
                                                                      walkersPerKernel,
                                                                      stepsList[sIdx],
                                                                      map_columns,
                                                                      map_rows,
                                                                      map_depth,
                                                                      shiftConverter);
                cudaDeviceSynchronize();
            }

            // Host data copy
            // copy device data to host
            cudaMemcpy(collisions, d_collisions, walkersPerKernel * sizeof(uint), cudaMemcpyDeviceToHost);            
            cudaMemcpy(walker_px, d_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_py, d_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_pz, d_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(seed, d_seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
            // copy collisions host data to class members
            if(this->rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = walkersPerKernel;
                int loop_start, loop_finish;

                #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, collisions, seed, walkers) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        this->walkers[id + packOffset].collisions = collisions[id];
                        this->walkers[id + packOffset].position_x = walker_px[id];
                        this->walkers[id + packOffset].position_y = walker_py[id];
                        this->walkers[id + packOffset].position_z = walker_pz[id]; 
                        this->walkers[id + packOffset].currentSeed = seed[id];
                    }
                }
            } else
            {
                
                for (uint id = 0; id < walkersPerKernel; id++)
                {
                    this->walkers[id + packOffset].collisions = collisions[id];
                    this->walkers[id + packOffset].position_x = walker_px[id];
                    this->walkers[id + packOffset].position_y = walker_py[id];
                    this->walkers[id + packOffset].position_z = walker_pz[id]; 
                    this->walkers[id + packOffset].currentSeed = seed[id]; 
                }
            }
    
            
        }
    
        if (lastWalkerPackSize > 0)
        { 
            // last pack is done explicitly
            // set offset in walkers vector
            uint packOffset = (numberOfWalkerPacks - 1) * walkersPerKernel;
    
            // Host data copy
            // copy original walkers' data to temporary host arrays
            if(this->rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = lastWalkerPackSize;
                int loop_start, loop_finish;

                #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, collisions, seed, walkers) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        walker_px[id] = this->walkers[id + packOffset].position_x;
                        walker_py[id] = this->walkers[id + packOffset].position_y;
                        walker_pz[id] = this->walkers[id + packOffset].position_z;
                        collisions[id] = 0;
                        seed[id] = this->walkers[id + packOffset].currentSeed;
                    }
                }
            } else
            {
                for (uint id = 0; id < lastWalkerPackSize; id++)
                {
                    walker_px[id] = this->walkers[id + packOffset].position_x;
                    walker_py[id] = this->walkers[id + packOffset].position_y;
                    walker_pz[id] = this->walkers[id + packOffset].position_z;
                    collisions[id] = 0;
                    seed[id] = this->walkers[id + packOffset].currentSeed;
                }
            }
    
            // Device data copy
            // copy host data to device
            cudaMemcpy(d_walker_px, walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_walker_py, walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_walker_pz, walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_collisions, collisions, lastWalkerPackSize * sizeof(uint), cudaMemcpyHostToDevice);
            cudaMemcpy(d_seed, seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
            //////////////////////////////////////////////////////////////////////
            // Launch kernel for GPU computation
            // kernel "map" launch
            for(uint sIdx = 0; sIdx < stepsList.size(); sIdx++)
            {
                map_3D_periodic<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                                      d_walker_py,
                                                                      d_walker_pz,
                                                                      d_collisions,
                                                                      d_seed,
                                                                      d_bitBlock,
                                                                      bitBlockColumns,
                                                                      bitBlockRows,
                                                                      lastWalkerPackSize,
                                                                      stepsList[sIdx],
                                                                      map_columns,
                                                                      map_rows,
                                                                      map_depth,
                                                                      shiftConverter);
                cudaDeviceSynchronize();
            }
    
            // Host data copy
            // copy device data to host
            cudaMemcpy(collisions, d_collisions, lastWalkerPackSize * sizeof(uint), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_px, d_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_py, d_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_pz, d_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(seed, d_seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
    
            // copy collisions host data to class members
            if(this->rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = lastWalkerPackSize;
                int loop_start, loop_finish;

                #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, collisions, seed, walkers) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        this->walkers[id + packOffset].collisions = collisions[id];
                        this->walkers[id + packOffset].position_x = walker_px[id];
                        this->walkers[id + packOffset].position_y = walker_py[id];
                        this->walkers[id + packOffset].position_z = walker_pz[id]; 
                        this->walkers[id + packOffset].currentSeed = seed[id];
                    }
                }
            } else
            {
                
                for (uint id = 0; id < lastWalkerPackSize; id++)
                {
                    this->walkers[id + packOffset].collisions = collisions[id];
                    this->walkers[id + packOffset].position_x = walker_px[id];
                    this->walkers[id + packOffset].position_y = walker_py[id];
                    this->walkers[id + packOffset].position_z = walker_pz[id]; 
                    this->walkers[id + packOffset].currentSeed = seed[id]; 
                }
            }
        }

        // create histogram
        (*this).createHistogram(hst_ID, steps);

        // reset collision count, but keep summation in alternative count
        if(this->rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = this->numberOfWalkers;
            int loop_start, loop_finish;

            #pragma omp parallel shared(walkers) private(loop_start, loop_finish) 
            {
                const int thread_id = omp_get_thread_num();
                OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                loop_start = looper.getStart();
                loop_finish = looper.getFinish(); 

                for (uint id = loop_start; id < loop_finish; id++)
                {
                    this->walkers[id].tCollisions += this->walkers[id].collisions;
                    this->walkers[id].resetCollisions();
                }
            }
        } else
        {
            for (uint id = 0; id < this->numberOfWalkers; id++)
            {
                this->walkers[id].tCollisions += this->walkers[id].collisions;
                this->walkers[id].resetCollisions();
            }
        }
    }
    // histogram loop is finished

    // recover walkers collisions from total sum and create a global histogram
    if(this->rwNMR_config.getOpenMPUsage())
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->numberOfWalkers;
        int loop_start, loop_finish;

        #pragma omp parallel shared(walkers) private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish(); 

            for (uint id = loop_start; id < loop_finish; id++)
            {
                this->walkers[id].collisions = this->walkers[id].tCollisions;
            }
        }

    } else
    {
        for (uint id = 0; id < this->numberOfWalkers; id++)
        {
            this->walkers[id].collisions = this->walkers[id].tCollisions;   
        }
    }

    // create collision histogram
    (*this).createHistogram();

    // free pointers in host
    free(walker_px);
    free(walker_py);
    free(walker_pz);
    free(collisions);
    free(seed);

    // and direct them to NULL
    walker_px = NULL;
    walker_py = NULL;
    walker_pz = NULL;
    collisions = NULL;
    seed = NULL;

    // also direct the bitBlock pointer created in this context
    // (original data is kept safe)
    bitBlock = NULL;

    // free device global memory
    cudaFree(d_walker_px);
    cudaFree(d_walker_py);
    cudaFree(d_walker_pz);
    cudaFree(d_collisions);
    cudaFree(d_seed);
    cudaFree(d_bitBlock);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Completed.\telapsed time: " << elapsedTime * 1.0e-3 << endl;
}


/////////////////////////////////////////////////////////////////////
//////////////////////// DEVICE FUNCTIONS ///////////////////////////
/////////////////////////////////////////////////////////////////////

__device__ direction computeNextDirection_3D(uint64_t &seed)
{
    // generate random number using xorshift algorithm
    xorshift64_state xor_state;
    xor_state.a = seed;
    seed = xorShift64_3D(&xor_state);
    uint64_t rand = seed;

    // set direction based on the random number
    direction nextDirection = (direction)(mod6_3D(rand) + 1);
    return nextDirection;
}

__device__ uint64_t xorShift64_3D(struct xorshift64_state *state)
{
    uint64_t x = state->a;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return state->a = x;
}

__device__ uint64_t mod6_3D(uint64_t a)
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

__device__ direction checkBorder_3D(int walker_px,
                                    int walker_py,
                                    int walker_pz,
                                    direction &nextDirection,
                                    const int map_columns,
                                    const int map_rows,
                                    const int map_depth)
{
    switch (nextDirection)
    {
    case North:
        if (walker_py == 0)
        {
            nextDirection = South;
        }
        break;

    case South:

        if (walker_py == map_rows - 1)
        {
            nextDirection = North;
        }
        break;

    case West:
        if (walker_px == 0)
        {
            nextDirection = East;
        }
        break;

    case East:
        if (walker_px == map_columns - 1)
        {
            nextDirection = West;
        }
        break;

    case Up:
        if (walker_pz == map_depth - 1)
        {
            nextDirection = Down;
        }
        break;

    case Down:
        if (walker_pz == 0)
        {
            nextDirection = Up;
        }
        break;
    }

    return nextDirection;
}

__device__ void computeNextPosition_3D(int &walker_px,
                                       int &walker_py,
                                       int &walker_pz,
                                       direction nextDirection,
                                       int &next_x,
                                       int &next_y,
                                       int &next_z)
{
    next_x = walker_px;
    next_y = walker_py;
    next_z = walker_pz;

    switch (nextDirection)
    {
    case North:
        next_y = next_y - 1;
        break;

    case West:
        next_x = next_x - 1;
        break;

    case South:
        next_y = next_y + 1;
        break;

    case East:
        next_x = next_x + 1;
        break;

    case Up:
        next_z = next_z + 1;
        break;

    case Down:
        next_z = next_z - 1;
        break;
    }
}

__device__ bool checkNextPosition_3D(int next_x,
                                     int next_y,
                                     int next_z,
                                     const uint64_t *bitBlock,
                                     const int bitBlockColumns,
                                     const int bitBlockRows)
{
    int blockIndex = findBlockIndex_3D(next_x, next_y, next_z, bitBlockColumns, bitBlockRows);
    int nextBit = findBitIndex_3D(next_x, next_y, next_z);
    uint64_t nextBlock = bitBlock[blockIndex];

    return (!checkIfBlockBitIsWall_3D(nextBlock, nextBit));
};

__device__ int findBlockIndex_3D(int next_x, int next_y, int next_z, int bitBlockColumns, int bitBlockRows)
{
    // "x >> 2" is like "x / 4" in bitwise operation
    int block_x = next_x >> 2;
    int block_y = next_y >> 2;
    int block_z = next_z >> 2;
    int blockIndex = block_x + block_y * bitBlockColumns + block_z * (bitBlockColumns * bitBlockRows);

    return blockIndex;
}

__device__ int findBitIndex_3D(int next_x, int next_y, int next_z)
{
    // "x & (n - 1)" is lise "x % n" in bitwise operation
    int bit_x = next_x & (COLUMNSPERBLOCK3D - 1);
    int bit_y = next_y & (ROWSPERBLOCK3D - 1);
    int bit_z = next_z & (DEPTHPERBLOCK3D - 1);
    // "x << 3" is like "x * 8" in bitwise operation
    int bitIndex = bit_x + (bit_y << 2) + ((bit_z << 2) << 2);

    return bitIndex;
}

__device__ bool checkIfBlockBitIsWall_3D(uint64_t nextBlock, int nextBit)
{
    return ((nextBlock >> nextBit) & 1ull);
}

__device__ double compute_pfgse_k_value(double gradientMagnitude, double tiny_delta, double giromagneticRatio)
{
    return (tiny_delta * 1.0e-03) * (TWO_PI * giromagneticRatio * 1.0e+06) * (gradientMagnitude * 1.0e-08);
}

__device__ int convertLocalToGlobal_3D(int _localPos, uint _shiftConverter)
{
    return (_localPos >> _shiftConverter);
}