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

//include
#include "../Walker/walker.h"
#include "../RNG/xorshift.h"
#include "NMR_Simulation.h"
#include "NMR_cudaUtils_2D.h"

// GPU kernel for Mapping simulation - a.k.a. walker's collision count
// in this kernel, each thread will behave as a solitary walker
__global__ void map(int *walker_px,
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
            nextDirection = computeNextDirection(local_seed);

            nextDirection = checkBorder(position_x,
                                        position_y,
                                        nextDirection,
                                        map_columns,
                                        map_rows);

            computeNextPosition(position_x,
                                position_y,
                                nextDirection,
                                next_x,
                                next_y);

            if (checkNextPosition(next_x, next_y, bitBlock, bitBlockColumns))
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

// GPU kernel for NMR simulation - a.k.a. walker's relaxation/demagnetization
// in this kernel, each thread will behave as a solitary walker
__global__ void walk(int *walker_px,
                     int *walker_py,
                     double *decreaseFactor,
                     double *energy,
                     uint64_t *seed,
                     const uint64_t *bitBlock,
                     const int bitBlockColumns,
                     const uint numberOfWalkers,
                     const uint energyArraySize,
                     const int echoesPerKernel,
                     const int stepsPerEcho,
                     const int map_columns,
                     const int map_rows,
                     const uint shift_convert)
{
    // identify thread's walker
    int walkerId = threadIdx.x + blockIdx.x * blockDim.x;

    // Local variables for unique read from device global memory
    int position_x, position_y;
    double local_dFactor;
    uint64_t local_seed;

    // 1st energy array offset
    // in first echo, walker's energy is stored in last echo of previous kernel launch
    uint energy_OFFSET = (echoesPerKernel - 1) * energyArraySize;
    double energyLvl;

    // thread variables for future movements
    int next_x, next_y;
    direction nextDirection = None;

    // now begin the "walk" procedure de facto
    if (walkerId < numberOfWalkers)
    {
        // Local variables for unique read from device global memory
        position_x = walker_px[walkerId];
        position_y = walker_py[walkerId];
        local_dFactor = decreaseFactor[walkerId];
        local_seed = seed[walkerId];
        energyLvl = energy[walkerId + energy_OFFSET];

        for (int echo = 0; echo < echoesPerKernel; echo++)
        {
            // update the offset
            energy_OFFSET = echo * energyArraySize;

            for (int step = 0; step < stepsPerEcho; step++)
            {
                nextDirection = computeNextDirection(local_seed);

                nextDirection = checkBorder(position_x,
                                            position_y,
                                            nextDirection,
                                            map_columns,
                                            map_rows);

                computeNextPosition(position_x,
                                    position_y,
                                    nextDirection,
                                    next_x,
                                    next_y);

                if (checkNextPosition(next_x, next_y, bitBlock, bitBlockColumns))
                {
                    // position is valid
                    position_x = next_x;
                    position_y = next_y;
                }
                else
                {
                    // walker chocks with wall and comes back to the same position
                    // walker loses energy due to this collision
                    energyLvl = energyLvl * local_dFactor;
                }
            }

            // walker's energy device global memory update
            // must be done for each echo
            energy[walkerId + energy_OFFSET] = energyLvl;
        }

        // position and seed device global memory update
        // must be done for each kernel
        walker_px[walkerId] = position_x;
        walker_py[walkerId] = position_y;
        seed[walkerId] = local_seed;
    }
}

// GPU kernel for reducing energy array into a global energy vector
__global__ void energyReduce_shared(double *energy,
                                    double *collector,
                                    const uint energyArraySize,
                                    const uint collectorSize,
                                    const int echoesPerKernel)
{
    extern __shared__ double sharedData[];

    // each thread loads two elements from global to shared mem
    uint threadId = threadIdx.x;
    uint globalId = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    uint OFFSET;

    for (int echo = 0; echo < echoesPerKernel; echo++)
    {
        OFFSET = echo * energyArraySize;

        sharedData[threadId] = energy[globalId + OFFSET] + energy[globalId + blockDim.x + OFFSET];
        __syncthreads();

        // do reduction in shared mem
        for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadId < stride)
            {
                sharedData[threadId] += sharedData[threadId + stride];
            }
            __syncthreads();
        }

        // write result for this block to global mem
        if (threadId == 0)
        {
            collector[blockIdx.x + (echo * collectorSize)] = sharedData[0];
        }
        __syncthreads();
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
    int *walker_px = setIntArray(numberOfWalkers);
    int *walker_py = setIntArray(numberOfWalkers);
    uint *collisions = setUIntArray(numberOfWalkers);
    uint64_t *seed = setUInt64Array(numberOfWalkers);

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
    map<<<threadsPerBlock, blocksPerKernel>>>(d_walker_px,
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
    int *walker_px = setIntArray(numberOfWalkers);
    int *walker_py = setIntArray(numberOfWalkers);
    uint *collisions = setUIntArray(numberOfWalkers);
    uint64_t *seed = setUInt64Array(numberOfWalkers);

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
        map<<<threadsPerBlock, blocksPerKernel>>>(d_walker_px,
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

// function to call GPU kernel to execute
// walker's "walk" method in Graphics Processing Unit
void NMR_Simulation::walkSimulation_CUDA_2D()
{
    cout << "initializing RW-NMR simulation in GPU... ";
    // reset walker's initial state 
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        // this->walkers[id].resetPosition();
        this->walkers[id].resetSeed();
        this->walkers[id].resetEnergy();
    }

    // reset vector to store energy decay
    (*this).resetGlobalEnergy();
    this->globalEnergy.reserve(this->numberOfEchoes);

    // get initial energy state
    double energySum = 0.0;
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        energySum += this->walkers[id].energy;
    }
    this->globalEnergy.push_back(energySum);

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values for sizing issues
    int bitBlockColumns = this->bitBlock.blockColumns;
    int numberOfBitBlocks = this->bitBlock.numberOfBlocks;
    uint numberOfWalkers = this->numberOfWalkers;
    uint numberOfEchoes = this->numberOfEchoes;
    int stepsPerEcho = this->stepsPerEcho;
    int map_columns = this->bitBlock.imageColumns;
    int map_rows = this->bitBlock.imageRows;
    uint shiftConverter = log2(this->voxelDivision);

    // number of echos that each walker in kernel call will perform
    int echoesPerKernel = this->rwNMR_config.getEchoesPerKernel();
    int kernelCalls = (int)ceil((double)numberOfEchoes / (double)echoesPerKernel);

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    int threadsPerBlock = this->rwNMR_config.getThreadsPerBlock();
    int blocksPerKernel = (int)ceil(double(numberOfWalkers) / double(threadsPerBlock));

    if (blocksPerKernel % 2 == 1)
    {
        // blocks per kernel should be multiple of 2
        blocksPerKernel += 1;
    }

    // NOTE:
    // I should also treat the case when the value "blocksPerKernel" or "threadsPerBlock"
    // is greater than the max value CUDA allows the Grid and Block dimensions
    // but for the 2D simulations, this value should not be surpassed

    int blockTail = (blocksPerKernel * threadsPerBlock) - numberOfWalkers;
    uint energyArraySize = numberOfWalkers + blockTail;
    uint energyCollectorSize = (blocksPerKernel / 2);

    // Copy bitBlock2D data from host to device (only once)
    // assign pointer to bitBlock datastructure
    uint64_t *bitBlock;
    bitBlock = this->bitBlock.blocks;

    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // pointers using in array conversion
    int *walker_px = setIntArray(numberOfWalkers);
    int *walker_py = setIntArray(numberOfWalkers);
    double *decreaseFactor = setDoubleArray(numberOfWalkers);
    double *energy = setDoubleArray(echoesPerKernel * energyArraySize);
    double *energyCollector = setDoubleArray(echoesPerKernel * energyCollectorSize);
    uint64_t *seed = setUInt64Array(numberOfWalkers);

    // temporary array to collect energy contributions
    double *temp_globalEnergy = setDoubleArray((uint)echoesPerKernel);

    // Host data copy
    // copy original walkers' data to temporary host arrays
    // complete data of energy arrays with zero entries
    for (uint i = 0; i < (echoesPerKernel * energyArraySize); i++)
    {
        energy[i] = 0.0;
    }
    for (uint i = 0; i < (echoesPerKernel * energyCollectorSize); i++)
    {
        energyCollector[i] = 0.0;
    }

    for (uint i = 0; i < numberOfWalkers; i++)
    {
        walker_px[i] = this->walkers[i].initialPosition.x;
        walker_py[i] = this->walkers[i].initialPosition.y;
        decreaseFactor[i] = this->walkers[i].decreaseFactor;
        energy[i + ((echoesPerKernel - 1) * energyArraySize)] = this->walkers[i].energy;
        seed[i] = this->walkers[i].initialSeed;
    }

    // Device memory allocation
    // Declaration of device data arrays
    int *d_walker_px;
    int *d_walker_py;
    double *d_decreaseFactor;
    double *d_energy;
    double *d_energyCollector;
    uint64_t *d_seed;

    // Memory allocation in device for data arrays
    cudaMalloc((void **)&d_walker_px, numberOfWalkers * sizeof(int));
    cudaMalloc((void **)&d_walker_py, numberOfWalkers * sizeof(int));
    cudaMalloc((void **)&d_decreaseFactor, numberOfWalkers * sizeof(double));
    cudaMalloc((void **)&d_energy, echoesPerKernel * energyArraySize * sizeof(double));
    cudaMalloc((void **)&d_energyCollector, echoesPerKernel * energyCollectorSize * sizeof(double));
    cudaMalloc((void **)&d_seed, numberOfWalkers * sizeof(uint64_t));

    // Device data copy
    // copy host data to device
    cudaMemcpy(d_walker_px, walker_px, numberOfWalkers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_walker_py, walker_py, numberOfWalkers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_decreaseFactor, decreaseFactor, numberOfWalkers * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energy, energy, echoesPerKernel * energyArraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed, seed, numberOfWalkers * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Launch kernel for GPU computation
    for (int kernelId = 0; kernelId < kernelCalls; kernelId++)
    {
        // call "walk" method kernel
        walk<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                   d_walker_py,
                                                   d_decreaseFactor,
                                                   d_energy,
                                                   d_seed,
                                                   d_bitBlock,
                                                   bitBlockColumns,
                                                   numberOfWalkers,
                                                   energyArraySize,
                                                   echoesPerKernel,
                                                   stepsPerEcho,
                                                   map_columns,
                                                   map_rows,
                                                   shiftConverter);
        cudaDeviceSynchronize();

        // launch globalEnergy "reduce" kernel
        energyReduce_shared<<<blocksPerKernel / 2,
                              threadsPerBlock,
                              threadsPerBlock * sizeof(double)>>>(d_energy,
                                                                  d_energyCollector,
                                                                  energyArraySize,
                                                                  energyCollectorSize,
                                                                  echoesPerKernel);
        cudaDeviceSynchronize();

        // copy data from gatherer array
        cudaMemcpy(energyCollector,
                   d_energyCollector,
                   echoesPerKernel * energyCollectorSize * sizeof(double),
                   cudaMemcpyDeviceToHost);

        //last reduce is done in CPU parallel-style using openMP
        reduce_omp(temp_globalEnergy, energyCollector, echoesPerKernel, blocksPerKernel / 2);

        // copy data from temporary array to NMR_Simulation2D "globalEnergy" vector class member
        for (int echo = 0; echo < echoesPerKernel; echo++)
        {
            this->globalEnergy.push_back(temp_globalEnergy[echo]);
        }
    }

    // free pointers in host
    free(walker_px);
    free(walker_py);
    free(decreaseFactor);
    free(energy);
    free(energyCollector);
    free(temp_globalEnergy);
    free(seed);

    // and direct them to NULL
    walker_px = NULL;
    walker_py = NULL;
    decreaseFactor = NULL;
    energy = NULL;
    energyCollector = NULL;
    temp_globalEnergy = NULL;
    seed = NULL;

    // also direct the bitBlock pointer created in this context
    // (original data is kept safe)
    bitBlock = NULL;

    // free device global memory
    cudaFree(d_walker_px);
    cudaFree(d_walker_py);
    cudaFree(d_decreaseFactor);
    cudaFree(d_energy);
    cudaFree(d_energyCollector);
    cudaFree(d_seed);
    cudaFree(d_bitBlock);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Completed.\telapsed time: " << elapsedTime * 1.0e-3 << endl;

    cudaDeviceReset();
}

/////////////////////////////////////////////////////////////////////
//////////////////////// HOST FUNCTIONS ///////////////////////////
/////////////////////////////////////////////////////////////////////
void reduce_omp(double *temp_collector, double *array, int numberOfEchoes, uint arraySizePerEcho)
{
    // declaring shared variables
    uint offset;
    double arraySum;
    uint id;

    for (int echo = 0; echo < numberOfEchoes; echo++)
    {
        arraySum = 0.0;
        offset = (echo * arraySizePerEcho);

// #pragma omp parallel for reduction(+ \
//                                    : arraySum) private(id) shared(array, offset)
        for (id = 0; id < arraySizePerEcho; id++)
        {
            arraySum += array[id + offset];
        }

        temp_collector[echo] = arraySum;
    }
}

void test_omp(uint size)
{
    uint thread[4];
    thread[0] = 0;
    thread[1] = 0;
    thread[2] = 0;
    thread[3] = 0;

// #pragma omp parallel for
    for (uint i = 0; i < size; i++)
    {
        thread[i % 4]++;
    }

    cout << "final do teste" << endl;
    cout << "thread 00: " << thread[0] << endl;
    cout << "thread 01: " << thread[1] << endl;
    cout << "thread 02: " << thread[2] << endl;
    cout << "thread 03: " << thread[3] << endl;
}

/////////////////////////////////////////////////////////////////////
////////////////////////// HOST FUNCTIONS ///////////////////////////
/////////////////////////////////////////////////////////////////////
int *setIntArray(uint size)
{
    int *array;

    array = (int *)malloc(size * sizeof(int));
    if (array == NULL)
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    return array;
}

uint *setUIntArray(uint size)
{
    uint *array;

    array = (uint *)malloc(size * sizeof(uint));
    if (array == NULL)
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    return array;
}

double *setDoubleArray(uint size)
{

    double *array;

    array = (double *)malloc(size * sizeof(double));
    if (array == NULL)
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    return array;
}

uint64_t *setUInt64Array(uint size)
{
    uint64_t *array;

    array = (uint64_t *)malloc(size * sizeof(uint64_t));
    if (array == NULL)
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    return array;
}

void copyVectorBtoA(int a[], int b[], uint size)
{
    for (uint i = 0; i < size; i++)
    {
        a[i] = b[i];
    }
}

void copyVectorBtoA(double a[], double b[], int size)
{
    for (uint i = 0; i < size; i++)
    {
        a[i] = b[i];
    }
}

void copyVectorBtoA(uint64_t a[], uint64_t b[], int size)
{
    for (uint i = 0; i < size; i++)
    {
        a[i] = b[i];
    }
}

void vectorElementSwap(int *vector, uint index1, uint index2)
{
    int temp;

    temp = vector[index1];
    vector[index1] = vector[index2];
    vector[index2] = temp;
}

void vectorElementSwap(double *vector, uint index1, uint index2)
{
    double temp;

    temp = vector[index1];
    vector[index1] = vector[index2];
    vector[index2] = temp;
}

void vectorElementSwap(uint64_t *vector, uint index1, uint index2)
{
    uint64_t temp;

    temp = vector[index1];
    vector[index1] = vector[index2];
    vector[index2] = temp;
}

/////////////////////////////////////////////////////////////////////
//////////////////////// DEVICE FUNCTIONS ///////////////////////////
/////////////////////////////////////////////////////////////////////

__device__ direction computeNextDirection(uint64_t &seed)
{
    // generate random number using xorshift algorithm
    xorshift64_state xor_state;
    xor_state.a = seed;
    seed = xorShift64(&xor_state);

    // set direction based on the random number
    direction nextDirection = (direction)((seed % 4) + 1);
    return nextDirection;
}

__device__ uint64_t xorShift64(struct xorshift64_state *state)
{
    uint64_t x = state->a;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return state->a = x;
}

__device__ direction checkBorder(int walker_px,
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

__device__ void computeNextPosition(int &walker_px,
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

__device__ bool checkNextPosition(int next_x,
                                  int next_y,
                                  const uint64_t *bitBlock,
                                  const int bitBlockColumns)
{
    int blockIndex = findBlockIndex(next_x, next_y, bitBlockColumns);
    int nextBit = findBitIndex(next_x, next_y);
    uint64_t nextBlock = bitBlock[blockIndex];

    return (!checkIfBlockBitIsWall(nextBlock, nextBit));
};

__device__ int findBlockIndex(int next_x, int next_y, int bitBlockColumns)
{
    // "x >> 3" is like "x / 8" in bitwise operation
    int block_x = next_x >> 3;
    int block_y = next_y >> 3;
    int blockIndex = block_x + block_y * bitBlockColumns;

    return blockIndex;
}

__device__ int findBitIndex(int next_x, int next_y)
{
    // "x & (n - 1)" is lise "x % n" in bitwise operation
    int bit_x = next_x & (COLUMNSPERBLOCK2D - 1);
    int bit_y = next_y & (ROWSPERBLOCK2D - 1);
    // "x << 3" is like "x * 8" in bitwise operation
    int bitIndex = bit_x + (bit_y << 3);

    return bitIndex;
}

__device__ bool checkIfBlockBitIsWall(uint64_t nextBlock, int nextBit)
{
    return ((nextBlock >> nextBit) & 1ull);
}

__device__ uint convertLocalToGlobal_2D(uint _localPos, uint _shiftConverter)
{
    return (_localPos >> _shiftConverter);
}