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

//include
#include "../Walker/walker.h"
#include "../RNG/xorshift.h"
#include "NMR_Simulation.h"
#include "NMR_cudaUtils_3D.h"

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

// GPU kernel for NMR simulation - a.k.a. walker's relaxation/demagnetization
// in this kernel, each thread will behave as a solitary walker
__global__ void walk_3D(int *walker_px,
                        int *walker_py,
                        int *walker_pz,
                        double *penalty,
                        double *energy,
                        uint64_t *seed,
                        const uint64_t *bitBlock,
                        const uint bitBlockColumns,
                        const uint bitBlockRows,
                        const uint numberOfWalkers,
                        const uint energyArraySize,
                        const uint echoesPerKernel,
                        const uint stepsPerEcho,
                        const uint map_columns,
                        const uint map_rows,
                        const uint map_depth,
                        const uint shift_convert)
{
    // identify thread's walker
    int walkerId = threadIdx.x + blockIdx.x * blockDim.x;

    // Local variables for unique read from device global memory
    int position_x, position_y, position_z;
    double local_dFactor;
    uint64_t local_seed;

    // 1st energy array offset
    // in first echo, walker's energy is stored in last echo of previous kernel launch
    uint energy_OFFSET = (echoesPerKernel - 1) * energyArraySize;
    double energyLvl;

    // thread variables for future movements
    int next_x, next_y, next_z;
    direction nextDirection = None;

    // now begin the "walk" procedure de facto
    if (walkerId < numberOfWalkers)
    {
        position_x = walker_px[walkerId];
        position_y = walker_py[walkerId];
        position_z = walker_pz[walkerId];
        local_dFactor = penalty[walkerId];
        local_seed = seed[walkerId];
        energyLvl = energy[walkerId + energy_OFFSET];

        for (int echo = 0; echo < echoesPerKernel; echo++)
        {
            // update the offset
            energy_OFFSET = echo * energyArraySize;

            for (int step = 0; step < stepsPerEcho; step++)
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
                    // position is valid
                    position_x = next_x;
                    position_y = next_y;
                    position_z = next_z;
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
        walker_pz[walkerId] = position_z;
        seed[walkerId] = local_seed;
    }
}

// GPU kernel for reducing energy array into a global energy vector
__global__ void energyReduce_shared_3D(double *energy,
                                       double *collector,
                                       const uint energyArraySize,
                                       const uint collectorSize,
                                       const uint echoesPerKernel)
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

// GPU kernel for NMR simulation - a.k.a. walker's relaxation/demagnetization
// in this kernel, each thread will behave as a solitary walker
__global__ void walk_PFG(int *walker_px,
                         int *walker_py,
                         int *walker_pz,
                         double *penalty,
                         double *energy,
                         uint64_t *seed,
                         const uint64_t *bitBlock,
                         const uint bitBlockColumns,
                         const uint bitBlockRows,
                         const uint numberOfWalkers,
                         const uint energyArraySize,
                         const uint numberOfSteps,
                         const uint map_columns,
                         const uint map_rows,
                         const uint map_depth,
                         const uint shift_convert)
{
    // identify thread's walker
    int walkerId = threadIdx.x + blockIdx.x * blockDim.x;

    // Local variables for unique read from device global memory
    int position_x, position_y, position_z;
    double local_penalty;
    uint64_t local_seed;
    double energyLvl;

    // thread variables for future movements
    int next_x, next_y, next_z;
    direction nextDirection = None;

    // now begin the "walk" procedure de facto
    if (walkerId < numberOfWalkers)
    {
        position_x = walker_px[walkerId];
        position_y = walker_py[walkerId];
        position_z = walker_pz[walkerId];
        local_penalty = penalty[walkerId];
        local_seed = seed[walkerId];
        energyLvl = energy[walkerId];

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
                // position is valid
                position_x = next_x;
                position_y = next_y;
                position_z = next_z;
            }
            else
            {
                // walker chocks with wall and comes back to the same position
                // walker loses energy due to this collision
                energyLvl = energyLvl * local_penalty;
            }
        }

        // walker's energy device global memory update
        // must be done for each echo
        energy[walkerId] = energyLvl;

        // position and seed device global memory update
        // must be done for each kernel
        walker_px[walkerId] = position_x;
        walker_py[walkerId] = position_y;
        walker_pz[walkerId] = position_z;
        seed[walkerId] = local_seed;
    }
}

// GPU kernel for NMR simulation - a.k.a. walker's relaxation/demagnetization
// in this kernel, each thread will behave as a solitary walker
__global__ void measure_PFG(int *walker_x0,
                            int *walker_y0, 
                            int *walker_z0,
                            int *walker_xF,
                            int *walker_yF,
                            int *walker_zF,
                            double *energy,
                            double *phase,
                            const uint numberOfWalkers,
                            const double voxelResolution,
                            const double gradient,
                            const double tiny_delta,
                            const double giromagneticRatio)
{
    // identify thread's walker
    int walkerId = threadIdx.x + blockIdx.x * blockDim.x;

    if(walkerId < numberOfWalkers)
    {
        // compute local magnetizion magnitude
        double K = compute_pfgse_k_value(gradient, tiny_delta, giromagneticRatio);
        double local_phase = K * (walker_zF[walkerId] - walker_z0[walkerId]) * voxelResolution; 
        double magnitude_real_value = cos(local_phase);
        double magnitude_imag_value = sin(local_phase);
    
        // update global value 
        phase[walkerId] = magnitude_real_value * energy[walkerId];
    }
}

__global__ void reduce_PFG(double *data,
                           double *deposit,
                           const uint data_size,
                           const uint deposit_size)
{
    extern __shared__ double sdata[];

    // each thread loads one element from global to shared mem
    unsigned int threadId = threadIdx.x;
    unsigned int globalId = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    sdata[threadId] = data[globalId] + data[globalId + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadId < stride)
        {
            sdata[threadId] += sdata[threadId + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (threadId == 0)
    {
        deposit[blockIdx.x] = sdata[0];
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
    int *walker_px = setIntArray_3D(walkersPerKernel);
    int *walker_py = setIntArray_3D(walkersPerKernel);
    int *walker_pz = setIntArray_3D(walkersPerKernel);
    uint *collisions = setUIntArray_3D(walkersPerKernel);
    uint64_t *seed = setUInt64Array_3D(walkersPerKernel);

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
    int *walker_px = setIntArray_3D(walkersPerKernel);
    int *walker_py = setIntArray_3D(walkersPerKernel);
    int *walker_pz = setIntArray_3D(walkersPerKernel);
    uint *collisions = setUIntArray_3D(walkersPerKernel);
    uint64_t *seed = setUInt64Array_3D(walkersPerKernel);

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
void NMR_Simulation::walkSimulation_CUDA_3D()
{
    cout << "initializing RW-NMR simulation in GPU... ";

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
                this->walkers[id].resetEnergy();
            }
        }
    } else
    {
        // reset walker's initial state 
        for (uint id = 0; id < this->walkers.size(); id++)
        {
            this->walkers[id].resetPosition();
            this->walkers[id].resetSeed();
            this->walkers[id].resetEnergy();
        }
    }

    // reset vector to store energy decay
    (*this).resetGlobalEnergy();
    this->globalEnergy.reserve(this->numberOfEchoes + 1); // '+1' to accomodate time 0.0

    // get initial energy global state
    double energySum = ((double) this->walkers.size()) * this->walkers[0].getEnergy();
    this->globalEnergy.push_back(energySum);

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values for sizing issues
    uint bitBlockColumns = this->bitBlock.blockColumns;
    uint bitBlockRows = this->bitBlock.blockRows;
    uint numberOfBitBlocks = this->bitBlock.numberOfBlocks;
    uint numberOfWalkers = this->numberOfWalkers;
    uint map_columns = this->bitBlock.imageColumns;
    uint map_rows = this->bitBlock.imageRows;
    uint map_depth = this->bitBlock.imageDepth;
    uint shiftConverter = log2(this->voxelDivision);

    uint numberOfEchoes = this->numberOfEchoes;
    uint stepsPerEcho = this->stepsPerEcho;

    // number of echos that each walker in kernel call will perform
    uint echoesPerKernel = this->rwNMR_config.getEchoesPerKernel();
    uint kernelCalls = (uint) ceil(numberOfEchoes / (double) echoesPerKernel);
    // uint lastEchoTail = numberOfEchoes - (kernelCalls * echoesPerKernel);

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    uint threadsPerBlock = this->rwNMR_config.getThreadsPerBlock();
    uint blocksPerKernel = this->rwNMR_config.getBlocks();
    uint walkersPerKernel = threadsPerBlock * blocksPerKernel;

    // treat case when only one kernel is needed
    if (numberOfWalkers < walkersPerKernel)
    {
        blocksPerKernel = (uint)ceil((double)(numberOfWalkers) / (double)(threadsPerBlock));

        // blocks per kernel should be multiple of 2
        if (blocksPerKernel % 2 == 1)
        {
            blocksPerKernel += 1;
        }

        walkersPerKernel = threadsPerBlock * blocksPerKernel;
    }

    // Walker packs == groups of walkers in the same kernel
    // all threads in a pack represent a walker in the NMR simulation
    // But, in the last pack, some threads may be idle
    uint numberOfWalkerPacks = (numberOfWalkers / walkersPerKernel) + 1;
    uint lastWalkerPackSize = numberOfWalkers % walkersPerKernel;
    uint lastWalkerPackTail = walkersPerKernel - lastWalkerPackSize;
    uint energyArraySize = walkersPerKernel;
    uint energyCollectorSize = (blocksPerKernel / 2);

    // Copy bitBlock3D data from host to device (only once)
    // assign pointer to bitBlock datastructure
    uint64_t *bitBlock;
    bitBlock = this->bitBlock.blocks;

    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Host and Device memory data allocation
    // pointers used in host array conversion
    int *walker_px = setIntArray_3D(walkersPerKernel);
    int *walker_py = setIntArray_3D(walkersPerKernel);
    int *walker_pz = setIntArray_3D(walkersPerKernel);
    double *penalty = setDoubleArray_3D(walkersPerKernel);
    double *energy = setDoubleArray_3D(echoesPerKernel * energyArraySize);
    double *energyCollector = setDoubleArray_3D(echoesPerKernel * energyCollectorSize);
    uint64_t *seed = setUInt64Array_3D(walkersPerKernel);

    // temporary array to collect energy contributions for each echo in a kernel
    double *temp_globalEnergy = setDoubleArray_3D((uint)echoesPerKernel);
    double *h_globalEnergy = setDoubleArray_3D(kernelCalls * echoesPerKernel);

// #pragma omp parallel for
    for (uint echo = 0; echo < numberOfEchoes; echo++)
    {
        h_globalEnergy[echo] = 0.0;
    }

    // Declaration of pointers to device data arrays
    int *d_walker_px;
    int *d_walker_py;
    int *d_walker_pz;
    double *d_penalty;
    double *d_energy;
    double *d_energyCollector;
    uint64_t *d_seed;

    // Memory allocation in device for data arrays
    cudaMalloc((void **)&d_walker_px, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_py, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_pz, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_penalty, walkersPerKernel * sizeof(double));
    cudaMalloc((void **)&d_energy, echoesPerKernel * energyArraySize * sizeof(double));
    cudaMalloc((void **)&d_energyCollector, echoesPerKernel * energyCollectorSize * sizeof(double));
    cudaMalloc((void **)&d_seed, walkersPerKernel * sizeof(uint64_t));

// #pragma omp parallel for
    for (uint i = 0; i < energyArraySize * echoesPerKernel; i++)
    {
        energy[i] = 0.0;
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

            #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, penalty, energy, seed, walkers) private(loop_start, loop_finish) 
            {
                const int thread_id = omp_get_thread_num();
                OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                loop_start = looper.getStart();
                loop_finish = looper.getFinish(); 

                for (uint i = loop_start; i < loop_finish; i++)
                {
                    walker_px[i] = this->walkers[i + packOffset].initialPosition.x;
                    walker_py[i] = this->walkers[i + packOffset].initialPosition.y;
                    walker_pz[i] = this->walkers[i + packOffset].initialPosition.z;
                    penalty[i] = this->walkers[i + packOffset].decreaseFactor;
                    energy[i + ((echoesPerKernel - 1) * energyArraySize)] = this->walkers[i + packOffset].energy;
                    seed[i] = this->walkers[i + packOffset].initialSeed;
                }
            }
        } else
        {            
            for (uint i = 0; i < walkersPerKernel; i++)
            {
                walker_px[i] = this->walkers[i + packOffset].initialPosition.x;
                walker_py[i] = this->walkers[i + packOffset].initialPosition.y;
                walker_pz[i] = this->walkers[i + packOffset].initialPosition.z;
                penalty[i] = this->walkers[i + packOffset].decreaseFactor;
                energy[i + ((echoesPerKernel - 1) * energyArraySize)] = this->walkers[i + packOffset].energy;
                seed[i] = this->walkers[i + packOffset].initialSeed;
            }
        }        

        // Device data copy
        // copy host data to device
        cudaMemcpy(d_walker_px, walker_px, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, walker_py, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_penalty, penalty, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, energy, echoesPerKernel * energyArraySize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // Launch kernel for GPU computation
        for (uint kernelId = 0; kernelId < kernelCalls; kernelId++)
        {
            // define echo offset
            uint echoOffset = kernelId * echoesPerKernel;
            uint echoes = echoesPerKernel;

            // call "walk" method kernel
            walk_3D<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                          d_walker_py,
                                                          d_walker_pz,
                                                          d_penalty,
                                                          d_energy,
                                                          d_seed,
                                                          d_bitBlock,
                                                          bitBlockColumns,
                                                          bitBlockRows,
                                                          walkersPerKernel,
                                                          energyArraySize,
                                                          echoes,
                                                          stepsPerEcho,
                                                          map_columns,
                                                          map_rows,
                                                          map_depth,
                                                          shiftConverter);
            cudaDeviceSynchronize();

            // launch globalEnergy "reduce" kernel
            energyReduce_shared_3D<<<blocksPerKernel / 2,
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
            reduce_omp_3D(temp_globalEnergy, energyCollector, echoesPerKernel, blocksPerKernel / 2);

            // copy data from temporary array to NMR_Simulation2D "globalEnergy" vector class member
            for (uint echo = 0; echo < echoesPerKernel; echo++)
            {
                h_globalEnergy[echo + echoOffset] += temp_globalEnergy[echo];
            }
        }
    }

    if (lastWalkerPackSize > 0)
    {
        // last Walker pack is done explicitly
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

            #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, penalty, energy, seed, walkers) private(loop_start, loop_finish) 
            {
                const int thread_id = omp_get_thread_num();
                OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                loop_start = looper.getStart();
                loop_finish = looper.getFinish(); 

                for (uint i = loop_start; i < loop_finish; i++)
                {
                    walker_px[i] = this->walkers[i + packOffset].initialPosition.x;
                    walker_py[i] = this->walkers[i + packOffset].initialPosition.y;
                    walker_pz[i] = this->walkers[i + packOffset].initialPosition.z;
                    penalty[i] = this->walkers[i + packOffset].decreaseFactor;
                    energy[i + ((echoesPerKernel - 1) * energyArraySize)] = this->walkers[i + packOffset].energy;
                    seed[i] = this->walkers[i + packOffset].initialSeed;
                }
            }
        } else
        {            
            for (uint i = 0; i < lastWalkerPackSize; i++)
            {
                walker_px[i] = this->walkers[i + packOffset].initialPosition.x;
                walker_py[i] = this->walkers[i + packOffset].initialPosition.y;
                walker_pz[i] = this->walkers[i + packOffset].initialPosition.z;
                penalty[i] = this->walkers[i + packOffset].decreaseFactor;
                energy[i + ((echoesPerKernel - 1) * energyArraySize)] = this->walkers[i + packOffset].energy;
                seed[i] = this->walkers[i + packOffset].initialSeed;
            }
        }   

        // complete energy array data
        for (uint echo = 0; echo < echoesPerKernel; echo++)
        {
            for (uint i = 0; i < lastWalkerPackTail; i++)
            {
                {
                    energy[i + lastWalkerPackSize + (echo * energyArraySize)] = 0.0;
                }
            }
        }
        // Device data copy
        // copy host data to device
        cudaMemcpy(d_walker_px, walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_penalty, penalty, lastWalkerPackSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, energy, echoesPerKernel * energyArraySize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // Launch kernel for GPU computation
        for (uint kernelId = 0; kernelId < kernelCalls; kernelId++)
        {
            // define echo offset
            uint echoOffset = kernelId * echoesPerKernel;
            uint echoes = echoesPerKernel;

            // call "walk" method kernel
            walk_3D<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                          d_walker_py,
                                                          d_walker_pz,
                                                          d_penalty,
                                                          d_energy,
                                                          d_seed,
                                                          d_bitBlock,
                                                          bitBlockColumns,
                                                          bitBlockRows,
                                                          lastWalkerPackSize,
                                                          energyArraySize,
                                                          echoes,
                                                          stepsPerEcho,
                                                          map_columns,
                                                          map_rows,
                                                          map_depth,
                                                          shiftConverter);
            cudaDeviceSynchronize();

            // launch globalEnergy "reduce" kernel
            energyReduce_shared_3D<<<blocksPerKernel / 2,
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
            reduce_omp_3D(temp_globalEnergy, energyCollector, echoesPerKernel, blocksPerKernel / 2);

            // copy data from temporary array
            for (uint echo = 0; echo < echoesPerKernel; echo++)
            {
                h_globalEnergy[echo + echoOffset] += temp_globalEnergy[echo];
            }
        }
    }

    // insert to object energy values computed in gpu
    for (uint echo = 0; echo < numberOfEchoes; echo++)
    {
        this->globalEnergy.push_back(h_globalEnergy[echo]);
    }

    // free pointers in host
    free(walker_px);
    free(walker_py);
    free(walker_pz);
    free(penalty);
    free(h_globalEnergy);
    free(energy);
    free(energyCollector);
    free(temp_globalEnergy);
    free(seed);

    // and direct them to NULL
    walker_px = NULL;
    walker_py = NULL;
    walker_pz = NULL;
    penalty = NULL;
    h_globalEnergy = NULL;
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
    cudaFree(d_walker_pz);
    cudaFree(d_penalty);
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

// function to call GPU kernel to execute
// walker's "walk" method in Graphics Processing Unit
double NMR_Simulation::diffusionSimulation_CUDA(double gradientMagnitude, double tinyDelta, double giromagneticRatio)
{
    cout << "initializing RW-PFG NMR simulation in GPU... ";
    // reset walker's initial state with omp parallel for
// #pragma if(NMR_OPENMP) omp parallel for private(id) shared(walkers)
    for (uint id = 0; id < this->walkers.size(); id++)
    {
        this->walkers[id].resetPosition();
        this->walkers[id].resetSeed();
        this->walkers[id].resetEnergy();
    }

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values for sizing issues
    uint bitBlockColumns = this->bitBlock.blockColumns;
    uint bitBlockRows = this->bitBlock.blockRows;
    uint numberOfBitBlocks = this->bitBlock.numberOfBlocks;
    uint numberOfWalkers = this->numberOfWalkers;
    uint map_columns = this->bitBlock.imageColumns;
    uint map_rows = this->bitBlock.imageRows;
    uint map_depth = this->bitBlock.imageDepth;
    double voxelResolution = this->imageVoxelResolution;
    uint numberOfSteps = this->simulationSteps;
    uint shiftConverter = log2(this->voxelDivision);

    // uint numberOfEchoes = this->numberOfEchoes;
    // uint stepsPerEcho = this->stepsPerEcho;

    // number of echos that each walker in kernel call will perform
    // uint echoesPerKernel = ECHOESPERKERNEL;
    // uint kernelCalls = (uint)ceil((double)numberOfEchoes / (double)echoesPerKernel);

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    uint threadsPerBlock = this->rwNMR_config.getThreadsPerBlock();
    uint blocksPerKernel = this->rwNMR_config.getBlocks();
    uint walkersPerKernel = threadsPerBlock * blocksPerKernel;

    // treat case when only one kernel is needed
    if (numberOfWalkers < walkersPerKernel)
    {
        blocksPerKernel = (uint)ceil((double)(numberOfWalkers) / (double)(threadsPerBlock));

        // blocks per kernel should be multiple of 2
        if (blocksPerKernel % 2 == 1)
        {
            blocksPerKernel += 1;
        }

        walkersPerKernel = threadsPerBlock * blocksPerKernel;
    }

    // Walker packs == groups of walkers in the same kernel
    // all threads in a pack represent a walker in the NMR simulation
    // But, in the last pack, some threads may be idle
    uint numberOfWalkerPacks = (numberOfWalkers / walkersPerKernel) + 1;
    uint lastWalkerPackSize = numberOfWalkers % walkersPerKernel;
    uint lastWalkerPackTail = walkersPerKernel - lastWalkerPackSize;

    // // debug
    // cout << endl << endl;
    // cout << "threads per block: " << threadsPerBlock << endl;
    // cout << "blocks per kernel: " << blocksPerKernel << endl;
    // cout << "walkers per kernel: " << walkersPerKernel << endl;
    // cout << "walker packs: " << numberOfWalkerPacks << endl;
    // cout << "last pack size: " << lastWalkerPackSize << endl;
    // cout << "last pack tail: " << lastWalkerPackTail << endl << endl;


    // derivatives
    uint energyArraySize = walkersPerKernel;
    uint phaseArraySize = walkersPerKernel;
    uint energyCollectorSize = (blocksPerKernel / 2);
    uint phaseCollectorSize = (blocksPerKernel / 2);

    // Copy bitBlock3D data from host to device (only once)
    // assign pointer to bitBlock datastructure
    uint64_t *h_bitBlock;
    h_bitBlock = this->bitBlock.blocks;

    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, h_bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Host and Device memory data allocation
    // pointers used in host array conversion
    int *h_walker_x0 = setIntArray_3D(walkersPerKernel);
    int *h_walker_y0 = setIntArray_3D(walkersPerKernel);
    int *h_walker_z0 = setIntArray_3D(walkersPerKernel);
    int *h_walker_px = setIntArray_3D(walkersPerKernel);
    int *h_walker_py = setIntArray_3D(walkersPerKernel);
    int *h_walker_pz = setIntArray_3D(walkersPerKernel);
    double *h_penalty = setDoubleArray_3D(walkersPerKernel);
    uint64_t *h_seed = setUInt64Array_3D(walkersPerKernel);

    // temporary array to collect energy contributions for each echo in a kernel
    // double *temp_globalEnergy = setDoubleArray_3D((uint)echoesPerKernel);
    // double *h_globalEnergy = setDoubleArray_3D(kernelCalls * echoesPerKernel);
    double *h_energy = setDoubleArray_3D(energyArraySize);
    double *h_energyCollector = setDoubleArray_3D(energyCollectorSize);
    double h_globalEnergy = 0.0;

    // magnetization phase
    double *h_phase = setDoubleArray_3D(phaseArraySize);
    double *h_phaseCollector = setDoubleArray_3D(phaseCollectorSize);
    double h_globalPhase = 0.0;

    // #pragma omp parallel for
    // for (uint echo = 0; echo < numberOfEchoes; echo++)
    // {
    //     h_globalEnergy[echo] = 0.0;
    // }

    // Declaration of pointers to device data arrays
    int *d_walker_x0;
    int *d_walker_y0;
    int *d_walker_z0;
    int *d_walker_px;
    int *d_walker_py;
    int *d_walker_pz;
    double *d_penalty;
    uint64_t *d_seed;

    // magnetization & phase
    double *d_energy;
    double *d_energyCollector;   
    double *d_phase;
    double *d_phaseCollector;

    // Memory allocation in device for data arrays
    cudaMalloc((void **)&d_walker_x0, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_y0, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_z0, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_px, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_py, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_pz, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_penalty, walkersPerKernel * sizeof(double));
    cudaMalloc((void **)&d_seed, walkersPerKernel * sizeof(uint64_t));
    cudaMalloc((void **)&d_energy, energyArraySize * sizeof(double));
    cudaMalloc((void **)&d_energyCollector, energyCollectorSize * sizeof(double));
    cudaMalloc((void **)&d_phase, phaseArraySize * sizeof(double));
    cudaMalloc((void **)&d_phaseCollector, phaseCollectorSize * sizeof(double));
    

// #pragma omp parallel for
    for (uint idx = 0; idx < energyArraySize; idx++)
    {
        h_energy[idx] = 0.0;
    }
    
    for(uint idx = 0; idx < energyCollectorSize; idx++)
    {
        h_energyCollector[idx] = 0.0;
    }

    for (uint idx = 0; idx < phaseArraySize; idx++)
    {
        h_phase[idx] = 0.0;
    } 

    for(uint idx = 0; idx < phaseCollectorSize; idx++)
    {
        h_phaseCollector[idx] = 0.0;
    }

    // PFG main loop
    for (uint packId = 0; packId < (numberOfWalkerPacks - 1); packId++)
    {
        // set offset in walkers vector
        uint packOffset = packId * walkersPerKernel;

        // Host data copy
        // copy original walkers' data to temporary host arrays
        // #pragma omp parallel for
        for (uint i = 0; i < walkersPerKernel; i++)
        {
            h_walker_x0[i] = this->walkers[i + packOffset].initialPosition.x;
            h_walker_y0[i] = this->walkers[i + packOffset].initialPosition.y;
            h_walker_z0[i] = this->walkers[i + packOffset].initialPosition.z;
            h_walker_px[i] = this->walkers[i + packOffset].initialPosition.x;
            h_walker_py[i] = this->walkers[i + packOffset].initialPosition.y;
            h_walker_pz[i] = this->walkers[i + packOffset].initialPosition.z;
            h_penalty[i] = this->walkers[i + packOffset].decreaseFactor;
            h_seed[i] = this->walkers[i + packOffset].initialSeed;
            h_energy[i] = this->walkers[i + packOffset].energy;
            h_phase[i] = this->walkers[i + packOffset].energy;
        }

        // Device data copy
        // copy host data to device
        cudaMemcpy(d_walker_x0, h_walker_x0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_y0, h_walker_y0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_z0, h_walker_z0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_px, h_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, h_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, h_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_penalty, h_penalty, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, h_seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, h_energy, energyArraySize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase, h_phase, phaseArraySize * sizeof(double), cudaMemcpyHostToDevice);
        

        // Launch kernel for GPU computation
        // call "walk" method kernel
        walk_PFG<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                       d_walker_py,
                                                       d_walker_pz,
                                                       d_penalty,
                                                       d_energy,
                                                       d_seed,
                                                       d_bitBlock,
                                                       bitBlockColumns,
                                                       bitBlockRows,
                                                       walkersPerKernel,
                                                       energyArraySize,
                                                       numberOfSteps,
                                                       map_columns,
                                                       map_rows,
                                                       map_depth,
                                                       shiftConverter);
        cudaDeviceSynchronize();                  
        

       
        // kernel call to compute walkers individual phase
        measure_PFG<<<blocksPerKernel, threadsPerBlock>>> (d_walker_x0,
                                                           d_walker_y0, 
                                                           d_walker_z0,
                                                           d_walker_px,
                                                           d_walker_py,
                                                           d_walker_pz,
                                                           d_energy,
                                                           d_phase,
                                                           walkersPerKernel,
                                                           voxelResolution,
                                                           gradientMagnitude,
                                                           tinyDelta,
                                                           giromagneticRatio);
        cudaDeviceSynchronize();

        if(this->rwNMR_config.getReduceInGPU())
        {
            // Kernel call to reduce walker final phases
            reduce_PFG<<<blocksPerKernel/2, 
                         threadsPerBlock, 
                         threadsPerBlock * sizeof(double)>>>(d_phase,
                                                             d_phaseCollector,
                                                             phaseArraySize,
                                                             phaseCollectorSize);    
            cudaDeviceSynchronize();
    
            // Kernel call to reduce walker final energies
            reduce_PFG<<<blocksPerKernel/2, 
                         threadsPerBlock, 
                         threadsPerBlock * sizeof(double)>>>(d_energy,
                                                             d_energyCollector,
                                                             energyArraySize,
                                                             energyCollectorSize);     
            cudaDeviceSynchronize();
    
            // copy data from gatherer array
            cudaMemcpy(h_phaseCollector, d_phaseCollector, phaseCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_energyCollector, d_energyCollector, energyCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);
    
            // collector reductions
            for(uint idx = 0; idx < phaseCollectorSize; idx++)
            {
                h_globalPhase += h_phaseCollector[idx];
            }
            for(uint idx = 0; idx < energyCollectorSize; idx++)
            {
                h_globalEnergy += h_energyCollector[idx];
            }
        } 
        else
        {    
            
            cudaMemcpy(h_phase, d_phase, phaseArraySize * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_energy, d_energy, energyArraySize * sizeof(double), cudaMemcpyDeviceToHost);            
            for(uint idx = 0; idx < phaseArraySize; idx++)
            {
                h_globalPhase += h_phase[idx];
            }
            for(uint idx = 0; idx < energyArraySize; idx++)
            {
                h_globalEnergy += h_energy[idx];
            } 
        }   

    }

    if (lastWalkerPackSize > 0)
    {
        // last Walker pack is done explicitly
        // set offset in walkers vector
        uint packOffset = (numberOfWalkerPacks - 1) * walkersPerKernel;

        // Host data copy
        // copy original walkers' data to temporary host arrays
        for (uint i = 0; i < lastWalkerPackSize; i++)
        {
            h_walker_x0[i] = this->walkers[i + packOffset].initialPosition.x;
            h_walker_y0[i] = this->walkers[i + packOffset].initialPosition.y;
            h_walker_z0[i] = this->walkers[i + packOffset].initialPosition.z;
            h_walker_px[i] = this->walkers[i + packOffset].initialPosition.x;
            h_walker_py[i] = this->walkers[i + packOffset].initialPosition.y;
            h_walker_pz[i] = this->walkers[i + packOffset].initialPosition.z;
            h_penalty[i] = this->walkers[i + packOffset].decreaseFactor;
            h_seed[i] = this->walkers[i + packOffset].initialSeed;
            h_energy[i] = this->walkers[i + packOffset].energy;
            h_phase[i] = this->walkers[i + packOffset].energy;
        }

        // complete energy array data
        for (uint i = 0; i < lastWalkerPackTail; i++)
        {
            {
                h_energy[i + lastWalkerPackSize] = 0.0;
                h_phase[i + lastWalkerPackSize] = 0.0;
            }
        }

        // Device data copy
        // copy host data to device
        cudaMemcpy(d_walker_x0, h_walker_x0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_y0, h_walker_y0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_z0, h_walker_z0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_px, h_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, h_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, h_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_penalty, h_penalty, lastWalkerPackSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, h_seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, h_energy, energyArraySize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase, h_phase, phaseArraySize * sizeof(double), cudaMemcpyHostToDevice);
        

        // Launch kernel for GPU computation
        // call "walk" method kernel
        walk_PFG<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                       d_walker_py,
                                                       d_walker_pz,
                                                       d_penalty,
                                                       d_energy,
                                                       d_seed,
                                                       d_bitBlock,
                                                       bitBlockColumns,
                                                       bitBlockRows,
                                                       lastWalkerPackSize,
                                                       energyArraySize,
                                                       numberOfSteps,
                                                       map_columns,
                                                       map_rows,
                                                       map_depth,
                                                       shiftConverter);
        cudaDeviceSynchronize();
       

        // compute phase
        measure_PFG<<<blocksPerKernel, threadsPerBlock>>> (d_walker_x0,
                                                           d_walker_y0, 
                                                           d_walker_z0,
                                                           d_walker_px,
                                                           d_walker_py,
                                                           d_walker_pz,
                                                           d_energy,
                                                           d_phase,
                                                           lastWalkerPackSize,
                                                           voxelResolution,
                                                           gradientMagnitude,
                                                           tinyDelta,
                                                           giromagneticRatio);

        cudaDeviceSynchronize();

        if(this->rwNMR_config.getReduceInGPU())
        {
            // Kernel call to reduce walker final phases
            reduce_PFG<<<blocksPerKernel/2, 
                         threadsPerBlock, 
                         threadsPerBlock * sizeof(double)>>>(d_phase,
                                                             d_phaseCollector,
                                                             phaseArraySize,
                                                             phaseCollectorSize);
    
            cudaDeviceSynchronize();
    
            // Kernel call to reduce walker final energies
            reduce_PFG<<<blocksPerKernel/2, 
                         threadsPerBlock, 
                         threadsPerBlock * sizeof(double)>>>(d_energy,
                                                             d_energyCollector,
                                                             energyArraySize,
                                                             energyCollectorSize);
    
            cudaDeviceSynchronize();
    
            // copy data from gatherer array
            cudaMemcpy(h_phaseCollector, d_phaseCollector, phaseCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_energyCollector, d_energyCollector, energyCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);
    
            // collector reductions
            for(uint idx = 0; idx < phaseCollectorSize; idx++)
            {
                h_globalPhase += h_phaseCollector[idx];
            }
            for(uint idx = 0; idx < energyCollectorSize; idx++)
            {
                h_globalEnergy += h_energyCollector[idx];
            }
        }
        else
        {
            cudaMemcpy(h_phase, d_phase, phaseArraySize * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_energy, d_energy, energyArraySize * sizeof(double), cudaMemcpyDeviceToHost);
            
            for(uint idx = 0; idx < phaseArraySize; idx++)
            {
                h_globalPhase += h_phase[idx];
            }
            for(uint idx = 0; idx < energyArraySize; idx++)
            {
                h_globalEnergy += h_energy[idx];
            }
        }

    }

    // free pointers in host
    free(h_walker_x0);
    free(h_walker_y0);
    free(h_walker_z0);
    free(h_walker_px);
    free(h_walker_py);
    free(h_walker_pz);
    free(h_penalty);
    free(h_seed);
    free(h_energy);
    free(h_energyCollector);
    free(h_phase);
    free(h_phaseCollector);

    // and direct them to NULL
    h_walker_px = NULL;
    h_walker_py = NULL;
    h_walker_pz = NULL;
    h_penalty = NULL;
    h_energy = NULL;
    h_energyCollector = NULL;
    h_seed = NULL;
    h_phase = NULL;
    h_phaseCollector = NULL;

    // also direct the bitBlock pointer created in this context
    // (original data is kept safe)
    h_bitBlock = NULL;

    // free device global memory
    cudaFree(d_walker_x0);
    cudaFree(d_walker_y0);
    cudaFree(d_walker_z0);
    cudaFree(d_walker_px);
    cudaFree(d_walker_py);
    cudaFree(d_walker_pz);
    cudaFree(d_penalty);
    cudaFree(d_seed);
    cudaFree(d_energy);
    cudaFree(d_energyCollector);
    cudaFree(d_phase);
    cudaFree(d_phaseCollector);
    cudaFree(d_bitBlock);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Completed.\telapsed time: " << elapsedTime * 1.0e-3 << endl;

    // debug
    // cout << "Gradient: " << gradientMagnitude << "e+08 T/um" << endl;
    // cout << "tiny delta: " << tinyDelta << "ms" << endl;
    // cout << "Gradient: " << giromagneticRatio << " x 2pi x 1e6 /Ts" << endl;
    // cout << "final signal = " << h_globalEnergy << endl;
    // cout << "final phase = " << h_globalPhase << endl;
    // cout << "phase / signal = " << h_globalPhase / h_globalEnergy << endl;

    cudaDeviceReset();
    return (h_globalPhase / h_globalEnergy);
}

/////////////////////////////////////////////////////////////////////
//////////////////////// HOST FUNCTIONS ///////////////////////////
/////////////////////////////////////////////////////////////////////
void reduce_omp_3D(double *temp_collector, double *array, int numberOfEchoes, uint arraySizePerEcho)
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

void test_omp_3D(uint size)
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
int *setIntArray_3D(uint size)
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

uint *setUIntArray_3D(uint size)
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

double *setDoubleArray_3D(uint size)
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

uint64_t *setUInt64Array_3D(uint size)
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

void copyVectorBtoA_3D(int a[], int b[], uint size)
{
    for (uint i = 0; i < size; i++)
    {
        a[i] = b[i];
    }
}

void copyVectorBtoA_3D(double a[], double b[], int size)
{
    for (uint i = 0; i < size; i++)
    {
        a[i] = b[i];
    }
}

void copyVectorBtoA_3D(uint64_t a[], uint64_t b[], int size)
{
    for (uint i = 0; i < size; i++)
    {
        a[i] = b[i];
    }
}

void vectorElementSwap_3D(int *vector, uint index1, uint index2)
{
    int temp;

    temp = vector[index1];
    vector[index1] = vector[index2];
    vector[index2] = temp;
}

void vectorElementSwap_3D(double *vector, uint index1, uint index2)
{
    double temp;

    temp = vector[index1];
    vector[index1] = vector[index2];
    vector[index2] = temp;
}

void vectorElementSwap_3D(uint64_t *vector, uint index1, uint index2)
{
    uint64_t temp;

    temp = vector[index1];
    vector[index1] = vector[index2];
    vector[index2] = temp;
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