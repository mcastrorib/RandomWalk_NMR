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
#include <cmath>

// include C standard library for memory allocation using pointers
#include <stdlib.h>

// include OpenMP for multicore implementation
#include <omp.h>
#include "../Utils/OMPLoopEnabler.h"

//include
#include "NMR_defs.h"
#include "../Walker/walker.h"
#include "../RNG/xorshift.h"
#include "NMR_Simulation.h"
#include "NMR_pfgse.h"
#include "NMR_pfgse_cuda.h"


// GPU kernel for NMR simulation - a.k.a. walker's relaxation/demagnetization
// in this kernel, each thread will behave as a solitary walker
__global__ void PFG_walk(int *walker_px,
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
    int localPosX, localPosY, localPosZ;
    double local_penalty;
    uint64_t local_seed;
    double energyLvl;

    // thread variables for future movements
    int next_x, next_y, next_z;
    direction nextDirection = None;

    // now begin the "walk" procedure de facto
    if (walkerId < numberOfWalkers)
    {
        // Local variables for unique read from device global memory
        localPosX = walker_px[walkerId];
        localPosY = walker_py[walkerId];
        localPosZ = walker_pz[walkerId];
        local_penalty = penalty[walkerId];
        local_seed = seed[walkerId];
        energyLvl = energy[walkerId];

        // for (int echo = 0; echo < numberOfEchoes; echo++)
        // {
        
            for(int step = 0; step < numberOfSteps; step++)
            {
                nextDirection = computeNextDirection_PFG(local_seed);
            
                nextDirection = checkBorder_PFG(convertLocalToGlobal(localPosX, shift_convert),
                                                convertLocalToGlobal(localPosY, shift_convert),
                                                convertLocalToGlobal(localPosZ, shift_convert),
                                                nextDirection,
                                                map_columns,
                                                map_rows,
                                                map_depth);
    
                computeNextPosition_PFG(localPosX,
                                        localPosY,
                                        localPosZ,
                                        nextDirection,
                                        next_x,
                                        next_y,
                                        next_z);
    
                if (checkNextPosition_PFG(convertLocalToGlobal(next_x, shift_convert), 
                                          convertLocalToGlobal(next_y, shift_convert), 
                                          convertLocalToGlobal(next_z, shift_convert), 
                                          bitBlock, 
                                          bitBlockColumns, 
                                          bitBlockRows))
                {
                    // position is valid
                    localPosX = next_x;
                    localPosY = next_y;
                    localPosZ = next_z;
                }
                else
                {
                    // walker chocks with wall and comes back to the same position
                    // walker loses energy due to this collision
                    energyLvl = energyLvl * local_penalty;
                }
            }
        // }

        // walker's energy device global memory update
        // must be done for each echo
        energy[walkerId] = energyLvl;

        // position and seed device global memory update
        // must be done for each kernel
        walker_px[walkerId] = localPosX;
        walker_py[walkerId] = localPosY;
        walker_pz[walkerId] = localPosZ;
        seed[walkerId] = local_seed;
    }
}

// GPU kernel for NMR simulation - a.k.a. walker's relaxation/demagnetization
// in this kernel, each thread will behave as a solitary walker
__global__ void PFG_measure_old(int *walker_x0,
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
                                const double pulse_width,
                                const double giromagneticRatio)
{
    // identify thread's walker
    int walkerId = threadIdx.x + blockIdx.x * blockDim.x;

    if(walkerId < numberOfWalkers)
    {
        // compute local magnetizion magnitude
        double K = compute_PFG_k_value(gradient, pulse_width, giromagneticRatio);
        double local_phase = K * (walker_zF[walkerId] - walker_z0[walkerId]) * voxelResolution; 
        double magnitude_real_value = cos(local_phase);
        double magnitude_imag_value = sin(local_phase);
    
        // update global value 
        phase[walkerId] = magnitude_real_value * energy[walkerId];
    }
}

// GPU kernel for NMR simulation - a.k.a. walker's relaxation/demagnetization
// in this kernel, each thread will behave as a solitary walker
__global__ void PFG_measure(int *walker_x0,
                            int *walker_y0, 
                            int *walker_z0,
                            int *walker_xF,
                            int *walker_yF,
                            int *walker_zF,
                            double *energy,
                            double *phase,
                            const uint numberOfWalkers,
                            const double voxelResolution,
                            const double k_X,
                            const double k_Y,
                            const double k_Z)
{
    // identify thread's walker
    int walkerId = threadIdx.x + blockIdx.x * blockDim.x;

    if(walkerId < numberOfWalkers)
    {
        // compute local magnetizion magnitude
        double dX = (walker_xF[walkerId] - walker_x0[walkerId]);
        double dY = (walker_yF[walkerId] - walker_y0[walkerId]);
        double dZ = (walker_zF[walkerId] - walker_z0[walkerId]);

        double local_phase = dotProduct(k_X,k_Y,k_Z,dX,dY,dZ) * voxelResolution; 
        double magnitude_real_value = cos(local_phase);
        double magnitude_imag_value = sin(local_phase);
    
        // update global value 
        phase[walkerId] = magnitude_real_value * energy[walkerId];
    }
}

__global__ void PFG_reduce(double *data,
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
// walker's "walk" method in Graphics Processing Unit
void NMR_PFGSE::simulation_cuda_old()
{
    cout << "initializing RW-PFG NMR simulation in GPU... ";

    double gradientMagnitude = 0;
    double pulseWidth = this->pulseWidth;
    double giromagneticRatio = this->giromagneticRatio;
    if(!PFGSE_USE_TWOPI) giromagneticRatio /= TWO_PI;
    uint gradientPoints = this->gradientPoints;


    // reset walker's initial state with omp parallel for
    for (uint id = 0; id < this->NMR.walkers.size(); id++)
    {
        this->NMR.walkers[id].resetPosition();
        this->NMR.walkers[id].resetSeed();
        this->NMR.walkers[id].resetEnergy();
    }

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values for sizing issues
    uint bitBlockColumns = this->NMR.bitBlock.blockColumns;
    uint bitBlockRows = this->NMR.bitBlock.blockRows;
    uint numberOfBitBlocks = this->NMR.bitBlock.numberOfBlocks;
    uint numberOfWalkers = this->NMR.numberOfWalkers;
    uint map_columns = this->NMR.bitBlock.imageColumns;
    uint map_rows = this->NMR.bitBlock.imageRows;
    uint map_depth = this->NMR.bitBlock.imageDepth;
    uint shiftConverter = log2(this->NMR.voxelDivision);
    double voxelResolution = this->NMR.imageVoxelResolution;
    uint numberOfSteps = this->NMR.simulationSteps;

    // create a steps bucket
    uint stepsLimit = MAX_RWSTEPS;
    uint stepsSize = numberOfSteps/stepsLimit;
    vector<uint> steps;
    for(uint idx = 0; idx < stepsSize; idx++)
    {
        steps.push_back(stepsLimit);
    }
    // charge rebalance
    if((numberOfSteps % stepsLimit) > 0)
    {
        stepsSize++;
        steps.push_back(numberOfSteps%stepsLimit);
    } 
    

    // number of echos that each walker in kernel call will perform
    // uint echoesPerKernel = ECHOESPERKERNEL;
    // uint kernelCalls = (uint)ceil((double)numberOfEchoes / (double)echoesPerKernel);

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    uint threadsPerBlock = THREADSPERBLOCK;
    uint blocksPerKernel = BLOCKS;
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
    h_bitBlock = this->NMR.bitBlock.blocks;

    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, h_bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Host and Device memory data allocation
    // pointers used in host array conversion
    int *h_walker_x0 = setIntArray_PFG(walkersPerKernel);
    int *h_walker_y0 = setIntArray_PFG(walkersPerKernel);
    int *h_walker_z0 = setIntArray_PFG(walkersPerKernel);
    int *h_walker_px = setIntArray_PFG(walkersPerKernel);
    int *h_walker_py = setIntArray_PFG(walkersPerKernel);
    int *h_walker_pz = setIntArray_PFG(walkersPerKernel);
    double *h_penalty = setDoubleArray_PFG(walkersPerKernel);
    uint64_t *h_seed = setUInt64Array_PFG(walkersPerKernel);

    // temporary array to collect energy contributions for each echo in a kernel
    // double *temp_globalEnergy = setDoubleArray_3D((uint)echoesPerKernel);
    // double *h_globalEnergy = setDoubleArray_3D(kernelCalls * echoesPerKernel);
    double *h_energy = setDoubleArray_PFG(energyArraySize);
    double *h_energyCollector = setDoubleArray_PFG(energyCollectorSize);
    double *h_globalEnergy = setDoubleArray_PFG(gradientPoints);

    // magnetization phase
    double *h_phase = setDoubleArray_PFG(phaseArraySize);
    double *h_phaseCollector = setDoubleArray_PFG(phaseCollectorSize);
    double *h_globalPhase = setDoubleArray_PFG(gradientPoints);

    // #pragma omp parallel for
    for (uint point = 0; point < gradientPoints; point++)
    {
        h_globalEnergy[point] = 0.0;
        h_globalPhase[point] = 0.0;
    }

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
            h_walker_x0[i] = this->NMR.walkers[i + packOffset].initialPosition.x;
            h_walker_y0[i] = this->NMR.walkers[i + packOffset].initialPosition.y;
            h_walker_z0[i] = this->NMR.walkers[i + packOffset].initialPosition.z;
            h_walker_px[i] = this->NMR.walkers[i + packOffset].initialPosition.x;
            h_walker_py[i] = this->NMR.walkers[i + packOffset].initialPosition.y;
            h_walker_pz[i] = this->NMR.walkers[i + packOffset].initialPosition.z;
            h_penalty[i] = this->NMR.walkers[i + packOffset].decreaseFactor;
            h_seed[i] = this->NMR.walkers[i + packOffset].initialSeed;
            h_energy[i] = this->NMR.walkers[i + packOffset].energy;
            h_phase[i] = this->NMR.walkers[i + packOffset].energy;
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
        for(uint step = 0; step < steps.size(); step++)
        {
            PFG_walk<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
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
                                                           steps[step],
                                                           map_columns,
                                                           map_rows,
                                                           map_depth,
                                                           shiftConverter);
            cudaDeviceSynchronize();  
        }

        // recover last positions
        cudaMemcpy(h_walker_px, d_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_py, d_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_pz, d_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        for (uint i = 0; i < walkersPerKernel; i++)
        {
            this->NMR.walkers[i + packOffset].position_x = h_walker_px[i];
            this->NMR.walkers[i + packOffset].position_y = h_walker_py[i];
            this->NMR.walkers[i + packOffset].position_z = h_walker_pz[i];            
        }
        

        for(int point = 0; point < gradientPoints; point++)
        {
            gradientMagnitude = this->gradient[point];

            // kernel call to compute walkers individual phase
            PFG_measure_old<<<blocksPerKernel, threadsPerBlock>>> ( d_walker_x0,
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
                                                                    pulseWidth,
                                                                    giromagneticRatio);
            cudaDeviceSynchronize();

            if(REDUCE_IN_GPU)
            {
                // Kernel call to reduce walker final phases
                PFG_reduce<<<blocksPerKernel/2, 
                             threadsPerBlock, 
                             threadsPerBlock * sizeof(double)>>>(d_phase,
                                                                 d_phaseCollector,  
                                                                 phaseArraySize,
                                                                 phaseCollectorSize);       
                cudaDeviceSynchronize();

                // Kernel call to reduce walker final energies
                PFG_reduce<<<blocksPerKernel/2, 
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
                    h_globalPhase[point] += h_phaseCollector[idx];
                }
                for(uint idx = 0; idx < energyCollectorSize; idx++)
                {
                    h_globalEnergy[point] += h_energyCollector[idx];
                }
            } 
            else
            {    

                cudaMemcpy(h_phase, d_phase, phaseArraySize * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_energy, d_energy, energyArraySize * sizeof(double), cudaMemcpyDeviceToHost);            
                for(uint idx = 0; idx < phaseArraySize; idx++)
                {
                    h_globalPhase[point] += h_phase[idx];
                }
                for(uint idx = 0; idx < energyArraySize; idx++)
                {
                    h_globalEnergy[point] += h_energy[idx];
                } 
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
            h_walker_x0[i] = this->NMR.walkers[i + packOffset].initialPosition.x;
            h_walker_y0[i] = this->NMR.walkers[i + packOffset].initialPosition.y;
            h_walker_z0[i] = this->NMR.walkers[i + packOffset].initialPosition.z;
            h_walker_px[i] = this->NMR.walkers[i + packOffset].initialPosition.x;
            h_walker_py[i] = this->NMR.walkers[i + packOffset].initialPosition.y;
            h_walker_pz[i] = this->NMR.walkers[i + packOffset].initialPosition.z;
            h_penalty[i] = this->NMR.walkers[i + packOffset].decreaseFactor;
            h_seed[i] = this->NMR.walkers[i + packOffset].initialSeed;
            h_energy[i] = this->NMR.walkers[i + packOffset].energy;
            h_phase[i] = this->NMR.walkers[i + packOffset].energy;
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
        for(uint step = 0; step < steps.size(); step++)
        {
            PFG_walk<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
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
                                                           steps[step],
                                                           map_columns,
                                                           map_rows,
                                                           map_depth,
                                                           shiftConverter);
            cudaDeviceSynchronize();  
        }

        // recover last positions
        cudaMemcpy(h_walker_px, d_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_py, d_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_pz, d_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        for (uint i = 0; i < lastWalkerPackSize; i++)
        {
            this->NMR.walkers[i + packOffset].position_x = h_walker_px[i];
            this->NMR.walkers[i + packOffset].position_y = h_walker_py[i];
            this->NMR.walkers[i + packOffset].position_z = h_walker_pz[i];            
        }
       

        for(int point = 0; point < this->gradientPoints; point++)
        {
            gradientMagnitude = this->gradient[point];
            // compute phase
            PFG_measure_old<<<blocksPerKernel, threadsPerBlock>>> (d_walker_x0,
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
                                                                   pulseWidth,
                                                                   giromagneticRatio);

            cudaDeviceSynchronize();

            if(REDUCE_IN_GPU)
            {
                // Kernel call to reduce walker final phases
                PFG_reduce<<<blocksPerKernel/2, 
                             threadsPerBlock, 
                             threadsPerBlock * sizeof(double)>>>(d_phase,
                                                                 d_phaseCollector,
                                                                 phaseArraySize,
                                                                 phaseCollectorSize);
    
                cudaDeviceSynchronize();
    
                // Kernel call to reduce walker final energies
                PFG_reduce<<<blocksPerKernel/2, 
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
                    h_globalPhase[point] += h_phaseCollector[idx];
                }
                for(uint idx = 0; idx < energyCollectorSize; idx++)
                {
                    h_globalEnergy[point] += h_energyCollector[idx];
                }
            }
            else
            {
                cudaMemcpy(h_phase, d_phase, phaseArraySize * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_energy, d_energy, energyArraySize * sizeof(double), cudaMemcpyDeviceToHost);
                
                for(uint idx = 0; idx < phaseArraySize; idx++)
                {
                    h_globalPhase[point] += h_phase[idx];
                }
                for(uint idx = 0; idx < energyArraySize; idx++)
                {
                    h_globalEnergy[point] += h_energy[idx];
                }
            }
        }
    }

    // normalize magnitudes
    if(this->LHS.size() > 0) this->LHS.clear();
    this->LHS.reserve(this->gradientPoints);
    for(int point = 0; point < this->gradientPoints; point++)
    {
        this->LHS.push_back((h_globalPhase[point]/h_globalEnergy[point]));
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
    free(h_globalEnergy);
    free(h_globalPhase);

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
    h_globalEnergy = NULL;
    h_globalPhase = NULL;

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
    // cout << "pulse width: " << pulseWidth << "ms" << endl;
    // cout << "Gradient: " << giromagneticRatio << " x 2pi x 1e6 /Ts" << endl;
    // cout << "final signal = " << h_globalEnergy << endl;
    // cout << "final phase = " << h_globalPhase << endl;
    // cout << "phase / signal = " << h_globalPhase / h_globalEnergy << endl;

    cudaDeviceReset();
}

// function to call GPU kernel to execute
// walker's "walk" method in Graphics Processing Unit
void NMR_PFGSE::simulation_cuda()
{
    cout << "initializing RW-PFG NMR simulation in GPU... ";

    double gradientMagnitude = 0;
    double pulseWidth = this->pulseWidth;
    double giromagneticRatio = this->giromagneticRatio;
    if(!PFGSE_USE_TWOPI) giromagneticRatio /= TWO_PI;
    uint gradientPoints = this->gradientPoints;


    // reset walker's initial state with omp parallel for
    if(NMR_OPENMP)
    {
        // set omp variables for parallel loop throughout walker list
        const int num_cpu_threads = omp_get_max_threads();
        const int loop_size = this->NMR.walkers.size();
        int loop_start, loop_finish;

        #pragma omp parallel private(loop_start, loop_finish) 
        {
            const int thread_id = omp_get_thread_num();
            OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
            loop_start = looper.getStart();
            loop_finish = looper.getFinish(); 

            for (uint id = loop_start; id < loop_finish; id++)
            {
                this->NMR.walkers[id].resetPosition();
                this->NMR.walkers[id].resetSeed();
                this->NMR.walkers[id].resetEnergy();
            }
        }
    } else
    {
        for (uint id = 0; id < this->NMR.walkers.size(); id++)
        {
            this->NMR.walkers[id].resetPosition();
            this->NMR.walkers[id].resetSeed();
            this->NMR.walkers[id].resetEnergy();
        }
    }

    // reset vector to store energy decay
    this->NMR.resetGlobalEnergy();
    this->NMR.globalEnergy.reserve(this->NMR.numberOfEchoes + 1); // '+1' to accomodate time 0.0

    // get initial energy global state
    double energySum = ((double) this->NMR.walkers.size()) * this->NMR.walkers[0].getEnergy();
    this->NMR.globalEnergy.push_back(energySum);

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values for sizing issues
    uint bitBlockColumns = this->NMR.bitBlock.blockColumns;
    uint bitBlockRows = this->NMR.bitBlock.blockRows;
    uint numberOfBitBlocks = this->NMR.bitBlock.numberOfBlocks;
    uint numberOfWalkers = this->NMR.numberOfWalkers;
    uint map_columns = this->NMR.bitBlock.imageColumns;
    uint map_rows = this->NMR.bitBlock.imageRows;
    uint map_depth = this->NMR.bitBlock.imageDepth;
    uint shiftConverter = log2(this->NMR.voxelDivision);
    double voxelResolution = this->NMR.imageVoxelResolution;
    uint numberOfSteps = this->NMR.simulationSteps;

    // create a steps bucket
    uint stepsLimit = MAX_RWSTEPS;
    uint stepsSize = numberOfSteps/stepsLimit;
    vector<uint> steps;
    for(uint idx = 0; idx < stepsSize; idx++)
    {
        steps.push_back(stepsLimit);
    }
    // charge rebalance
    if((numberOfSteps % stepsLimit) > 0)
    {
        stepsSize++;
        steps.push_back(numberOfSteps%stepsLimit);
    } 
    

    // number of echos that each walker in kernel call will perform
    // uint echoesPerKernel = ECHOESPERKERNEL;
    // uint kernelCalls = (uint)ceil((double)numberOfEchoes / (double)echoesPerKernel);

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    uint threadsPerBlock = THREADSPERBLOCK;
    uint blocksPerKernel = BLOCKS;
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
    h_bitBlock = this->NMR.bitBlock.blocks;

    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, h_bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Host and Device memory data allocation
    // pointers used in host array conversion
    int *h_walker_x0 = setIntArray_PFG(walkersPerKernel);
    int *h_walker_y0 = setIntArray_PFG(walkersPerKernel);
    int *h_walker_z0 = setIntArray_PFG(walkersPerKernel);
    int *h_walker_px = setIntArray_PFG(walkersPerKernel);
    int *h_walker_py = setIntArray_PFG(walkersPerKernel);
    int *h_walker_pz = setIntArray_PFG(walkersPerKernel);
    double *h_penalty = setDoubleArray_PFG(walkersPerKernel);
    uint64_t *h_seed = setUInt64Array_PFG(walkersPerKernel);

    // temporary array to collect energy contributions for each echo in a kernel
    // double *temp_globalEnergy = setDoubleArray_3D((uint)echoesPerKernel);
    // double *h_globalEnergy = setDoubleArray_3D(kernelCalls * echoesPerKernel);
    double *h_energy = setDoubleArray_PFG(energyArraySize);
    double *h_energyCollector = setDoubleArray_PFG(energyCollectorSize);
    double *h_globalEnergy = setDoubleArray_PFG(gradientPoints);

    // magnetization phase
    double *h_phase = setDoubleArray_PFG(phaseArraySize);
    double *h_phaseCollector = setDoubleArray_PFG(phaseCollectorSize);
    double *h_globalPhase = setDoubleArray_PFG(gradientPoints);

    for (uint point = 0; point < gradientPoints; point++)
    {
        h_globalEnergy[point] = 0.0;
        h_globalPhase[point] = 0.0;
    }

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
        if(NMR_OPENMP)
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = walkersPerKernel;
            int loop_start, loop_finish;

            #pragma omp parallel shared(packOffset, h_walker_x0, h_walker_y0, h_walker_z0, h_walker_px, h_walker_py, h_walker_pz, h_penalty, h_seed, h_energy, h_phase) private(loop_start, loop_finish) 
            {
                const int thread_id = omp_get_thread_num();
                OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                loop_start = looper.getStart();
                loop_finish = looper.getFinish(); 

                for (uint id = loop_start; id < loop_finish; id++)
                {
                    h_walker_x0[id] = this->NMR.walkers[id + packOffset].initialPosition.x;
                    h_walker_y0[id] = this->NMR.walkers[id + packOffset].initialPosition.y;
                    h_walker_z0[id] = this->NMR.walkers[id + packOffset].initialPosition.z;
                    h_walker_px[id] = this->NMR.walkers[id + packOffset].initialPosition.x;
                    h_walker_py[id] = this->NMR.walkers[id + packOffset].initialPosition.y;
                    h_walker_pz[id] = this->NMR.walkers[id + packOffset].initialPosition.z;
                    h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                    h_seed[id] = this->NMR.walkers[id + packOffset].initialSeed;
                    h_energy[id] = this->NMR.walkers[id + packOffset].energy;
                    h_phase[id] = this->NMR.walkers[id + packOffset].energy;
                }
            }
        } else
        {
            for (uint id = 0; id < walkersPerKernel; id++)
            {
                h_walker_x0[id] = this->NMR.walkers[id + packOffset].initialPosition.x;
                h_walker_y0[id] = this->NMR.walkers[id + packOffset].initialPosition.y;
                h_walker_z0[id] = this->NMR.walkers[id + packOffset].initialPosition.z;
                h_walker_px[id] = this->NMR.walkers[id + packOffset].initialPosition.x;
                h_walker_py[id] = this->NMR.walkers[id + packOffset].initialPosition.y;
                h_walker_pz[id] = this->NMR.walkers[id + packOffset].initialPosition.z;
                h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                h_seed[id] = this->NMR.walkers[id + packOffset].initialSeed;
                h_energy[id] = this->NMR.walkers[id + packOffset].energy;
                h_phase[id] = this->NMR.walkers[id + packOffset].energy;
            }
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
        for(uint step = 0; step < steps.size(); step++)
        {
            PFG_walk<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
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
                                                           steps[step],
                                                           map_columns,
                                                           map_rows,
                                                           map_depth,
                                                           shiftConverter);
            cudaDeviceSynchronize();  
        }

        // recover last positions
        cudaMemcpy(h_walker_px, d_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_py, d_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_pz, d_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);      
        if(NMR_OPENMP)
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = walkersPerKernel;
            int loop_start, loop_finish;

            #pragma omp parallel shared(h_walker_px, h_walker_py, h_walker_pz, packOffset) private(loop_start, loop_finish) 
            {
                const int thread_id = omp_get_thread_num();
                OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                loop_start = looper.getStart();
                loop_finish = looper.getFinish(); 

                for (uint id = loop_start; id < loop_finish; id++)
                {
                    this->NMR.walkers[id + packOffset].position_x = h_walker_px[id];
                    this->NMR.walkers[id + packOffset].position_y = h_walker_py[id];
                    this->NMR.walkers[id + packOffset].position_z = h_walker_pz[id];
                }
            }
        } else
        {
            for (uint i = 0; i < walkersPerKernel; i++)
            {
                this->NMR.walkers[i + packOffset].position_x = h_walker_px[i];
                this->NMR.walkers[i + packOffset].position_y = h_walker_py[i];
                this->NMR.walkers[i + packOffset].position_z = h_walker_pz[i];            
            }
        }
        

        for(int point = 0; point < gradientPoints; point++)
        {
            double k_X = compute_k(this->vecGradient[point].getX(), giromagneticRatio, pulseWidth);
            double k_Y = compute_k(this->vecGradient[point].getY(), giromagneticRatio, pulseWidth);
            double k_Z = compute_k(this->vecGradient[point].getZ(), giromagneticRatio, pulseWidth);

            // kernel call to compute walkers individual phase
            PFG_measure<<<blocksPerKernel, threadsPerBlock>>>  (d_walker_x0,
                                                                d_walker_y0, 
                                                                d_walker_z0,
                                                                d_walker_px,
                                                                d_walker_py,
                                                                d_walker_pz,
                                                                d_energy,
                                                                d_phase,
                                                                walkersPerKernel,
                                                                voxelResolution,
                                                                k_X,
                                                                k_Y,
                                                                k_Z);
            cudaDeviceSynchronize();

            if(REDUCE_IN_GPU)
            {
                // Kernel call to reduce walker final phases
                PFG_reduce<<<blocksPerKernel/2, 
                             threadsPerBlock, 
                             threadsPerBlock * sizeof(double)>>>(d_phase,
                                                                 d_phaseCollector,  
                                                                 phaseArraySize,
                                                                 phaseCollectorSize);       
                cudaDeviceSynchronize();

                // Kernel call to reduce walker final energies
                PFG_reduce<<<blocksPerKernel/2, 
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
                    h_globalPhase[point] += h_phaseCollector[idx];
                }
                for(uint idx = 0; idx < energyCollectorSize; idx++)
                {
                    h_globalEnergy[point] += h_energyCollector[idx];
                }
            } 
            else
            {    

                cudaMemcpy(h_phase, d_phase, phaseArraySize * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_energy, d_energy, energyArraySize * sizeof(double), cudaMemcpyDeviceToHost);            
                for(uint idx = 0; idx < phaseArraySize; idx++)
                {
                    h_globalPhase[point] += h_phase[idx];
                }
                for(uint idx = 0; idx < energyArraySize; idx++)
                {
                    h_globalEnergy[point] += h_energy[idx];
                } 
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
        if(NMR_OPENMP)
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = lastWalkerPackSize;
            int loop_start, loop_finish;

            #pragma omp parallel shared(packOffset, h_walker_x0, h_walker_y0, h_walker_z0, h_walker_px, h_walker_py, h_walker_pz, h_penalty, h_seed, h_energy, h_phase) private(loop_start, loop_finish) 
            {
                const int thread_id = omp_get_thread_num();
                OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                loop_start = looper.getStart();
                loop_finish = looper.getFinish(); 

                for (uint id = loop_start; id < loop_finish; id++)
                {
                    h_walker_x0[id] = this->NMR.walkers[id + packOffset].initialPosition.x;
                    h_walker_y0[id] = this->NMR.walkers[id + packOffset].initialPosition.y;
                    h_walker_z0[id] = this->NMR.walkers[id + packOffset].initialPosition.z;
                    h_walker_px[id] = this->NMR.walkers[id + packOffset].initialPosition.x;
                    h_walker_py[id] = this->NMR.walkers[id + packOffset].initialPosition.y;
                    h_walker_pz[id] = this->NMR.walkers[id + packOffset].initialPosition.z;
                    h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                    h_seed[id] = this->NMR.walkers[id + packOffset].initialSeed;
                    h_energy[id] = this->NMR.walkers[id + packOffset].energy;
                    h_phase[id] = this->NMR.walkers[id + packOffset].energy;
                }
            }
        } else
        {
            for (uint id = 0; id < lastWalkerPackSize; id++)
            {
                h_walker_x0[id] = this->NMR.walkers[id + packOffset].initialPosition.x;
                h_walker_y0[id] = this->NMR.walkers[id + packOffset].initialPosition.y;
                h_walker_z0[id] = this->NMR.walkers[id + packOffset].initialPosition.z;
                h_walker_px[id] = this->NMR.walkers[id + packOffset].initialPosition.x;
                h_walker_py[id] = this->NMR.walkers[id + packOffset].initialPosition.y;
                h_walker_pz[id] = this->NMR.walkers[id + packOffset].initialPosition.z;
                h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                h_seed[id] = this->NMR.walkers[id + packOffset].initialSeed;
                h_energy[id] = this->NMR.walkers[id + packOffset].energy;
                h_phase[id] = this->NMR.walkers[id + packOffset].energy;
            }
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
        for(uint step = 0; step < steps.size(); step++)
        {
            PFG_walk<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
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
                                                           steps[step],
                                                           map_columns,
                                                           map_rows,
                                                           map_depth,
                                                           shiftConverter);
            cudaDeviceSynchronize();  
        }

        // recover last positions
        cudaMemcpy(h_walker_px, d_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_py, d_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_pz, d_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        if(NMR_OPENMP)
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = lastWalkerPackSize;
            int loop_start, loop_finish;

            #pragma omp parallel shared(h_walker_px, h_walker_py, h_walker_pz, packOffset) private(loop_start, loop_finish) 
            {
                const int thread_id = omp_get_thread_num();
                OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                loop_start = looper.getStart();
                loop_finish = looper.getFinish(); 

                for (uint id = loop_start; id < loop_finish; id++)
                {
                    this->NMR.walkers[id + packOffset].position_x = h_walker_px[id];
                    this->NMR.walkers[id + packOffset].position_y = h_walker_py[id];
                    this->NMR.walkers[id + packOffset].position_z = h_walker_pz[id];
                }
            }
        } else
        {
            for (uint id = 0; id < lastWalkerPackSize; id++)
            {
                this->NMR.walkers[id + packOffset].position_x = h_walker_px[id];
                this->NMR.walkers[id + packOffset].position_y = h_walker_py[id];
                this->NMR.walkers[id + packOffset].position_z = h_walker_pz[id];            
            }
        }

       

        for(int point = 0; point < this->gradientPoints; point++)
        {
            double k_X = compute_k(this->vecGradient[point].getX(), giromagneticRatio, pulseWidth);
            double k_Y = compute_k(this->vecGradient[point].getY(), giromagneticRatio, pulseWidth);
            double k_Z = compute_k(this->vecGradient[point].getZ(), giromagneticRatio, pulseWidth);

            // kernel call to compute walkers individual phase
            PFG_measure<<<blocksPerKernel, threadsPerBlock>>>  (d_walker_x0,
                                                                d_walker_y0, 
                                                                d_walker_z0,
                                                                d_walker_px,
                                                                d_walker_py,
                                                                d_walker_pz,
                                                                d_energy,
                                                                d_phase,
                                                                lastWalkerPackSize,
                                                                voxelResolution,
                                                                k_X,
                                                                k_Y,
                                                                k_Z);

            cudaDeviceSynchronize();

            if(REDUCE_IN_GPU)
            {
                // Kernel call to reduce walker final phases
                PFG_reduce<<<blocksPerKernel/2, 
                             threadsPerBlock, 
                             threadsPerBlock * sizeof(double)>>>(d_phase,
                                                                 d_phaseCollector,
                                                                 phaseArraySize,
                                                                 phaseCollectorSize);
    
                cudaDeviceSynchronize();
    
                // Kernel call to reduce walker final energies
                PFG_reduce<<<blocksPerKernel/2, 
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
                    h_globalPhase[point] += h_phaseCollector[idx];
                }
                for(uint idx = 0; idx < energyCollectorSize; idx++)
                {
                    h_globalEnergy[point] += h_energyCollector[idx];
                }
            }
            else
            {
                cudaMemcpy(h_phase, d_phase, phaseArraySize * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_energy, d_energy, energyArraySize * sizeof(double), cudaMemcpyDeviceToHost);
                
                for(uint idx = 0; idx < phaseArraySize; idx++)
                {
                    h_globalPhase[point] += h_phase[idx];
                }
                for(uint idx = 0; idx < energyArraySize; idx++)
                {
                    h_globalEnergy[point] += h_energy[idx];
                }
            }
        }
    }

    // collect energy data
    for (uint echo = 0; echo < this->NMR.numberOfEchoes; echo++)
    {
        this->NMR.globalEnergy.push_back(h_globalPhase[echo]);
    }

    // normalize magnitudes
    if(this->LHS.size() > 0) this->LHS.clear();
    this->LHS.reserve(this->gradientPoints);
    for(int point = 0; point < this->gradientPoints; point++)
    {
        this->LHS.push_back((h_globalPhase[point]/h_globalEnergy[point]));
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
    free(h_globalEnergy);
    free(h_globalPhase);

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
    h_globalEnergy = NULL;
    h_globalPhase = NULL;

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
    // cout << "pulse width: " << pulseWidth << "ms" << endl;
    // cout << "Gradient: " << giromagneticRatio << " x 2pi x 1e6 /Ts" << endl;
    // cout << "final signal = " << h_globalEnergy << endl;
    // cout << "final phase = " << h_globalPhase << endl;
    // cout << "phase / signal = " << h_globalPhase / h_globalEnergy << endl;

    cudaDeviceReset();
}


/////////////////////////////////////////////////////////////////////
////////////////////////// HOST FUNCTIONS ///////////////////////////
/////////////////////////////////////////////////////////////////////
int *setIntArray_PFG(uint size)
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

uint *setUIntArray_PFG(uint size)
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

double *setDoubleArray_PFG(uint size)
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

uint64_t *setUInt64Array_PFG(uint size)
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

void copyVectorBtoA_PFG(int a[], int b[], uint size)
{
    for (uint i = 0; i < size; i++)
    {
        a[i] = b[i];
    }
}

void copyVectorBtoA_PFG(double a[], double b[], int size)
{
    for (uint i = 0; i < size; i++)
    {
        a[i] = b[i];
    }
}

void copyVectorBtoA_PFG(uint64_t a[], uint64_t b[], int size)
{
    for (uint i = 0; i < size; i++)
    {
        a[i] = b[i];
    }
}

void vectorElementSwap_PFG(int *vector, uint index1, uint index2)
{
    int temp;

    temp = vector[index1];
    vector[index1] = vector[index2];
    vector[index2] = temp;
}

void vectorElementSwap_PFG(double *vector, uint index1, uint index2)
{
    double temp;

    temp = vector[index1];
    vector[index1] = vector[index2];
    vector[index2] = temp;
}

void vectorElementSwap_PFG(uint64_t *vector, uint index1, uint index2)
{
    uint64_t temp;

    temp = vector[index1];
    vector[index1] = vector[index2];
    vector[index2] = temp;
}

/////////////////////////////////////////////////////////////////////
//////////////////////// DEVICE FUNCTIONS ///////////////////////////
/////////////////////////////////////////////////////////////////////

__device__ direction computeNextDirection_PFG(uint64_t &seed)
{
    // generate random number using xorshift algorithm
    xorshift64_state xor_state;
    xor_state.a = seed;
    seed = xorShift64_PFG(&xor_state);
    uint64_t rand = seed;

    // set direction based on the random number
    direction nextDirection = (direction)(mod6_PFG(rand) + 1);
    return nextDirection;
}

__device__ uint64_t xorShift64_PFG(struct xorshift64_state *state)
{
    uint64_t x = state->a;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return state->a = x;
}

__device__ uint64_t mod6_PFG(uint64_t a)
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

__device__ direction checkBorder_PFG(int walker_px,
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

__device__ void computeNextPosition_PFG(int &walker_px,
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

__device__ bool checkNextPosition_PFG(int next_x,
                                     int next_y,
                                     int next_z,
                                     const uint64_t *bitBlock,
                                     const int bitBlockColumns,
                                     const int bitBlockRows)
{
    int blockIndex = findBlockIndex_PFG(next_x, next_y, next_z, bitBlockColumns, bitBlockRows);
    int nextBit = findBitIndex_PFG(next_x, next_y, next_z);
    uint64_t nextBlock = bitBlock[blockIndex];

    return (!checkIfBlockBitIsWall_PFG(nextBlock, nextBit));
};

__device__ int findBlockIndex_PFG(int next_x, int next_y, int next_z, int bitBlockColumns, int bitBlockRows)
{
    // "x >> 2" is like "x / 4" in bitwise operation
    int block_x = next_x >> 2;
    int block_y = next_y >> 2;
    int block_z = next_z >> 2;
    int blockIndex = block_x + block_y * bitBlockColumns + block_z * (bitBlockColumns * bitBlockRows);

    return blockIndex;
}

__device__ int findBitIndex_PFG(int next_x, int next_y, int next_z)
{
    // "x & (n - 1)" is lise "x % n" in bitwise operation
    int bit_x = next_x & (COLUMNSPERBLOCK3D - 1);
    int bit_y = next_y & (ROWSPERBLOCK3D - 1); 
    int bit_z = next_z & (DEPTHPERBLOCK3D - 1);
    // "x << 3" is like "x * 8" in bitwise operation
    int bitIndex = bit_x + (bit_y << 2) + ((bit_z << 2) << 2);

    return bitIndex;
}

__device__ bool checkIfBlockBitIsWall_PFG(uint64_t nextBlock, int nextBit)
{
    return ((nextBlock >> nextBit) & 1ull);
}

__device__ double compute_PFG_k_value(double gradientMagnitude, double pulse_width, double giromagneticRatio)
{
    return (pulse_width * 1.0e-03) * (TWO_PI * giromagneticRatio * 1.0e+06) * (gradientMagnitude * 1.0e-08);
}

__host__ double compute_k(double gradientMagnitude, double pulse_width, double giromagneticRatio)
{
    return (pulse_width * 1.0e-03) * (TWO_PI * giromagneticRatio * 1.0e+06) * (gradientMagnitude * 1.0e-08);
}

__device__ uint convertLocalToGlobal(uint _localPos, uint _shiftConverter)
{
    return (_localPos >> _shiftConverter);
}

__device__ double dotProduct(double vec1X, double vec1Y, double vec1Z, double vec2X, double vec2Y, double vec2Z)
{
    return (vec1X*vec2X + vec1Y*vec2Y + vec1Z*vec2Z);
}