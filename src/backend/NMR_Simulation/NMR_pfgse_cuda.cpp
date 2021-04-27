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
#include "../Utils/myAllocator.h"

//include
#include "NMR_defs.h"
#include "../Walker/walker.h"
#include "../RNG/xorshift.h"
#include "NMR_Simulation.h"
#include "NMR_pfgse.h"
#include "NMR_pfgse_cuda.h"

// GPU kernel for NMR simulation - a.k.a. walker's relaxation/demagnetization
// in this kernel, each thread will behave as a unique walker
__global__ void PFG_map_noflux(  int *walker_px,
                                 int *walker_py,
                                 int *walker_pz,
                                 uint *collisions,
                                 uint64_t *seed,
                                 const uint64_t *bitBlock,
                                 const uint bitBlockColumns,
                                 const uint bitBlockRows,
                                 const uint numberOfWalkers,
                                 const uint numberOfSteps,
                                 const int map_columns,
                                 const int map_rows,
                                 const int map_depth,
                                 const int shift_convert)
{
    // identify thread's walker
    int walkerId = threadIdx.x + blockIdx.x * blockDim.x;

    // Local variables for unique read from device global memory
    int localPosX, localPosY, localPosZ;
    uint localCollisions;
    uint64_t localSeed;

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
        localCollisions = collisions[walkerId];
        localSeed = seed[walkerId];
        
        for(int step = 0; step < numberOfSteps; step++)
        {
            nextDirection = computeNextDirection_PFG(localSeed);
        
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

// GPU kernel for NMR simulation - a.k.a. walker's relaxation/demagnetization
// in this kernel, each thread will behave as a unique walker
__global__ void PFG_map_periodic(int *walker_px,
                                 int *walker_py,
                                 int *walker_pz,
                                 uint *collisions,
                                 uint64_t *seed,
                                 const uint64_t *bitBlock,
                                 const uint bitBlockColumns,
                                 const uint bitBlockRows,
                                 const uint numberOfWalkers,
                                 const uint numberOfSteps,
                                 const int map_columns,
                                 const int map_rows,
                                 const int map_depth,
                                 const int shift_convert)
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

        // printf("\ngpu kernel pfgse_map_periodic() launched");
        // printf("\ninitial: {%d, %d, %d} \n", localPosX, localPosY, localPosZ);
            
        for(int step = 0; step < numberOfSteps; step++)
        {
            
            nextDirection = computeNextDirection_PFG(localSeed);            
        
            computeNextPosition_PFG(localPosX,
                                    localPosY,
                                    localPosZ,
                                    nextDirection,
                                    localNextX,
                                    localNextY,
                                    localNextZ);
            // printf("next: {%d, %d, %d} \t", localNextX, localNextY, localNextZ);

            // update img position
            imgPosX = convertLocalToGlobal(localNextX, shift_convert) % map_columns;
            if(imgPosX < 0) imgPosX += map_columns;

            imgPosY = convertLocalToGlobal(localNextY, shift_convert) % map_rows;
            if(imgPosY < 0) imgPosY += map_rows;

            imgPosZ = convertLocalToGlobal(localNextZ, shift_convert) % map_depth;
            if(imgPosZ < 0) imgPosZ += map_depth;

            // printf("img: {%d, %d, %d} ", imgPosX, imgPosY, imgPosZ);

            if (checkNextPosition_PFG(imgPosX, 
                                      imgPosY, 
                                      imgPosZ, 
                                      bitBlock, 
                                      bitBlockColumns, 
                                      bitBlockRows))
            {
                // update real position
                // printf("Ok! \n");
                localPosX = localNextX;
                localPosY = localNextY;
                localPosZ = localNextZ;                
            }
            else
            {
                // walker hits wall and comes back to the same position
                // collisions count is incremented
                // printf("Not ok, hit wall!\n");
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

// GPU kernel for magnetization level evaluation
__global__ void PFG_evaluate_energy( uint *collisions,
                                     double *penalty,
                                     double *energy,
                                     const uint numberOfWalkers)
{
    // identify thread's walker
    int walkerId = threadIdx.x + blockIdx.x * blockDim.x;

    if(walkerId < numberOfWalkers)
    {
        energy[walkerId] *= pow(penalty[walkerId], collisions[walkerId]);
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

        double local_phase = dotProduct(k_X, k_Y, k_Z, dX, dY, dZ) * voxelResolution; 
        double magnitude_real_value = cos(local_phase);
        // double magnitude_imag_value = sin(local_phase);
    
        // update global value 
        phase[walkerId] = magnitude_real_value * energy[walkerId];
    }
}

// GPU kernel for NMR simulation - a.k.a. walker's relaxation/demagnetization
// in this kernel, each thread will behave as a solitary walker
__global__ void PFG_measure_all_k(int *walker_x0,
                                  int *walker_y0, 
                                  int *walker_z0,
                                  int *walker_xF,
                                  int *walker_yF,
                                  int *walker_zF,
                                  double *energy,
                                  double *phase,
                                  const uint packOffset,
                                  const uint packSize,
                                  const uint numberOfWalkers,
                                  const double voxelResolution,
                                  const double *k_X,
                                  const double *k_Y,
                                  const double *k_Z,
                                  const uint kValues)
{
    // identify thread's walker
    int walkerId = threadIdx.x + blockIdx.x * blockDim.x;
    int gIndex = packOffset + walkerId;
    if(walkerId < packSize)
    {
        for(int kIdx = 0; kIdx < kValues; kIdx++)
        {
            double dX = (walker_xF[walkerId] - walker_x0[walkerId]);
            double dY = (walker_yF[walkerId] - walker_y0[walkerId]);
            double dZ = (walker_zF[walkerId] - walker_z0[walkerId]);
            double local_phase = dotProduct(k_X[kIdx], k_Y[kIdx], k_Z[kIdx], dX, dY, dZ) * voxelResolution; 
            double magnitude_real_value = cos(local_phase);
            
            gIndex += numberOfWalkers*kIdx;        
            phase[gIndex] = magnitude_real_value * energy[walkerId];    
        }        
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
void NMR_PFGSE::simulation_cuda_noflux()
{
    cout << "- starting RW-PFGSE simulation (in GPU) ";

    bool time_verbose = false;
    double copy_time = 0.0;
    double kernel_time = 0.0;
    double buffer_time = 0.0;
    double reduce_time = 0.0;

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
    int map_columns = this->NMR.bitBlock.imageColumns;
    int map_rows = this->NMR.bitBlock.imageRows;
    int map_depth = this->NMR.bitBlock.imageDepth;
    int shiftConverter = log2(this->NMR.voxelDivision);
    double voxelResolution = this->NMR.imageVoxelResolution;
    uint numberOfSteps = this->NMR.simulationSteps - this->stepsTaken;
    this->stepsTaken += numberOfSteps;
    cout << "[" << numberOfSteps << " RW-steps]... ";

    // create a steps bucket
    uint stepsLimit = this->NMR.rwNMR_config.getMaxRWSteps();
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
    

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    uint threadsPerBlock = this->NMR.rwNMR_config.getThreadsPerBlock();
    uint blocksPerKernel = this->NMR.rwNMR_config.getBlocks();
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


    // signal  
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
    myAllocator arrayFactory;
    int *h_walker_x0 = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_y0 = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_z0 = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_px = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_py = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_pz = arrayFactory.getIntArray(walkersPerKernel);
    uint *h_collisions = arrayFactory.getUIntArray(walkersPerKernel);
    double *h_penalty = arrayFactory.getDoubleArray(walkersPerKernel);
    uint64_t *h_seed = arrayFactory.getUInt64Array(walkersPerKernel);

    // magnetization and phase
    double *h_energy = arrayFactory.getDoubleArray(walkersPerKernel);
    double *h_energyCollector = arrayFactory.getDoubleArray(energyCollectorSize);
    double h_globalEnergy = 0.0;
    double *h_phase = arrayFactory.getDoubleArray(walkersPerKernel);
    double *h_phaseCollector = arrayFactory.getDoubleArray(phaseCollectorSize);
    double *h_globalPhase = arrayFactory.getDoubleArray(gradientPoints);

    double tick = omp_get_wtime();
    for (uint point = 0; point < gradientPoints; point++)
    {
        h_globalPhase[point] = 0.0;
    }
    buffer_time += omp_get_wtime() - tick;

    // Declaration of pointers to device data arrays
    int *d_walker_x0;
    int *d_walker_y0;
    int *d_walker_z0;
    int *d_walker_px;
    int *d_walker_py;
    int *d_walker_pz;
    uint *d_collisions;
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
    cudaMalloc((void **)&d_collisions, walkersPerKernel * sizeof(uint));
    cudaMalloc((void **)&d_penalty, walkersPerKernel * sizeof(double));
    cudaMalloc((void **)&d_seed, walkersPerKernel * sizeof(uint64_t));
    cudaMalloc((void **)&d_energy, walkersPerKernel * sizeof(double));
    cudaMalloc((void **)&d_energyCollector, energyCollectorSize * sizeof(double));
    cudaMalloc((void **)&d_phase, walkersPerKernel * sizeof(double));
    cudaMalloc((void **)&d_phaseCollector, phaseCollectorSize * sizeof(double));
    
    tick = omp_get_wtime();
    for (uint idx = 0; idx < walkersPerKernel; idx++)
    {
        h_energy[idx] = 0.0;
    }
    
    for(uint idx = 0; idx < energyCollectorSize; idx++)
    {
        h_energyCollector[idx] = 0.0;
    }

    for (uint idx = 0; idx < walkersPerKernel; idx++)
    {
        h_phase[idx] = 0.0;
    } 

    for(uint idx = 0; idx < phaseCollectorSize; idx++)
    {
        h_phaseCollector[idx] = 0.0;
    }
    buffer_time += omp_get_wtime() - tick;

    // PFG main loop
    for (uint packId = 0; packId < (numberOfWalkerPacks - 1); packId++)
    {
        // set offset in walkers vector
        uint packOffset = packId * walkersPerKernel;

        // Host data copy
        // copy original walkers' data to temporary host arrays
        tick = omp_get_wtime();
        if(this->NMR.rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = walkersPerKernel;
            int loop_start, loop_finish;

            #pragma omp parallel shared(packOffset, h_walker_x0, h_walker_y0, h_walker_z0, h_walker_px, h_walker_py, h_walker_pz, h_collisions, h_penalty, h_seed, h_energy, h_phase) private(loop_start, loop_finish) 
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
                    h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                    h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                    h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
                    h_collisions[id] = 0; // this->NMR.walkers[id + packOffset].collisions; // SERÁ?
                    h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                    h_seed[id] = this->NMR.walkers[id + packOffset].currentSeed;
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
                h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
                h_collisions[id] = 0; //this->NMR.walkers[id + packOffset].collisions;
                h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                h_seed[id] = this->NMR.walkers[id + packOffset].currentSeed;
                h_energy[id] = this->NMR.walkers[id + packOffset].energy;
                h_phase[id] = this->NMR.walkers[id + packOffset].energy;
            }
        }
        buffer_time += omp_get_wtime() - tick;

        // Device data copy
        // copy host data to device
        tick = omp_get_wtime();
        cudaMemcpy(d_walker_x0, h_walker_x0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_y0, h_walker_y0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_z0, h_walker_z0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_px, h_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, h_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, h_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_collisions, h_collisions, walkersPerKernel * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_penalty, h_penalty, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, h_seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, h_energy, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase, h_phase, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        copy_time += omp_get_wtime() - tick;

        // Launch kernel for GPU computation
        // call "walk" method kernel
        tick = omp_get_wtime();
        for(uint step = 0; step < steps.size(); step++)
        {
            PFG_map_noflux<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                                 d_walker_py,
                                                                 d_walker_pz,
                                                                 d_collisions,
                                                                 d_seed,
                                                                 d_bitBlock,
                                                                 bitBlockColumns,
                                                                 bitBlockRows,
                                                                 walkersPerKernel,
                                                                 steps[step],
                                                                 map_columns,
                                                                 map_rows,
                                                                 map_depth,
                                                                 shiftConverter);
            cudaDeviceSynchronize();  
        }

        PFG_evaluate_energy<<<blocksPerKernel, threadsPerBlock>>>(d_collisions,
                                                                  d_penalty, 
                                                                  d_energy,
                                                                  walkersPerKernel);
        cudaDeviceSynchronize();

        kernel_time = omp_get_wtime() - tick;

        // recover last positions
        tick = omp_get_wtime();
        cudaMemcpy(h_walker_px, d_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_py, d_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_pz, d_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);  
        cudaMemcpy(h_collisions, d_collisions, walkersPerKernel * sizeof(uint), cudaMemcpyDeviceToHost);  
        cudaMemcpy(h_energy, d_energy, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);    
        cudaMemcpy(h_seed, d_seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyDeviceToHost);    
        copy_time = omp_get_wtime() - tick;

        tick = omp_get_wtime();
        if(this->NMR.rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = walkersPerKernel;
            int loop_start, loop_finish;

            #pragma omp parallel shared(h_walker_px, h_walker_py, h_walker_pz, h_collisions, h_energy, h_seed, packOffset) private(loop_start, loop_finish) 
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
                    this->NMR.walkers[id + packOffset].collisions += h_collisions[id];
                    this->NMR.walkers[id + packOffset].energy = h_energy[id];
                    this->NMR.walkers[id + packOffset].currentSeed = h_seed[id];
                }
            }
        } else
        {
            for (uint id = 0; id < walkersPerKernel; id++)
            {
                this->NMR.walkers[id + packOffset].position_x = h_walker_px[id];
                this->NMR.walkers[id + packOffset].position_y = h_walker_py[id];
                this->NMR.walkers[id + packOffset].position_z = h_walker_pz[id];            
                this->NMR.walkers[id + packOffset].collisions += h_collisions[id];
                this->NMR.walkers[id + packOffset].energy = h_energy[id];
                this->NMR.walkers[id + packOffset].currentSeed = h_seed[id];
            }
        }
        buffer_time += omp_get_wtime() - tick;

        tick = omp_get_wtime();
        for(int point = 0; point < gradientPoints; point++)
        {
            double k_X = this->vecK[point].getX();
            double k_Y = this->vecK[point].getY();
            double k_Z = this->vecK[point].getZ();
            
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

            if(this->NMR.rwNMR_config.getReduceInGPU())
            {
                // Kernel call to reduce walker final phases
                PFG_reduce<<<blocksPerKernel/2, 
                             threadsPerBlock, 
                             threadsPerBlock * sizeof(double)>>>(d_phase,
                                                                 d_phaseCollector,  
                                                                 walkersPerKernel,
                                                                 phaseCollectorSize);       
                cudaDeviceSynchronize();

                // copy data from gatherer array
                cudaMemcpy(h_phaseCollector, d_phaseCollector, phaseCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);

                // collector reductions
                for(uint idx = 0; idx < phaseCollectorSize; idx++)
                {
                    h_globalPhase[point] += h_phaseCollector[idx];
                }
            } 
            else
            {    

                cudaMemcpy(h_phase, d_phase, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);          
                for(uint idx = 0; idx < walkersPerKernel; idx++)
                {
                    h_globalPhase[point] += h_phase[idx];
                }
            }
        }   

    

        if(this->NMR.rwNMR_config.getReduceInGPU())
        {
            // Kernel call to reduce walker final energies
            PFG_reduce<<<blocksPerKernel/2, 
                            threadsPerBlock, 
                            threadsPerBlock * sizeof(double)>>>(d_energy,
                                                                d_energyCollector,
                                                                walkersPerKernel,
                                                                energyCollectorSize);     
            cudaDeviceSynchronize();

            // copy data from gatherer array
            cudaMemcpy(h_energyCollector, d_energyCollector, energyCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);

            // collector reductions
            for(uint idx = 0; idx < energyCollectorSize; idx++)
            {
                h_globalEnergy += h_energyCollector[idx];
            }
        } 
        else
        {    
            cudaMemcpy(h_energy, d_energy, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);            
            for(uint idx = 0; idx < walkersPerKernel; idx++)
            {
                h_globalEnergy += h_energy[idx];
            } 
        }
    } 
    reduce_time += omp_get_wtime() - tick;

    if (lastWalkerPackSize > 0)
    {
        // last Walker pack is done explicitly
        // set offset in walkers vector
        uint packOffset = (numberOfWalkerPacks - 1) * walkersPerKernel;

        // Host data copy
        // copy original walkers' data to temporary host arrays
        tick = omp_get_wtime();
        if(this->NMR.rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = lastWalkerPackSize;
            int loop_start, loop_finish;

            #pragma omp parallel shared(packOffset, h_walker_x0, h_walker_y0, h_walker_z0, h_walker_px, h_walker_py, h_walker_pz, h_collisions, h_penalty, h_seed, h_energy, h_phase) private(loop_start, loop_finish) 
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
                    h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                    h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                    h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
                    h_collisions[id] = 0; // this->NMR.walkers[id + packOffset].collisions;
                    h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                    h_seed[id] = this->NMR.walkers[id + packOffset].currentSeed;
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
                h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
                h_collisions[id] = 0; // this->NMR.walkers[id + packOffset].collisions;
                h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                h_seed[id] = this->NMR.walkers[id + packOffset].currentSeed;
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
        buffer_time += omp_get_wtime() - tick;

        // Device data copy
        // copy host data to device
        tick = omp_get_wtime();
        cudaMemcpy(d_walker_x0, h_walker_x0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_y0, h_walker_y0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_z0, h_walker_z0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_px, h_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, h_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, h_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_collisions, h_collisions, lastWalkerPackSize * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_penalty, h_penalty, lastWalkerPackSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, h_seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, h_energy, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase, h_phase, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        copy_time += omp_get_wtime() - tick;

        // Launch kernel for GPU computation
        // call "walk" method kernel

        tick = omp_get_wtime();
        for(uint step = 0; step < steps.size(); step++)
        {
            PFG_map_noflux<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                                 d_walker_py,
                                                                 d_walker_pz,
                                                                 d_collisions,
                                                                 d_seed,
                                                                 d_bitBlock,
                                                                 bitBlockColumns,
                                                                 bitBlockRows,
                                                                 lastWalkerPackSize,
                                                                 steps[step],
                                                                 map_columns,
                                                                 map_rows,
                                                                 map_depth,
                                                                 shiftConverter);
            cudaDeviceSynchronize();  
        }

        PFG_evaluate_energy<<<blocksPerKernel, threadsPerBlock>>> (d_collisions,
                                                                   d_penalty,
                                                                   d_energy,
                                                                   lastWalkerPackSize);
        cudaDeviceSynchronize();
        kernel_time += omp_get_wtime() - tick;

        // recover last positions
        tick = omp_get_wtime();
        cudaMemcpy(h_walker_px, d_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_py, d_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_pz, d_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_collisions, d_collisions, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_energy, d_energy, lastWalkerPackSize * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_seed, d_seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        copy_time += omp_get_wtime() - tick;

        tick = omp_get_wtime();
        if(this->NMR.rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = lastWalkerPackSize;
            int loop_start, loop_finish;

            #pragma omp parallel shared(h_walker_px, h_walker_py, h_walker_pz, h_collisions, h_energy, h_seed, packOffset) private(loop_start, loop_finish) 
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
                    this->NMR.walkers[id + packOffset].collisions += h_collisions[id];
                    this->NMR.walkers[id + packOffset].energy = h_energy[id];
                    this->NMR.walkers[id + packOffset].currentSeed = h_seed[id];
                }
            }
        } else
        {
            for (uint id = 0; id < lastWalkerPackSize; id++)
            {
                this->NMR.walkers[id + packOffset].position_x = h_walker_px[id];
                this->NMR.walkers[id + packOffset].position_y = h_walker_py[id];
                this->NMR.walkers[id + packOffset].position_z = h_walker_pz[id];
                this->NMR.walkers[id + packOffset].collisions += h_collisions[id];
                this->NMR.walkers[id + packOffset].energy = h_energy[id];    
                this->NMR.walkers[id + packOffset].currentSeed = h_seed[id];        
            }
        }
        buffer_time += omp_get_wtime() - tick;
       
        tick = omp_get_wtime();
        for(int point = 0; point < this->gradientPoints; point++)
        {
            double k_X = this->vecK[point].getX();
            double k_Y = this->vecK[point].getY();
            double k_Z = this->vecK[point].getZ();

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

            if(this->NMR.rwNMR_config.getReduceInGPU())
            {
                // Kernel call to reduce walker final phases
                PFG_reduce<<<blocksPerKernel/2, 
                             threadsPerBlock, 
                             threadsPerBlock * sizeof(double)>>>(d_phase,
                                                                 d_phaseCollector,
                                                                 walkersPerKernel,
                                                                 phaseCollectorSize);
    
                cudaDeviceSynchronize();    
    
                // copy data from gatherer array
                cudaMemcpy(h_phaseCollector, d_phaseCollector, phaseCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);
                
                // collector reductions
                for(uint idx = 0; idx < phaseCollectorSize; idx++)
                {
                    h_globalPhase[point] += h_phaseCollector[idx];
                }                
            }
            else
            {
                cudaMemcpy(h_phase, d_phase, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);
                
                for(uint idx = 0; idx < walkersPerKernel; idx++)
                {
                    h_globalPhase[point] += h_phase[idx];
                }
            }
        }

        if(this->NMR.rwNMR_config.getReduceInGPU())
        {
            // Kernel call to reduce walker final energies
            PFG_reduce<<<blocksPerKernel/2, 
                            threadsPerBlock, 
                            threadsPerBlock * sizeof(double)>>>(d_energy,
                                                                d_energyCollector,
                                                                walkersPerKernel,
                                                                energyCollectorSize);

            cudaDeviceSynchronize();

            // copy data from gatherer array
            cudaMemcpy(h_energyCollector, d_energyCollector, energyCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);    
            // collector reductions
            for(uint idx = 0; idx < energyCollectorSize; idx++)
            {
                h_globalEnergy += h_energyCollector[idx];
            }
        }
        else
        {
            cudaMemcpy(h_energy, d_energy, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);                
            for(uint idx = 0; idx < walkersPerKernel; idx++)
            {
                h_globalEnergy += h_energy[idx];
            }
        }
        reduce_time += omp_get_wtime() - tick;

    }

    // collect energy data -- REVISE!!!!
    for (uint point = 0; point < gradientPoints; point++)
    {
        this->NMR.globalEnergy.push_back(h_globalEnergy);
    }

    // get magnitudes M(k,t) - new
    if(this->Mkt.size() > 0) this->Mkt.clear();
    this->Mkt.reserve(this->gradientPoints);
    for(int point = 0; point < this->gradientPoints; point++)
    {
        this->Mkt.push_back((h_globalPhase[point]/h_globalEnergy));
    }

    // free pointers in host
    free(h_walker_x0);
    free(h_walker_y0);
    free(h_walker_z0);
    free(h_walker_px);
    free(h_walker_py);
    free(h_walker_pz);
    free(h_collisions);
    free(h_penalty);
    free(h_seed);
    free(h_energy);
    free(h_energyCollector);
    free(h_phase);
    free(h_phaseCollector);
    free(h_globalPhase);

    // and direct them to NULL
    h_walker_px = NULL;
    h_walker_py = NULL;
    h_walker_pz = NULL;
    h_collisions = NULL;
    h_penalty = NULL;
    h_energy = NULL;
    h_energyCollector = NULL;
    h_seed = NULL;
    h_phase = NULL;
    h_phaseCollector = NULL;
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
    cudaFree(d_collisions);
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
    cout << "Done.\nCpu/Gpu elapsed time: " << elapsedTime * 1.0e-3 << " s" << endl;
    cudaDeviceReset();

    if(time_verbose)
    {
        cout << "--- Time analysis ---" << endl;
        cout << "cpu data buffer: " << buffer_time << " s" << endl;
        cout << "gpu data copy: " << copy_time << " s" << endl;
        cout << "gpu kernel launch: " << kernel_time << " s" << endl;
        cout << "gpu reduce launch: " << reduce_time << " s" << endl;
        cout << "---------------------" << endl;
    }
}

// function to call GPU kernel to execute
// walker's "walk" method in Graphics Processing Unit
void NMR_PFGSE::simulation_cuda_periodic()
{
    cout << "- starting RW-PFGSE simulation (in GPU) ";

    bool time_verbose = true;
    double copy_time = 0.0;
    double kernel_time = 0.0;
    double buffer_time = 0.0;
    double reduce_time = 0.0;

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
    int map_columns = this->NMR.bitBlock.imageColumns;
    int map_rows = this->NMR.bitBlock.imageRows;
    int map_depth = this->NMR.bitBlock.imageDepth;
    int shiftConverter = log2(this->NMR.voxelDivision);
    double voxelResolution = this->NMR.imageVoxelResolution;
    uint numberOfSteps = this->NMR.simulationSteps - this->stepsTaken;
    this->stepsTaken += numberOfSteps;
    cout << "[" << numberOfSteps << " RW-steps]... ";

    // create a steps bucket
    uint stepsLimit = this->NMR.rwNMR_config.getMaxRWSteps();
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
    

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    uint threadsPerBlock = this->NMR.rwNMR_config.getThreadsPerBlock();
    uint blocksPerKernel = this->NMR.rwNMR_config.getBlocks();
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


    // signal  
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
    myAllocator arrayFactory;
    int *h_walker_x0 = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_y0 = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_z0 = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_px = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_py = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_pz = arrayFactory.getIntArray(walkersPerKernel);
    uint *h_collisions = arrayFactory.getUIntArray(walkersPerKernel);
    double *h_penalty = arrayFactory.getDoubleArray(walkersPerKernel);
    uint64_t *h_seed = arrayFactory.getUInt64Array(walkersPerKernel);

    // magnetization and phase
    double *h_energy = arrayFactory.getDoubleArray(walkersPerKernel);
    double *h_energyCollector = arrayFactory.getDoubleArray(energyCollectorSize);
    double h_globalEnergy = 0.0;
    double *h_phase = arrayFactory.getDoubleArray(walkersPerKernel);
    double *h_phaseCollector = arrayFactory.getDoubleArray(phaseCollectorSize);
    double *h_globalPhase = arrayFactory.getDoubleArray(gradientPoints);

    double tick = omp_get_wtime();
    for (uint point = 0; point < gradientPoints; point++)
    {
        h_globalPhase[point] = 0.0;
    }
    buffer_time += omp_get_wtime() - tick;

    // Declaration of pointers to device data arrays
    int *d_walker_x0;
    int *d_walker_y0;
    int *d_walker_z0;
    int *d_walker_px;
    int *d_walker_py;
    int *d_walker_pz;
    uint *d_collisions;
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
    cudaMalloc((void **)&d_collisions, walkersPerKernel * sizeof(uint));
    cudaMalloc((void **)&d_penalty, walkersPerKernel * sizeof(double));
    cudaMalloc((void **)&d_seed, walkersPerKernel * sizeof(uint64_t));
    cudaMalloc((void **)&d_energy, walkersPerKernel * sizeof(double));
    cudaMalloc((void **)&d_energyCollector, energyCollectorSize * sizeof(double));
    cudaMalloc((void **)&d_phase, walkersPerKernel * sizeof(double));
    cudaMalloc((void **)&d_phaseCollector, phaseCollectorSize * sizeof(double));
    
    tick = omp_get_wtime();
    for (uint idx = 0; idx < walkersPerKernel; idx++)
    {
        h_energy[idx] = 0.0;
    }
    
    for(uint idx = 0; idx < energyCollectorSize; idx++)
    {
        h_energyCollector[idx] = 0.0;
    }

    for (uint idx = 0; idx < walkersPerKernel; idx++)
    {
        h_phase[idx] = 0.0;
    } 

    for(uint idx = 0; idx < phaseCollectorSize; idx++)
    {
        h_phaseCollector[idx] = 0.0;
    }
    buffer_time += omp_get_wtime() - tick;

    // PFG main loop
    for (uint packId = 0; packId < (numberOfWalkerPacks - 1); packId++)
    {
        // set offset in walkers vector
        uint packOffset = packId * walkersPerKernel;

        // Host data copy
        // copy original walkers' data to temporary host arrays
        tick = omp_get_wtime();
        if(this->NMR.rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = walkersPerKernel;
            int loop_start, loop_finish;

            #pragma omp parallel shared(packOffset, h_walker_x0, h_walker_y0, h_walker_z0, h_walker_px, h_walker_py, h_walker_pz, h_collisions, h_penalty, h_seed, h_energy, h_phase) private(loop_start, loop_finish) 
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
                    h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                    h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                    h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
                    h_collisions[id] = 0; // this->NMR.walkers[id + packOffset].collisions; // SERÁ?
                    h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                    h_seed[id] = this->NMR.walkers[id + packOffset].currentSeed;
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
                h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
                h_collisions[id] = 0; //this->NMR.walkers[id + packOffset].collisions;
                h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                h_seed[id] = this->NMR.walkers[id + packOffset].currentSeed;
                h_energy[id] = this->NMR.walkers[id + packOffset].energy;
                h_phase[id] = this->NMR.walkers[id + packOffset].energy;
            }
        }
        buffer_time += omp_get_wtime() - tick;

        // Device data copy
        // copy host data to device
        tick = omp_get_wtime();
        cudaMemcpy(d_walker_x0, h_walker_x0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_y0, h_walker_y0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_z0, h_walker_z0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_px, h_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, h_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, h_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_collisions, h_collisions, walkersPerKernel * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_penalty, h_penalty, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, h_seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, h_energy, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase, h_phase, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        copy_time += omp_get_wtime() - tick;

        // Launch kernel for GPU computation
        // call "walk" method kernel
        tick = omp_get_wtime();
        for(uint step = 0; step < steps.size(); step++)
        {
            PFG_map_periodic<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                                 d_walker_py,
                                                                 d_walker_pz,
                                                                 d_collisions,
                                                                 d_seed,
                                                                 d_bitBlock,
                                                                 bitBlockColumns,
                                                                 bitBlockRows,
                                                                 walkersPerKernel,
                                                                 steps[step],
                                                                 map_columns,
                                                                 map_rows,
                                                                 map_depth,
                                                                 shiftConverter);
            cudaDeviceSynchronize();  
        }

        PFG_evaluate_energy<<<blocksPerKernel, threadsPerBlock>>>(d_collisions,
                                                                  d_penalty, 
                                                                  d_energy,
                                                                  walkersPerKernel);
        cudaDeviceSynchronize();

        kernel_time = omp_get_wtime() - tick;

        // recover last positions
        tick = omp_get_wtime();
        cudaMemcpy(h_walker_px, d_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_py, d_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_pz, d_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);  
        cudaMemcpy(h_collisions, d_collisions, walkersPerKernel * sizeof(uint), cudaMemcpyDeviceToHost);  
        cudaMemcpy(h_energy, d_energy, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);    
        cudaMemcpy(h_seed, d_seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyDeviceToHost);    
        copy_time = omp_get_wtime() - tick;

        tick = omp_get_wtime();
        if(this->NMR.rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = walkersPerKernel;
            int loop_start, loop_finish;

            #pragma omp parallel shared(h_walker_px, h_walker_py, h_walker_pz, h_collisions, h_energy, h_seed, packOffset) private(loop_start, loop_finish) 
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
                    this->NMR.walkers[id + packOffset].collisions += h_collisions[id];
                    this->NMR.walkers[id + packOffset].energy = h_energy[id];
                    this->NMR.walkers[id + packOffset].currentSeed = h_seed[id];
                }
            }
        } else
        {
            for (uint id = 0; id < walkersPerKernel; id++)
            {
                this->NMR.walkers[id + packOffset].position_x = h_walker_px[id];
                this->NMR.walkers[id + packOffset].position_y = h_walker_py[id];
                this->NMR.walkers[id + packOffset].position_z = h_walker_pz[id];            
                this->NMR.walkers[id + packOffset].collisions += h_collisions[id];
                this->NMR.walkers[id + packOffset].energy = h_energy[id];
                this->NMR.walkers[id + packOffset].currentSeed = h_seed[id];
            }
        }
        buffer_time += omp_get_wtime() - tick;

        tick = omp_get_wtime();
        for(int point = 0; point < gradientPoints; point++)
        {
            double k_X = this->vecK[point].getX();
            double k_Y = this->vecK[point].getY();
            double k_Z = this->vecK[point].getZ();
            
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

            if(this->NMR.rwNMR_config.getReduceInGPU())
            {
                // Kernel call to reduce walker final phases
                PFG_reduce<<<blocksPerKernel/2, 
                             threadsPerBlock, 
                             threadsPerBlock * sizeof(double)>>>(d_phase,
                                                                 d_phaseCollector,  
                                                                 walkersPerKernel,
                                                                 phaseCollectorSize);       
                cudaDeviceSynchronize();

                // copy data from gatherer array
                cudaMemcpy(h_phaseCollector, d_phaseCollector, phaseCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);

                // collector reductions
                for(uint idx = 0; idx < phaseCollectorSize; idx++)
                {
                    h_globalPhase[point] += h_phaseCollector[idx];
                }
            } 
            else
            {    

                cudaMemcpy(h_phase, d_phase, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);          
                for(uint idx = 0; idx < walkersPerKernel; idx++)
                {
                    h_globalPhase[point] += h_phase[idx];
                }
            }
        }   

    

        if(this->NMR.rwNMR_config.getReduceInGPU())
        {
            // Kernel call to reduce walker final energies
            PFG_reduce<<<blocksPerKernel/2, 
                            threadsPerBlock, 
                            threadsPerBlock * sizeof(double)>>>(d_energy,
                                                                d_energyCollector,
                                                                walkersPerKernel,
                                                                energyCollectorSize);     
            cudaDeviceSynchronize();

            // copy data from gatherer array
            cudaMemcpy(h_energyCollector, d_energyCollector, energyCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);

            // collector reductions
            for(uint idx = 0; idx < energyCollectorSize; idx++)
            {
                h_globalEnergy += h_energyCollector[idx];
            }
        } 
        else
        {    
            cudaMemcpy(h_energy, d_energy, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);            
            for(uint idx = 0; idx < walkersPerKernel; idx++)
            {
                h_globalEnergy += h_energy[idx];
            } 
        }
    } 
    reduce_time += omp_get_wtime() - tick;

    if (lastWalkerPackSize > 0)
    {
        // last Walker pack is done explicitly
        // set offset in walkers vector
        uint packOffset = (numberOfWalkerPacks - 1) * walkersPerKernel;

        // Host data copy
        // copy original walkers' data to temporary host arrays
        tick = omp_get_wtime();
        if(this->NMR.rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = lastWalkerPackSize;
            int loop_start, loop_finish;

            #pragma omp parallel shared(packOffset, h_walker_x0, h_walker_y0, h_walker_z0, h_walker_px, h_walker_py, h_walker_pz, h_collisions, h_penalty, h_seed, h_energy, h_phase) private(loop_start, loop_finish) 
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
                    h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                    h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                    h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
                    h_collisions[id] = 0; // this->NMR.walkers[id + packOffset].collisions;
                    h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                    h_seed[id] = this->NMR.walkers[id + packOffset].currentSeed;
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
                h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
                h_collisions[id] = 0; // this->NMR.walkers[id + packOffset].collisions;
                h_penalty[id] = this->NMR.walkers[id + packOffset].decreaseFactor;
                h_seed[id] = this->NMR.walkers[id + packOffset].currentSeed;
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
        buffer_time += omp_get_wtime() - tick;

        // Device data copy
        // copy host data to device
        tick = omp_get_wtime();
        cudaMemcpy(d_walker_x0, h_walker_x0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_y0, h_walker_y0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_z0, h_walker_z0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_px, h_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, h_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, h_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_collisions, h_collisions, lastWalkerPackSize * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_penalty, h_penalty, lastWalkerPackSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, h_seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, h_energy, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase, h_phase, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        copy_time += omp_get_wtime() - tick;

        // Launch kernel for GPU computation
        // call "walk" method kernel

        tick = omp_get_wtime();
        for(uint step = 0; step < steps.size(); step++)
        {
            PFG_map_periodic<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
                                                                 d_walker_py,
                                                                 d_walker_pz,
                                                                 d_collisions,
                                                                 d_seed,
                                                                 d_bitBlock,
                                                                 bitBlockColumns,
                                                                 bitBlockRows,
                                                                 lastWalkerPackSize,
                                                                 steps[step],
                                                                 map_columns,
                                                                 map_rows,
                                                                 map_depth,
                                                                 shiftConverter);
            cudaDeviceSynchronize();  
        }

        PFG_evaluate_energy<<<blocksPerKernel, threadsPerBlock>>> (d_collisions,
                                                                   d_penalty,
                                                                   d_energy,
                                                                   lastWalkerPackSize);
        cudaDeviceSynchronize();
        kernel_time += omp_get_wtime() - tick;

        // recover last positions
        tick = omp_get_wtime();
        cudaMemcpy(h_walker_px, d_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_py, d_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_walker_pz, d_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_collisions, d_collisions, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_energy, d_energy, lastWalkerPackSize * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_seed, d_seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        copy_time += omp_get_wtime() - tick;

        tick = omp_get_wtime();
        if(this->NMR.rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = lastWalkerPackSize;
            int loop_start, loop_finish;

            #pragma omp parallel shared(h_walker_px, h_walker_py, h_walker_pz, h_collisions, h_energy, h_seed, packOffset) private(loop_start, loop_finish) 
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
                    this->NMR.walkers[id + packOffset].collisions += h_collisions[id];
                    this->NMR.walkers[id + packOffset].energy = h_energy[id];
                    this->NMR.walkers[id + packOffset].currentSeed = h_seed[id];
                }
            }
        } else
        {
            for (uint id = 0; id < lastWalkerPackSize; id++)
            {
                this->NMR.walkers[id + packOffset].position_x = h_walker_px[id];
                this->NMR.walkers[id + packOffset].position_y = h_walker_py[id];
                this->NMR.walkers[id + packOffset].position_z = h_walker_pz[id];
                this->NMR.walkers[id + packOffset].collisions += h_collisions[id];
                this->NMR.walkers[id + packOffset].energy = h_energy[id];    
                this->NMR.walkers[id + packOffset].currentSeed = h_seed[id];        
            }
        }
        buffer_time += omp_get_wtime() - tick;
       
        tick = omp_get_wtime();
        for(int point = 0; point < this->gradientPoints; point++)
        {
            double k_X = this->vecK[point].getX();
            double k_Y = this->vecK[point].getY();
            double k_Z = this->vecK[point].getZ();

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

            if(this->NMR.rwNMR_config.getReduceInGPU())
            {
                // Kernel call to reduce walker final phases
                PFG_reduce<<<blocksPerKernel/2, 
                             threadsPerBlock, 
                             threadsPerBlock * sizeof(double)>>>(d_phase,
                                                                 d_phaseCollector,
                                                                 walkersPerKernel,
                                                                 phaseCollectorSize);
    
                cudaDeviceSynchronize();    
    
                // copy data from gatherer array
                cudaMemcpy(h_phaseCollector, d_phaseCollector, phaseCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);
                
                // collector reductions
                for(uint idx = 0; idx < phaseCollectorSize; idx++)
                {
                    h_globalPhase[point] += h_phaseCollector[idx];
                }                
            }
            else
            {
                cudaMemcpy(h_phase, d_phase, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);
                
                for(uint idx = 0; idx < walkersPerKernel; idx++)
                {
                    h_globalPhase[point] += h_phase[idx];
                }
            }
        }

        if(this->NMR.rwNMR_config.getReduceInGPU())
        {
            // Kernel call to reduce walker final energies
            PFG_reduce<<<blocksPerKernel/2, 
                            threadsPerBlock, 
                            threadsPerBlock * sizeof(double)>>>(d_energy,
                                                                d_energyCollector,
                                                                walkersPerKernel,
                                                                energyCollectorSize);

            cudaDeviceSynchronize();

            // copy data from gatherer array
            cudaMemcpy(h_energyCollector, d_energyCollector, energyCollectorSize * sizeof(double), cudaMemcpyDeviceToHost);    
            // collector reductions
            for(uint idx = 0; idx < energyCollectorSize; idx++)
            {
                h_globalEnergy += h_energyCollector[idx];
            }
        }
        else
        {
            cudaMemcpy(h_energy, d_energy, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);                
            for(uint idx = 0; idx < walkersPerKernel; idx++)
            {
                h_globalEnergy += h_energy[idx];
            }
        }
        reduce_time += omp_get_wtime() - tick;

    }

    // collect energy data -- REVISE!!!!
    for (uint point = 0; point < gradientPoints; point++)
    {
        this->NMR.globalEnergy.push_back(h_globalEnergy);
    }

    // get magnitudes M(k,t) - new
    if(this->Mkt.size() > 0) this->Mkt.clear();
    this->Mkt.reserve(this->gradientPoints);
    for(int point = 0; point < this->gradientPoints; point++)
    {
        this->Mkt.push_back((h_globalPhase[point]/h_globalEnergy));
    }

    // free pointers in host
    free(h_walker_x0);
    free(h_walker_y0);
    free(h_walker_z0);
    free(h_walker_px);
    free(h_walker_py);
    free(h_walker_pz);
    free(h_collisions);
    free(h_penalty);
    free(h_seed);
    free(h_energy);
    free(h_energyCollector);
    free(h_phase);
    free(h_phaseCollector);
    free(h_globalPhase);

    // and direct them to NULL
    h_walker_px = NULL;
    h_walker_py = NULL;
    h_walker_pz = NULL;
    h_collisions = NULL;
    h_penalty = NULL;
    h_energy = NULL;
    h_energyCollector = NULL;
    h_seed = NULL;
    h_phase = NULL;
    h_phaseCollector = NULL;
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
    cudaFree(d_collisions);
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
    cout << "Done.\nCpu/Gpu elapsed time: " << elapsedTime * 1.0e-3 << " s" << endl;
    cudaDeviceReset();

    if(time_verbose)
    {
        cout << "--- Time analysis ---" << endl;
        cout << "cpu data buffer: " << buffer_time << " s" << endl;
        cout << "gpu data copy: " << copy_time << " s" << endl;
        cout << "gpu kernel launch: " << kernel_time << " s" << endl;
        cout << "gpu reduce launch: " << reduce_time << " s" << endl;
        cout << "---------------------" << endl;
    }
}

double ** NMR_PFGSE::computeWalkerPhaseMagnitudesWithGpu()
{
    double **magnitudes;
    magnitudes = new double*[this->gradientPoints];
    for(uint kIdx = 0; kIdx < this->gradientPoints; kIdx++)
    {
        magnitudes[kIdx] = new double[this->NMR.walkers.size()];
    }

    cout << "trying in gpu ^^" << endl;

    bool time_verbose = true;
    double tick;
    double copy_time = 0.0;
    double kernel_time = 0.0;
    double buffer_time = 0.0;

    // CUDA event recorder to measure computation time in device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // integer values for sizing issues
    uint numberOfWalkers = this->NMR.numberOfWalkers;
    double voxelResolution = this->NMR.imageVoxelResolution; 

    // define parameters for CUDA kernel launch: blockDim, gridDim etc
    uint threadsPerBlock = this->NMR.rwNMR_config.getThreadsPerBlock();
    uint blocksPerKernel = this->NMR.rwNMR_config.getBlocks();
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

    // Host and Device memory data allocation
    // pointers used in host array conversion
    myAllocator arrayFactory;
    int *h_walker_x0 = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_y0 = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_z0 = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_px = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_py = arrayFactory.getIntArray(walkersPerKernel);
    int *h_walker_pz = arrayFactory.getIntArray(walkersPerKernel);
    double *h_energy = arrayFactory.getDoubleArray(walkersPerKernel);
    double *h_phase = arrayFactory.getDoubleArray(walkersPerKernel);

    // Declaration of pointers to device data arrays
    int *d_walker_x0;
    int *d_walker_y0;
    int *d_walker_z0;
    int *d_walker_px;
    int *d_walker_py;
    int *d_walker_pz;
    double *d_energy;
    double *d_phase;

    // Memory allocation in device for data arrays
    cudaMalloc((void **)&d_walker_x0, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_y0, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_z0, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_px, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_py, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_pz, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_energy, walkersPerKernel * sizeof(double));
    cudaMalloc((void **)&d_phase, walkersPerKernel * sizeof(double));
    
    tick = omp_get_wtime();
    for (uint idx = 0; idx < walkersPerKernel; idx++)
    {
        h_energy[idx] = 0.0;
    }
    
    for (uint idx = 0; idx < walkersPerKernel; idx++)
    {
        h_phase[idx] = 0.0;
    } 
    buffer_time += omp_get_wtime() - tick;

    // PFG main loop
    for (uint packId = 0; packId < (numberOfWalkerPacks - 1); packId++)
    {
        // set offset in walkers vector
        uint packOffset = packId * walkersPerKernel;

        // Host data copy
        // copy original walkers' data to temporary host arrays
        tick = omp_get_wtime();
        if(this->NMR.rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = walkersPerKernel;
            int loop_start, loop_finish;

            #pragma omp parallel shared(packOffset, h_walker_x0, h_walker_y0, h_walker_z0, h_walker_px, h_walker_py, h_walker_pz, h_energy, h_phase) private(loop_start, loop_finish) 
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
                    h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                    h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                    h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
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
                h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
                h_energy[id] = this->NMR.walkers[id + packOffset].energy;
                h_phase[id] = this->NMR.walkers[id + packOffset].energy;
            }
        }
        buffer_time += omp_get_wtime() - tick;

        // Device data copy
        // copy host data to device
        tick = omp_get_wtime();
        cudaMemcpy(d_walker_x0, h_walker_x0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_y0, h_walker_y0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_z0, h_walker_z0, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_px, h_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, h_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, h_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, h_energy, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase, h_phase, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        copy_time += omp_get_wtime() - tick;

        // Launch kernel for GPU computation
        
        for(int kIdx = 0; kIdx < gradientPoints; kIdx++)
        {
            double k_X = this->vecK[kIdx].getX();
            double k_Y = this->vecK[kIdx].getY();
            double k_Z = this->vecK[kIdx].getZ();
            
            // kernel call to compute walkers individual phase
            tick = omp_get_wtime();
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
            kernel_time += omp_get_wtime() - tick;

            tick = omp_get_wtime();
            cudaMemcpy(h_phase, d_phase, walkersPerKernel * sizeof(double), cudaMemcpyDeviceToHost);
            copy_time += omp_get_wtime() - tick;

            // Host data copy
            // copy original walkers' data to temporary host arrays
            tick = omp_get_wtime();
            if(this->NMR.rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = walkersPerKernel;
                int loop_start, loop_finish;

                #pragma omp parallel shared(packOffset, h_phase) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        magnitudes[kIdx][id + packOffset] = h_phase[id];
                    }
                }
            } else
            {
                for (uint id = 0; id < walkersPerKernel; id++)
                {
                    magnitudes[kIdx][id + packOffset] = h_phase[id];
                }
            }
            buffer_time += omp_get_wtime() - tick;   
        }         
    }     

    if (lastWalkerPackSize > 0)
    {
        // last Walker pack is done explicitly
        // set offset in walkers vector
        uint packOffset = (numberOfWalkerPacks - 1) * walkersPerKernel;

        // Host data copy
        // copy original walkers' data to temporary host arrays
        tick = omp_get_wtime();
        if(this->NMR.rwNMR_config.getOpenMPUsage())
        {
            // set omp variables for parallel loop throughout walker list
            const int num_cpu_threads = omp_get_max_threads();
            const int loop_size = lastWalkerPackSize;
            int loop_start, loop_finish;

            #pragma omp parallel shared(packOffset, h_walker_x0, h_walker_y0, h_walker_z0, h_walker_px, h_walker_py, h_walker_pz, h_energy, h_phase) private(loop_start, loop_finish) 
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
                    h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                    h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                    h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
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
                h_walker_px[id] = this->NMR.walkers[id + packOffset].position_x;
                h_walker_py[id] = this->NMR.walkers[id + packOffset].position_y;
                h_walker_pz[id] = this->NMR.walkers[id + packOffset].position_z;
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
        buffer_time += omp_get_wtime() - tick;

        // Device data copy
        // copy host data to device
        tick = omp_get_wtime();
        cudaMemcpy(d_walker_x0, h_walker_x0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_y0, h_walker_y0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_z0, h_walker_z0, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_px, h_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, h_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, h_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, h_energy, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase, h_phase, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        copy_time += omp_get_wtime() - tick;
       
        tick = omp_get_wtime();
        for(int kIdx = 0; kIdx < this->gradientPoints; kIdx++)
        {
            double k_X = this->vecK[kIdx].getX();
            double k_Y = this->vecK[kIdx].getY();
            double k_Z = this->vecK[kIdx].getZ();
            
            // kernel call to compute walkers individual phase
            tick = omp_get_wtime();
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
            kernel_time += omp_get_wtime() - tick;

            tick = omp_get_wtime();
            cudaMemcpy(h_phase, d_phase, lastWalkerPackSize * sizeof(double), cudaMemcpyDeviceToHost);
            copy_time += omp_get_wtime() - tick;

            // Host data copy
            // copy original walkers' data to temporary host arrays
            tick = omp_get_wtime();
            if(this->NMR.rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = lastWalkerPackSize;
                int loop_start, loop_finish;

                #pragma omp parallel shared(packOffset, h_phase) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        magnitudes[kIdx][id + packOffset] = h_phase[id];
                    }
                }
            } else
            {
                for (uint id = 0; id < lastWalkerPackSize; id++)
                {
                    magnitudes[kIdx][id + packOffset] = h_phase[id];
                }
            }
            buffer_time += omp_get_wtime() - tick; 

        }

    }  

    // free pointers in host
    free(h_walker_x0);
    free(h_walker_y0);
    free(h_walker_z0);
    free(h_walker_px);
    free(h_walker_py);
    free(h_walker_pz);
    free(h_energy);
    free(h_phase);

    // and direct them to NULL
    h_walker_px = NULL;
    h_walker_py = NULL;
    h_walker_pz = NULL;
    h_energy = NULL;
    h_phase = NULL;

    // free device global memory
    cudaFree(d_walker_x0);
    cudaFree(d_walker_y0);
    cudaFree(d_walker_z0);
    cudaFree(d_walker_px);
    cudaFree(d_walker_py);
    cudaFree(d_walker_pz);
    cudaFree(d_energy);
    cudaFree(d_phase);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Done.\nCpu/Gpu elapsed time: " << elapsedTime * 1.0e-3 << " s" << endl;
    cudaDeviceReset();

    if(time_verbose)
    {
        cout << "--- Time analysis ---" << endl;
        cout << "cpu data buffer: " << buffer_time << " s" << endl;
        cout << "gpu data copy: " << copy_time << " s" << endl;
        cout << "gpu kernel launch: " << kernel_time << " s" << endl;
        cout << "---------------------" << endl;
    }

    return magnitudes;
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
    return (pulse_width * 1.0e-03) * (giromagneticRatio * 1.0e+06) * (gradientMagnitude * 1.0e-08);
}

__device__ int convertLocalToGlobal(int _localPos, int _shiftConverter)
{
    return (_localPos >> _shiftConverter);
}

__device__ double dotProduct(double vec1X, double vec1Y, double vec1Z, double vec2X, double vec2Y, double vec2Z)
{
    return (vec1X*vec2X + vec1Y*vec2Y + vec1Z*vec2Z);
}