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
#include "NMR_cpmg.h"
#include "NMR_cpmg_cuda.h"



/* 
    GPU kernel for NMR CPMG simulation 
    in this kernel, each thread will represent a unique walker
    noflux condition is applied as image boundary treatment
*/
__global__ void CPMG_walk_noflux(int *walker_px,
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
                nextDirection = computeNextDirection_CPMG(local_seed);

                nextDirection = checkBorder_CPMG(convertLocalToGlobal_CPMG(position_x, shift_convert),
                                                 convertLocalToGlobal_CPMG(position_y, shift_convert),
                                                 convertLocalToGlobal_CPMG(position_z, shift_convert),
                                                 nextDirection,
                                                 map_columns,
                                                 map_rows,
                                                 map_depth);

                computeNextPosition_CPMG(position_x,
                                       position_y,
                                       position_z,
                                       nextDirection,
                                       next_x,
                                       next_y,
                                       next_z);

                if (checkNextPosition_CPMG(convertLocalToGlobal_CPMG(next_x, shift_convert),
                                           convertLocalToGlobal_CPMG(next_y, shift_convert),
                                           convertLocalToGlobal_CPMG(next_z, shift_convert),
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

/* 
    GPU kernel for NMR CPMG simulation 
    in this kernel, each thread will represent a unique walker
    noflux condition is applied as image boundary treatment
*/
__global__ void CPMG_walk_noflux_field(int *walker_px,
                                       int *walker_py,
                                       int *walker_pz,
                                       double *penalty,
                                       double *energy,
                                       double *phase,
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
                                       const uint shift_convert,
                                       const double gamma,
                                       const double timeInterval,
                                       const double *field)
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

    // thread variables for phase computation
    double local_phase;
    double gammatau = 0.5*gamma*timeInterval;
    uint stepsPerInversion = stepsPerEcho / 2;
    long fieldIdx;
    int row_scale = map_columns;
    int depth_scale = map_columns * map_rows;

    // now begin the "walk" procedure de facto
    if (walkerId < numberOfWalkers)
    {
        position_x = walker_px[walkerId];
        position_y = walker_py[walkerId];
        position_z = walker_pz[walkerId];
        local_dFactor = penalty[walkerId];
        local_seed = seed[walkerId];
        local_phase = phase[walkerId];
        energyLvl = energy[walkerId + energy_OFFSET];
      

        for (int echo = 0; echo < echoesPerKernel; echo++)
        {    
            // update the offset
            energy_OFFSET = echo * energyArraySize;

            for(uint inv = 0; inv < 2; inv++)
            {
                // phase inversion
                local_phase = -local_phase;

                for (int step = 0; step < stepsPerInversion; step++)
                {
                    // update phase at starting point
                    fieldIdx = getFieldIndex(convertLocalToGlobal_CPMG(position_x, shift_convert),
                                             convertLocalToGlobal_CPMG(position_y, shift_convert),
                                             convertLocalToGlobal_CPMG(position_z, shift_convert),
                                             row_scale, 
                                             depth_scale);
                    local_phase += gammatau * field[fieldIdx];
                    
                    // compute next direction and next position
                    nextDirection = computeNextDirection_CPMG(local_seed);
                    nextDirection = checkBorder_CPMG(convertLocalToGlobal_CPMG(position_x, shift_convert),
                                                     convertLocalToGlobal_CPMG(position_y, shift_convert),
                                                     convertLocalToGlobal_CPMG(position_z, shift_convert),
                                                     nextDirection,
                                                     map_columns,
                                                     map_rows,
                                                     map_depth);
                    computeNextPosition_CPMG(position_x, 
                                             position_y, 
                                             position_z, 
                                             nextDirection, 
                                             next_x, 
                                             next_y, 
                                             next_z);

                    // check if next position is a valid position
                    if (checkNextPosition_CPMG(convertLocalToGlobal_CPMG(next_x, shift_convert),
                                               convertLocalToGlobal_CPMG(next_y, shift_convert),
                                               convertLocalToGlobal_CPMG(next_z, shift_convert),
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
                        // walker hits the wall and comes back to the same position
                        // walker loses energy due to this collision
                        energyLvl = energyLvl * local_dFactor;
                    }

                    // update phase at finishing point
                                        // update phase at starting point
                    fieldIdx = getFieldIndex(convertLocalToGlobal_CPMG(position_x, shift_convert),
                                             convertLocalToGlobal_CPMG(position_y, shift_convert),
                                             convertLocalToGlobal_CPMG(position_z, shift_convert),
                                             row_scale, 
                                             depth_scale);
                    local_phase += gammatau * field[fieldIdx];                
                }
            }

            // account for phase relaxation
            energyLvl *= cos(local_phase);

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
        phase[walkerId] = local_phase;
    }
}

/* 
    GPU kernel for NMR CPMG simulation 
    in this kernel, each thread will represent a unique walker
    periodic condition is applied as image boundary treatment
*/
__global__ void CPMG_walk_periodic(int *walker_px,
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
    int localPosX, localPosY, localPosZ;
    int imgPosX, imgPosY, imgPosZ;
    double localDFactor;
    uint64_t localSeed;

    // thread variables for future movements
    int localNextX, localNextY, localNextZ;
    direction nextDirection = None;

    // 1st energy array offset
    // in first echo, walker's energy is stored in last echo of previous kernel launch
    uint energy_OFFSET = (echoesPerKernel - 1) * energyArraySize;
    double energyLvl;

    // now begin the "walk" procedure de facto
    if (walkerId < numberOfWalkers)
    {
        // Local variables for unique read from device global memory
        localPosX = walker_px[walkerId];
        localPosY = walker_py[walkerId];
        localPosZ = walker_pz[walkerId];
        localDFactor = penalty[walkerId];
        localSeed = seed[walkerId];
        energyLvl = energy[walkerId + energy_OFFSET];
            
        for (int echo = 0; echo < echoesPerKernel; echo++)
        {
            // update the offset
            energy_OFFSET = echo * energyArraySize;

            for (int step = 0; step < stepsPerEcho; step++)
            {            
                nextDirection = computeNextDirection_CPMG(localSeed); 
                computeNextPosition_CPMG(localPosX,
                                         localPosY,
                                         localPosZ,
                                         nextDirection,
                                         localNextX,
                                         localNextY,
                                         localNextZ);

                // update img position
                imgPosX = convertLocalToGlobal_CPMG(localNextX, shift_convert) % map_columns;
                if(imgPosX < 0) imgPosX += map_columns;

                imgPosY = convertLocalToGlobal_CPMG(localNextY, shift_convert) % map_rows;
                if(imgPosY < 0) imgPosY += map_rows;

                imgPosZ = convertLocalToGlobal_CPMG(localNextZ, shift_convert) % map_depth;
                if(imgPosZ < 0) imgPosZ += map_depth;

                if (checkNextPosition_CPMG(imgPosX, 
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
                    energyLvl = energyLvl * localDFactor;
                }
            }

            // walker's energy device global memory update
            // must be done for each echo
            energy[walkerId + energy_OFFSET] = energyLvl;
        }

        // position and seed device global memory update
        // must be done for each kernel
        walker_px[walkerId] = localPosX;
        walker_py[walkerId] = localPosY;
        walker_pz[walkerId] = localPosZ;
        seed[walkerId] = localSeed;
    }
}

/* 
    GPU kernel for NMR CPMG simulation 
    in this kernel, each thread will represent a unique walker
    mirror condition is applied as image boundary treatment
*/
__global__ void CPMG_walk_mirror(int *walker_px,
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
    int globalPosX, globalPosY, globalPosZ;
    int localPosX, localPosY, localPosZ;
    int imgPosX, imgPosY, imgPosZ;
    int mirror, antimirror;
    double localDFactor;
    uint64_t localSeed;

    // thread variables for future movements
    int localNextX, localNextY, localNextZ;
    direction nextDirection = None;

    // 1st energy array offset
    // in first echo, walker's energy is stored in last echo of previous kernel launch
    uint energy_OFFSET = (echoesPerKernel - 1) * energyArraySize;
    double energyLvl;

    // now begin the "walk" procedure de facto
    if (walkerId < numberOfWalkers)
    {
        // Local variables for unique read from device global memory
        localPosX = walker_px[walkerId];
        localPosY = walker_py[walkerId];
        localPosZ = walker_pz[walkerId];
        localDFactor = penalty[walkerId];
        localSeed = seed[walkerId];
        energyLvl = energy[walkerId + energy_OFFSET];
            
        for (int echo = 0; echo < echoesPerKernel; echo++)
        {
            // update the offset
            energy_OFFSET = echo * energyArraySize;

            for (int step = 0; step < stepsPerEcho; step++)
            {            
                nextDirection = computeNextDirection_CPMG(localSeed); 
                computeNextPosition_CPMG(localPosX,
                                         localPosY,
                                         localPosZ,
                                         nextDirection,
                                         localNextX,
                                         localNextY,
                                         localNextZ);

                // update img position
                /*
                    coordinate X
                */
                globalPosX = convertLocalToGlobal_CPMG(localNextX, shift_convert);
                imgPosX = globalPosX % map_columns;
                if(imgPosX < 0) imgPosX += map_columns;

                if(globalPosX > 0) mirror = (globalPosX / map_columns) % 2;
                else mirror = ((-globalPosX - 1 + map_columns) / map_columns) % 2; 

                antimirror = (mirror + 1) % 2;
                imgPosX = (antimirror * imgPosX) + (mirror * (map_columns - 1 - imgPosX));    

                /*
                    coordinate Y
                */
                globalPosY = convertLocalToGlobal_CPMG(localNextY, shift_convert);
                imgPosY = globalPosY % map_rows;
                if(imgPosY < 0) imgPosY += map_rows;

                if(globalPosY > 0) mirror = (globalPosY / map_rows) % 2;
                else mirror = ((-globalPosY - 1 + map_rows) / map_rows) % 2; 

                antimirror = (mirror + 1) % 2;
                imgPosY = (antimirror * imgPosY) + (mirror * (map_rows - 1 - imgPosY));

                /*
                    coordinate Z
                */
                globalPosZ = convertLocalToGlobal_CPMG(localNextZ, shift_convert);
                imgPosZ = globalPosZ % map_depth;
                if(imgPosZ < 0) imgPosZ += map_depth;

                if(globalPosZ > 0) mirror = (globalPosZ / map_depth) % 2;
                else mirror = ((-globalPosZ - 1 + map_depth) / map_depth) % 2; 

                antimirror = (mirror + 1) % 2;
                imgPosZ = (antimirror * imgPosZ) + (mirror * (map_depth - 1 - imgPosZ));

                if (checkNextPosition_CPMG(imgPosX, 
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
                    energyLvl = energyLvl * localDFactor;
                }
            }

            // walker's energy device global memory update
            // must be done for each echo
            energy[walkerId + energy_OFFSET] = energyLvl;
        }

        // position and seed device global memory update
        // must be done for each kernel
        walker_px[walkerId] = localPosX;
        walker_py[walkerId] = localPosY;
        walker_pz[walkerId] = localPosZ;
        seed[walkerId] = localSeed;
    }
}

// GPU kernel for reducing energy array into a global energy vector
__global__ void CPMG_energyReduce(double *energy,
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

// function to call GPU kernel to execute
// walker's "walk" method in Graphics Processing Unit
void NMR_cpmg::image_simulation_cuda()
{
    string bc = this->NMR.boundaryCondition;
    cout << "- starting RW-CPMG simulation (in GPU) [bc:" << bc << "]...";

    bool time_verbose = this->CPMG_config.getTimeVerbose();
    double reset_time = 0.0;
    double copy_time = 0.0;
    double kernel_time = 0.0;
    double buffer_time = 0.0;
    double reduce_time = 0.0;
    
    double tick = omp_get_wtime();
    if(this->NMR.rwNMR_config.getOpenMPUsage())
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
        // reset walker's initial state 
        for (uint id = 0; id < this->NMR.walkers.size(); id++)
        {
            this->NMR.walkers[id].resetPosition();
            this->NMR.walkers[id].resetSeed();
            this->NMR.walkers[id].resetEnergy();
        }
    }

    // reset vector to store energy decay
    (*this).resetSignal();
    this->signal_amps.reserve(this->NMR.getNumberOfEchoes() + 1); // '+1' to accomodate time 0.0

    // get initial energy global state
    double energySum = ((double) this->NMR.walkers.size()) * this->NMR.walkers[0].getEnergy();
    this->signal_amps.push_back(energySum);

    reset_time += omp_get_wtime() - tick;


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

    uint numberOfEchoes = this->NMR.numberOfEchoes;
    uint stepsPerEcho = this->NMR.stepsPerEcho;
    uint echoesPerKernel = this->NMR.rwNMR_config.getEchoesPerKernel();
    uint kernelCalls = (uint) ceil(numberOfEchoes / (double) echoesPerKernel);


    // THIS NEEDS TO BE REVISED LATER!!!
    bool applyField = (this->internalField == NULL) ? applyField = false : applyField = true;
    double *field = (*this).getInternalFieldData();
    long fieldSize = (*this).getInternalFieldSize();
    double timeInterval = 1.0e-3 * this->NMR.getTimeInterval(); 
    double gamma = 1.0e+06 * this->NMR.getGiromagneticRatio();
    
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
    uint energyArraySize = walkersPerKernel;
    uint energyCollectorSize = (blocksPerKernel / 2);

    // Copy bitBlock3D data from host to device (only once)
    // assign pointer to bitBlock datastructure
    uint64_t *bitBlock;
    bitBlock = this->NMR.bitBlock.blocks;
    uint64_t *d_bitBlock;
    cudaMalloc((void **)&d_bitBlock, numberOfBitBlocks * sizeof(uint64_t));
    cudaMemcpy(d_bitBlock, bitBlock, numberOfBitBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Host and Device memory data allocation
    // pointers used in host array conversion
    myAllocator arrayFactory;
    int *walker_px = arrayFactory.getIntArray(walkersPerKernel);
    int *walker_py = arrayFactory.getIntArray(walkersPerKernel);
    int *walker_pz = arrayFactory.getIntArray(walkersPerKernel);
    double *penalty = arrayFactory.getDoubleArray(walkersPerKernel);
    double *phase = arrayFactory.getDoubleArray(walkersPerKernel);
    double *energy = arrayFactory.getDoubleArray(echoesPerKernel * energyArraySize);
    double *energyCollector = arrayFactory.getDoubleArray(echoesPerKernel * energyCollectorSize);
    uint64_t *seed = arrayFactory.getUInt64Array(walkersPerKernel);
    
    // temporary array to collect energy contributions for each echo in a kernel
    double *temp_globalEnergy = arrayFactory.getDoubleArray((uint)echoesPerKernel);
    double *h_globalEnergy = arrayFactory.getDoubleArray(kernelCalls * echoesPerKernel);

    tick = omp_get_wtime();
    for (uint echo = 0; echo < numberOfEchoes; echo++)
    {
        h_globalEnergy[echo] = 0.0;
    }
    buffer_time += omp_get_wtime() - tick;

    // Declaration of pointers to device data arrays
    int *d_walker_px;
    int *d_walker_py;
    int *d_walker_pz;
    double *d_penalty;
    double *d_phase;
    double *d_field;
    double *d_energy;
    double *d_energyCollector;
    uint64_t *d_seed;

    // Memory allocation in device for data arrays
    cudaMalloc((void **)&d_walker_px, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_py, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_walker_pz, walkersPerKernel * sizeof(int));
    cudaMalloc((void **)&d_penalty, walkersPerKernel * sizeof(double));
    if(applyField) cudaMalloc((void **)&d_phase, walkersPerKernel * sizeof(double));
    cudaMalloc((void **)&d_energy, echoesPerKernel * energyArraySize * sizeof(double));
    cudaMalloc((void **)&d_energyCollector, echoesPerKernel * energyCollectorSize * sizeof(double));
    cudaMalloc((void **)&d_seed, walkersPerKernel * sizeof(uint64_t));
    
    if(applyField)
    {
        cudaMalloc((void **)&d_field, fieldSize * sizeof(double));
        cudaMemcpy(d_field, field, fieldSize * sizeof(double), cudaMemcpyHostToDevice);
    }

    tick = omp_get_wtime();
    for (uint i = 0; i < energyArraySize * echoesPerKernel; i++)
    {
        energy[i] = 0.0;
    }
    buffer_time += omp_get_wtime() - tick;

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

            #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, penalty, energy, seed) private(loop_start, loop_finish) 
            {
                const int thread_id = omp_get_thread_num();
                OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                loop_start = looper.getStart();
                loop_finish = looper.getFinish(); 

                for (uint i = loop_start; i < loop_finish; i++)
                {
                    walker_px[i] = this->NMR.walkers[i + packOffset].initialPosition.x;
                    walker_py[i] = this->NMR.walkers[i + packOffset].initialPosition.y;
                    walker_pz[i] = this->NMR.walkers[i + packOffset].initialPosition.z;
                    penalty[i] = this->NMR.walkers[i + packOffset].decreaseFactor;
                    phase[i] = 0.0;
                    energy[i + ((echoesPerKernel - 1) * energyArraySize)] = this->NMR.walkers[i + packOffset].energy;
                    seed[i] = this->NMR.walkers[i + packOffset].initialSeed;
                }
            }
        } else
        {            
            for (uint i = 0; i < walkersPerKernel; i++)
            {
                walker_px[i] = this->NMR.walkers[i + packOffset].initialPosition.x;
                walker_py[i] = this->NMR.walkers[i + packOffset].initialPosition.y;
                walker_pz[i] = this->NMR.walkers[i + packOffset].initialPosition.z;
                penalty[i] = this->NMR.walkers[i + packOffset].decreaseFactor;
                phase[i] = 0.0;
                energy[i + ((echoesPerKernel - 1) * energyArraySize)] = this->NMR.walkers[i + packOffset].energy;
                seed[i] = this->NMR.walkers[i + packOffset].initialSeed;
            }
        }  
        buffer_time += omp_get_wtime() - tick;      

        // Device data copy
        // copy host data to device
        tick = omp_get_wtime();
        cudaMemcpy(d_walker_px, walker_px, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, walker_py, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_penalty, penalty, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        if(applyField) cudaMemcpy(d_phase, phase, walkersPerKernel * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, energy, echoesPerKernel * energyArraySize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, seed, walkersPerKernel * sizeof(uint64_t), cudaMemcpyHostToDevice);
        copy_time += omp_get_wtime() - tick;

        // Launch kernel for GPU computation
        for (uint kernelId = 0; kernelId < kernelCalls; kernelId++)
        {
            // define echo offset
            uint echoOffset = kernelId * echoesPerKernel;
            uint echoes = echoesPerKernel;

            /* 
                Call adequate RW kernel depending on the chosen boundary treatment
            */
            tick = omp_get_wtime();
            if(applyField)
            {
                CPMG_walk_noflux_field<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px, 
                                                                             d_walker_py, 
                                                                             d_walker_pz, 
                                                                             d_penalty, 
                                                                             d_energy,
                                                                             d_phase, 
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
                                                                             shiftConverter,
                                                                             gamma,
                                                                             timeInterval,
                                                                             d_field);
            }
            else if(!applyField and bc == "periodic")
            {
                CPMG_walk_periodic<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
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
            }
            else if(!applyField and bc == "mirror")
            {
                CPMG_walk_mirror<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
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
            } else 
            {
                CPMG_walk_noflux<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
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
            }
            cudaDeviceSynchronize();
            kernel_time += omp_get_wtime() - tick;

            
            // launch globalEnergy "reduce" kernel
            tick = omp_get_wtime();
            CPMG_energyReduce<<<blocksPerKernel / 2,
                                threadsPerBlock,
                                threadsPerBlock * sizeof(double)>>>(d_energy,
                                                                    d_energyCollector,
                                                                    energyArraySize,
                                                                    energyCollectorSize,
                                                                    echoesPerKernel);
            cudaDeviceSynchronize();
            reduce_time += omp_get_wtime() - tick;

            // copy data from gatherer array
            tick = omp_get_wtime();
            cudaMemcpy(energyCollector,
                       d_energyCollector,
                       echoesPerKernel * energyCollectorSize * sizeof(double),
                       cudaMemcpyDeviceToHost);
            copy_time += omp_get_wtime() - tick;

            //last reduce is done in CPU parallel-style using openMP
            tick = omp_get_wtime();
            CPMG_reduce_omp(temp_globalEnergy, energyCollector, echoesPerKernel, blocksPerKernel / 2);
            reduce_time += omp_get_wtime() - tick;

            // copy data from temporary array to NMR_Simulation2D "globalEnergy" vector class member
            tick = omp_get_wtime();
            for (uint echo = 0; echo < echoesPerKernel; echo++)
            {
                h_globalEnergy[echo + echoOffset] += temp_globalEnergy[echo];
            }
            buffer_time += omp_get_wtime() - tick;

            // recover last positions
            tick = omp_get_wtime();
            cudaMemcpy(walker_px, d_walker_px, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_py, d_walker_py, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_pz, d_walker_pz, walkersPerKernel * sizeof(int), cudaMemcpyDeviceToHost);      
            copy_time = omp_get_wtime() - tick;

            tick = omp_get_wtime();
            if(this->NMR.rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = walkersPerKernel;
                int loop_start, loop_finish;

                #pragma omp parallel shared(walker_px, walker_py, walker_pz, packOffset) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        this->NMR.walkers[id + packOffset].position_x = walker_px[id];
                        this->NMR.walkers[id + packOffset].position_y = walker_py[id];
                        this->NMR.walkers[id + packOffset].position_z = walker_pz[id];
                    }
                }
            } else
            {
                for (uint i = 0; i < walkersPerKernel; i++)
                {
                    this->NMR.walkers[i + packOffset].position_x = walker_px[i];
                    this->NMR.walkers[i + packOffset].position_y = walker_py[i];
                    this->NMR.walkers[i + packOffset].position_z = walker_pz[i];            
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

            #pragma omp parallel shared(packOffset, walker_px, walker_py, walker_pz, penalty, energy, seed) private(loop_start, loop_finish) 
            {
                const int thread_id = omp_get_thread_num();
                OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                loop_start = looper.getStart();
                loop_finish = looper.getFinish(); 

                for (uint i = loop_start; i < loop_finish; i++)
                {
                    walker_px[i] = this->NMR.walkers[i + packOffset].initialPosition.x;
                    walker_py[i] = this->NMR.walkers[i + packOffset].initialPosition.y;
                    walker_pz[i] = this->NMR.walkers[i + packOffset].initialPosition.z;
                    penalty[i] = this->NMR.walkers[i + packOffset].decreaseFactor;
                    phase[i] = 0.0;
                    energy[i + ((echoesPerKernel - 1) * energyArraySize)] = this->NMR.walkers[i + packOffset].energy;
                    seed[i] = this->NMR.walkers[i + packOffset].initialSeed;
                }
            }
        } else
        {            
            for (uint i = 0; i < lastWalkerPackSize; i++)
            {
                walker_px[i] = this->NMR.walkers[i + packOffset].initialPosition.x;
                walker_py[i] = this->NMR.walkers[i + packOffset].initialPosition.y;
                walker_pz[i] = this->NMR.walkers[i + packOffset].initialPosition.z;
                penalty[i] = this->NMR.walkers[i + packOffset].decreaseFactor;
                phase[i] = 0.0;
                energy[i + ((echoesPerKernel - 1) * energyArraySize)] = this->NMR.walkers[i + packOffset].energy;
                seed[i] = this->NMR.walkers[i + packOffset].initialSeed;
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
        buffer_time += omp_get_wtime() - tick;
        

        // Device data copy
        // copy host data to device
        tick = omp_get_wtime();
        cudaMemcpy(d_walker_px, walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_py, walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_walker_pz, walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_penalty, penalty, lastWalkerPackSize * sizeof(double), cudaMemcpyHostToDevice);
        if(applyField) cudaMemcpy(d_phase, phase, lastWalkerPackSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_energy, energy, echoesPerKernel * energyArraySize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, seed, lastWalkerPackSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
        copy_time += omp_get_wtime() - tick;
        

        // Launch kernel for GPU computation
        for (uint kernelId = 0; kernelId < kernelCalls; kernelId++)
        {
            // define echo offset
            uint echoOffset = kernelId * echoesPerKernel;
            uint echoes = echoesPerKernel;

            /* 
                Call adequate RW kernel depending on the chosen boundary treatment
            */
            tick = omp_get_wtime();
            if(applyField)
            {
                CPMG_walk_noflux_field<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px, 
                                                                             d_walker_py, 
                                                                             d_walker_pz, 
                                                                             d_penalty, 
                                                                             d_energy,
                                                                             d_phase, 
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
                                                                             shiftConverter,
                                                                             gamma,
                                                                             timeInterval,
                                                                             d_field);
            }
            else if(!applyField and bc == "periodic")
            {
                CPMG_walk_periodic<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
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
            }
            else if(!applyField and bc == "mirror")
            {
                CPMG_walk_mirror<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
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
            }
            else
            {
                CPMG_walk_noflux<<<blocksPerKernel, threadsPerBlock>>>(d_walker_px,
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
            }
            cudaDeviceSynchronize();
            kernel_time += omp_get_wtime() - tick;

            // launch globalEnergy "reduce" kernel
            tick = omp_get_wtime();
            CPMG_energyReduce<<<blocksPerKernel / 2,
                                threadsPerBlock,
                                threadsPerBlock * sizeof(double)>>>(d_energy,
                                                                    d_energyCollector,
                                                                    energyArraySize,
                                                                    energyCollectorSize,
                                                                    echoesPerKernel);
            cudaDeviceSynchronize();
            reduce_time += omp_get_wtime() - tick;

            // copy data from gatherer array
            tick = omp_get_wtime();
            cudaMemcpy(energyCollector,
                       d_energyCollector,
                       echoesPerKernel * energyCollectorSize * sizeof(double),
                       cudaMemcpyDeviceToHost);
            copy_time += omp_get_wtime() - tick;

            //last reduce is done in CPU parallel-style using openMP
            tick = omp_get_wtime();
            CPMG_reduce_omp(temp_globalEnergy, energyCollector, echoesPerKernel, blocksPerKernel / 2);
            reduce_time += omp_get_wtime() - tick;

            // copy data from temporary array
            tick = omp_get_wtime();
            for (uint echo = 0; echo < echoesPerKernel; echo++)
            {
                h_globalEnergy[echo + echoOffset] += temp_globalEnergy[echo];
            }
            buffer_time += omp_get_wtime() - tick;

            // recover last positions
            tick = omp_get_wtime();
            cudaMemcpy(walker_px, d_walker_px, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_py, d_walker_py, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(walker_pz, d_walker_pz, lastWalkerPackSize * sizeof(int), cudaMemcpyDeviceToHost);      
            copy_time += omp_get_wtime() - tick;

            tick = omp_get_wtime();
            if(this->NMR.rwNMR_config.getOpenMPUsage())
            {
                // set omp variables for parallel loop throughout walker list
                const int num_cpu_threads = omp_get_max_threads();
                const int loop_size = lastWalkerPackSize;
                int loop_start, loop_finish;

                #pragma omp parallel shared(walker_px, walker_py, walker_pz, packOffset) private(loop_start, loop_finish) 
                {
                    const int thread_id = omp_get_thread_num();
                    OMPLoopEnabler looper(thread_id, num_cpu_threads, loop_size);
                    loop_start = looper.getStart();
                    loop_finish = looper.getFinish(); 

                    for (uint id = loop_start; id < loop_finish; id++)
                    {
                        this->NMR.walkers[id + packOffset].position_x = walker_px[id];
                        this->NMR.walkers[id + packOffset].position_y = walker_py[id];
                        this->NMR.walkers[id + packOffset].position_z = walker_pz[id];
                    }
                }
            } else
            {
                for (uint i = 0; i < lastWalkerPackSize; i++)
                {
                    this->NMR.walkers[i + packOffset].position_x = walker_px[i];
                    this->NMR.walkers[i + packOffset].position_y = walker_py[i];
                    this->NMR.walkers[i + packOffset].position_z = walker_pz[i];            
                }
            }
            buffer_time += omp_get_wtime() - tick;  
        }
    }

    // insert to object energy values computed in gpu
    tick = omp_get_wtime();
    for (uint echo = 0; echo < numberOfEchoes; echo++)
    {
        this->signal_amps.push_back(h_globalEnergy[echo]);
    }
    buffer_time += omp_get_wtime() - tick;

    // free pointers in host
    free(walker_px);
    free(walker_py);
    free(walker_pz);
    free(penalty);
    free(phase);
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
    phase = NULL;
    h_globalEnergy = NULL;
    energy = NULL;
    energyCollector = NULL;
    temp_globalEnergy = NULL;
    seed = NULL;

    // also direct the bitBlock pointer created in this context
    // (original data is kept safe)
    bitBlock = NULL;
    field = NULL;

    // free device global memory
    cudaFree(d_walker_px);
    cudaFree(d_walker_py);
    cudaFree(d_walker_pz);
    cudaFree(d_penalty);
    if(applyField) cudaFree(d_phase);
    if(applyField) cudaFree(d_field);
    cudaFree(d_energy);
    cudaFree(d_energyCollector);
    cudaFree(d_seed);
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
        cout << "cpu data reset:    \t" << reset_time << " s" << endl;
        cout << "cpu data buffer:   \t" << buffer_time << " s" << endl;
        cout << "gpu data copy:     \t" << copy_time << " s" << endl;
        cout << "gpu kernel launch: \t" << kernel_time << " s" << endl;
        cout << "gpu reduce launch: \t" << reduce_time << " s" << endl;
        cout << "---------------------" << endl;
    }
}


/////////////////////////////////////////////////////////////////////
//////////////////////// HOST FUNCTIONS ///////////////////////////
/////////////////////////////////////////////////////////////////////
void CPMG_reduce_omp(double *temp_collector, double *array, int numberOfEchoes, uint arraySizePerEcho)
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

/////////////////////////////////////////////////////////////////////
//////////////////////// DEVICE FUNCTIONS ///////////////////////////
/////////////////////////////////////////////////////////////////////

__device__ direction computeNextDirection_CPMG(uint64_t &seed)
{
    // generate random number using xorshift algorithm
    xorshift64_state xor_state;
    xor_state.a = seed;
    seed = xorShift64_CPMG(&xor_state);
    uint64_t rand = seed;

    // set direction based on the random number
    direction nextDirection = (direction)(mod6_CPMG(rand) + 1);
    return nextDirection;
}

__device__ uint64_t xorShift64_CPMG(struct xorshift64_state *state)
{
    uint64_t x = state->a;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return state->a = x;
}

__device__ uint64_t mod6_CPMG(uint64_t a)
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

__device__ direction checkBorder_CPMG(int walker_px,
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

__device__ void computeNextPosition_CPMG(int &walker_px,
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

__device__ bool checkNextPosition_CPMG(int next_x,
                                     int next_y,
                                     int next_z,
                                     const uint64_t *bitBlock,
                                     const int bitBlockColumns,
                                     const int bitBlockRows)
{
    int blockIndex = findBlockIndex_CPMG(next_x, next_y, next_z, bitBlockColumns, bitBlockRows);
    int nextBit = findBitIndex_CPMG(next_x, next_y, next_z);
    uint64_t nextBlock = bitBlock[blockIndex];

    return (!checkIfBlockBitIsWall_CPMG(nextBlock, nextBit));
};

__device__ int findBlockIndex_CPMG(int next_x, int next_y, int next_z, int bitBlockColumns, int bitBlockRows)
{
    // "x >> 2" is like "x / 4" in bitwise operation
    int block_x = next_x >> 2;
    int block_y = next_y >> 2;
    int block_z = next_z >> 2;
    int blockIndex = block_x + block_y * bitBlockColumns + block_z * (bitBlockColumns * bitBlockRows);

    return blockIndex;
}

__device__ int findBitIndex_CPMG(int next_x, int next_y, int next_z)
{
    // "x & (n - 1)" is lise "x % n" in bitwise operation
    int bit_x = next_x & (COLUMNSPERBLOCK3D - 1);
    int bit_y = next_y & (ROWSPERBLOCK3D - 1);
    int bit_z = next_z & (DEPTHPERBLOCK3D - 1);
    // "x << 3" is like "x * 8" in bitwise operation
    int bitIndex = bit_x + (bit_y << 2) + ((bit_z << 2) << 2);

    return bitIndex;
}

__device__ bool checkIfBlockBitIsWall_CPMG(uint64_t nextBlock, int nextBit)
{
    return ((nextBlock >> nextBit) & 1ull);
}

__device__ int convertLocalToGlobal_CPMG(int _localPos, uint _shiftConverter)
{
    return (_localPos >> _shiftConverter);
}

__device__ long getFieldIndex(int _x, int _y, int _z, int _rowScale, int _depthScale)
{ 
    return (_x + (_y * _rowScale) + (_z * _depthScale)); 
}