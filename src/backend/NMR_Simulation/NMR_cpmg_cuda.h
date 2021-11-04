#ifndef NMR_CPMG_CUDA_H_
#define NMR_CPMG_CUDA_H_

#include "stdint.h"
#include "../RNG/xorshift.h"
#include "NMR_defs.h"
#include "cuda_runtime.h"


// Kernel declarations
// walker class "walk" method GPU kernel
__global__ void CPMG_walk_noflux(int *walker_px,
                                 int *walker_py,
                                 int *walker_pz,
                                 double *decreaseFactor,
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
                                 const uint shift_convert);

__global__ void CPMG_walk_periodic(int *walker_px,
                                   int *walker_py,
                                   int *walker_pz,
                                   double *decreaseFactor,
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
                                   const uint shift_convert);

__global__ void CPMG_walk_mirror(int *walker_px,
                                 int *walker_py,
                                 int *walker_pz,
                                 double *decreaseFactor,
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
                                 const uint shift_convert);

// GPU kernel for reducing energy array into a global energy vector
__global__ void CPMG_energyReduce(double *energy,
                                  double *collector,
                                  const uint energyArraySize,
                                  const uint collectorSize,
                                  const uint echoesPerKernel);



// Host functions
void CPMG_reduce_omp(double *temp_collector, double *array, int numberOfEchoes, uint arraySizePerEcho);


// Device functions for 3D simulation
__device__ direction computeNextDirection_CPMG(uint64_t &seed);

__device__ direction checkBorder_CPMG(int walker_px,
                                    int walker_py,
                                    int walker_pz,
                                    direction &nextDirection,
                                    const int map_columns,
                                    const int map_rows,
                                    const int map_depth);

__device__ void computeNextPosition_CPMG(int &walker_px,
                                       int &walker_py,
                                       int &walker_pz,
                                       direction nextDirection,
                                       int &next_x,
                                       int &next_y,
                                       int &next_z);

__device__ bool checkNextPosition_CPMG(int next_x,
                                     int next_y,
                                     int next_z,
                                     const uint64_t *bitBlock,
                                     const int bitBlockColumns,
                                     const int bitBlockRows);

__device__ int findBlockIndex_CPMG(int next_x, int next_y, int next_z, int bitBlockColumns, int bitBlockRows);
__device__ int findBitIndex_CPMG(int next_x, int next_y, int next_z);
__device__ bool checkIfBlockBitIsWall_CPMG(uint64_t currentBlock, int currentBit);
__device__ uint64_t xorShift64_CPMG(struct xorshift64_state *state);
__device__ uint64_t mod6_CPMG(uint64_t a);
__device__ int convertLocalToGlobal_CPMG(int _localPos, uint _shiftConverter);
#endif
