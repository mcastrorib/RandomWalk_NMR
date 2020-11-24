#ifndef PFG_CUDA_H_
#define PFG_CUDA_H_

#include "stdint.h"
#include "../RNG/xorshift.h"
#include "NMR_defs.h"
#include "cuda_runtime.h"


// Kernel declarations
__global__ void PFG_walk(int *walker_px,
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
                         const uint numberOfSteps,
                         const uint map_columns,
                         const uint map_rows,
                         const uint map_depth,
                         const uint shift_convert);

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
                            const double k_Z);

__global__ void PFG_reduce(double *data,
                           double *deposit,
                           const uint data_size,
                           const uint deposit_size);

// Device functions for 3D simulation
__device__ direction computeNextDirection_PFG(uint64_t &seed);

__device__ direction checkBorder_PFG(int walker_px,
                                    int walker_py,
                                    int walker_pz,
                                    direction &nextDirection,
                                    const int map_columns,
                                    const int map_rows,
                                    const int map_depth);

__device__ void computeNextPosition_PFG(int &walker_px,
                                       int &walker_py,
                                       int &walker_pz,
                                       direction nextDirection,
                                       int &next_x,
                                       int &next_y,
                                       int &next_z);

__device__ bool checkNextPosition_PFG(int next_x,
                                     int next_y,
                                     int next_z,
                                     const uint64_t *bitBlock,
                                     const int bitBlockColumns,
                                     const int bitBlockRows);

__device__ int findBlockIndex_PFG(int next_x, int next_y, int next_z, int bitBlockColumns, int bitBlockRows);
__device__ int findBitIndex_PFG(int next_x, int next_y, int next_z);
__device__ bool checkIfBlockBitIsWall_PFG(uint64_t currentBlock, int currentBit);
__device__ uint64_t xorShift64_PFG(struct xorshift64_state *state);
__device__ uint64_t mod6_PFG(uint64_t a);

__device__ double compute_PFG_k_value(double gradientMagnitude, double pulse_width, double giromagneticRatio);
__host__ double compute_k(double gradientMagnitude, double pulse_width, double giromagneticRatio);
__device__ uint convertLocalToGlobal(uint _localPos, uint _shiftConverter);
__device__ double dotProduct(double vec1X, double vec1Y, double vec1Z, double vec2X, double vec2Y, double vec2Z);

#endif
