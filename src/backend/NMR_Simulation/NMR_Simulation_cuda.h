#ifndef NMR_SIMULATION_CUDA_H_
#define NMR_SIMULATION_CUDA_H_

#include "stdint.h"
#include "../RNG/xorshift.h"
#include "NMR_defs.h"
#include "cuda_runtime.h"


// Kernel declarations

// -- 2D
__global__ void map_2D(int *walker_px,
                       int *walker_py,
                       uint *collisions,
                       uint64_t *seed,
                       const uint64_t *bitBlock,
                       const int bitBlockColumns,
                       const uint numberOfWalkers,
                       const uint numberOfSteps,
                       const int map_columns,
                       const int map_rows,
                       const uint shift_convert);



// Device functions for 2D simulation
__device__ direction computeNextDirection_2D(uint64_t &seed);

__device__ direction checkBorder_2D(int walker_px,
                                    int walker_py,
                                    direction &nextDirection,
                                    const int map_columns,
                                    const int map_rows);

__device__ void computeNextPosition_2D(int &walker_px,
                                       int &walker_py,
                                       direction nextDirection,
                                       int &next_x,
                                       int &next_y);

__device__ bool checkNextPosition_2D(int next_x,
                                     int next_y,
                                     const uint64_t *bitBlock,
                                     const int bitBlockColumns);

__device__ int findBlockIndex_2D(int next_x, int next_y, int bitBlockColumns);
__device__ int findBitIndex_2D(int next_x, int next_y);
__device__ bool checkIfBlockBitIsWall_2D(uint64_t currentBlock, int currentBit);
__device__ uint64_t xorShift64_2D(struct xorshift64_state *state);
__device__ uint convertLocalToGlobal_2D(uint _localPos, uint _shiftConverter);

// -----
// -- 3D
__global__ void map_3D_noflux( int *walker_px,
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
                               const uint shift_convert);

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
                                const uint shift_convert);


// Device functions for 3D simulation
__device__ direction computeNextDirection_3D(uint64_t &seed);

__device__ direction checkBorder_3D(int walker_px,
                                    int walker_py,
                                    int walker_pz,
                                    direction &nextDirection,
                                    const int map_columns,
                                    const int map_rows,
                                    const int map_depth);

__device__ void computeNextPosition_3D(int &walker_px,
                                       int &walker_py,
                                       int &walker_pz,
                                       direction nextDirection,
                                       int &next_x,
                                       int &next_y,
                                       int &next_z);

__device__ bool checkNextPosition_3D(int next_x,
                                     int next_y,
                                     int next_z,
                                     const uint64_t *bitBlock,
                                     const int bitBlockColumns,
                                     const int bitBlockRows);

__device__ int findBlockIndex_3D(int next_x, int next_y, int next_z, int bitBlockColumns, int bitBlockRows);
__device__ int findBitIndex_3D(int next_x, int next_y, int next_z);
__device__ bool checkIfBlockBitIsWall_3D(uint64_t currentBlock, int currentBit);
__device__ uint64_t xorShift64_3D(struct xorshift64_state *state);
__device__ uint64_t mod6_3D(uint64_t a);
__device__ int convertLocalToGlobal_3D(int _localPos, uint _shiftConverter);

#endif