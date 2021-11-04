#ifndef PFG_CUDA_H_
#define PFG_CUDA_H_

#include "stdint.h"
#include "../RNG/xorshift.h"
#include "NMR_defs.h"
#include "cuda_runtime.h"


// Kernel declarations
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
                                 const int shift_convert);

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
                                 const int shift_convert);

__global__ void PFG_map_mirror ( int *walker_px,
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
                                 const int shift_convert);

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
                            const double k_Z,
                            const uint kValues);

__global__ void PFG_measure_with_sampling(int *walker_x0,
                                          int *walker_y0, 
                                          int *walker_z0,
                                          int *walker_xF,
                                          int *walker_yF,
                                          int *walker_zF,
                                          double *energy,
                                          double *phase,
                                          const uint blocksPerSample,
                                          const uint walkersPerSample,
                                          const uint sampleTail,
                                          const double voxelResolution,
                                          const double k_X,
                                          const double k_Y,
                                          const double k_Z);

__global__ void PFG_measure_with_sampling_all_K ( int *walker_x0,
                                                  int *walker_y0, 
                                                  int *walker_z0,
                                                  int *walker_xF,
                                                  int *walker_yF,
                                                  int *walker_zF,
                                                  double *energy,
                                                  double *phase,
                                                  const double *k_X,
                                                  const double *k_Y,
                                                  const double *k_Z,
                                                  const uint kValues,
                                                  const uint blocksPerSample,
                                                  const uint walkersPerSample,
                                                  const uint sampleTail,
                                                  const double voxelResolution);

__global__ void PFG_reduce_with_sampling(double *MktCollector,
                                         double *phase); 

__global__ void PFG_evaluate_energy( uint *collisions,
                                     double *penalty,
                                     double *energy,
                                     const uint numberOfWalkers);

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
__device__ int convertLocalToGlobal_PFG(int _localPos, int _shiftConverter);

__device__ double compute_PFG_k_value(double gradientMagnitude, double pulse_width, double giromagneticRatio);

__device__ double dotProduct(double vec1X, double vec1Y, double vec1Z, double vec2X, double vec2Y, double vec2Z);

#endif
