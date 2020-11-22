#ifndef CUDAUTIL_3D_H_
#define CUDAUTIL_3D_H_

#include "stdint.h"
#include "../RNG/xorshift.h"
#include "NMR_defs.h"
#include "cuda_runtime.h"

// walker class "map" method GPU kernel
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
                       const uint shift_convert);

// walker class "walk" method GPU kernel
__global__ void walk_3D(int *walker_px,
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
__global__ void energyReduce_shared_3D(double *energy,
                                       double *collector,
                                       const uint energyArraySize,
                                       const uint collectorSize,
                                       const uint echoesPerKernel);

__global__ void walk_PFG(int *walker_px,
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
                            const double giromagneticRatio);

__global__ void reduce_PFG(double *data,
                           double *deposit,
                           const uint data_size,
                           const uint deposit_size);

// Host functions
void reduce_omp_3D(double *temp_collector, double *array, int numberOfEchoes, uint arraySizePerEcho);
void test_omp_3D(uint size);

int *setIntArray_3D(uint size);
uint *setUIntArray_3D(uint size);
double *setDoubleArray_3D(uint size);
uint64_t *setUInt64Array_3D(uint size);

void copyVectorBtoA_3D(int a[], int b[], uint size);
void copyVectorBtoA_3D(double a[], double b[], uint size);
void copyVectorBtoA_3D(uint64_t a[], uint64_t b[], uint size);

void vectorElementSwap_3D(int *vector, uint index1, uint index2);
void vectorElementSwap_3D(double *vector, uint index1, uint index2);
void vectorElementSwap_3D(uint64_t *vector, uint index1, uint index2);

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

__device__ double compute_pfgse_k_value(double gradientMagnitude, double tiny_delta = PFGSE_TINY_DELTA, double giromagneticRatio = GIROMAGNETIC_RATIO);
__device__ int convertLocalToGlobal_3D(int _localPos, uint _shiftConverter);
#endif
