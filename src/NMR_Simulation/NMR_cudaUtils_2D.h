#ifndef CUDAUTIL_2D_H_
#define CUDAUTIL_2D_H_

#include "stdint.h"
#include "cuda_runtime.h"
#include "NMR_defs.h"


// Kernel declarations

// walker class "map" method GPU kernel
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
                    const uint shift_convert);

// walker class "walk" method GPU kernel
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
                     const uint shift_convert);

// GPU kernel for reducing energy array into a global energy vector
__global__ void energyReduce_shared(double *energy,
                                    double *collector,
                                    const uint energyArraySize,
                                    const uint collectorSize,
                                    const int echoesPerKernel);

// Host functions
void reduce_omp(double *temp_collector, double *array, int numberOfEchoes, uint arraySizePerEcho);
void test_omp(uint size);

uint *setUIntArray(uint size);
int *setIntArray(uint size);
double *setDoubleArray(uint size);
uint64_t *setUInt64Array(uint size);

void copyVectorBtoA(int a[], int b[], uint size);
void copyVectorBtoA(double a[], double b[], uint size);
void copyVectorBtoA(uint64_t a[], uint64_t b[], uint size);

void vectorElementSwap(int *vector, uint index1, uint index2);
void vectorElementSwap(double *vector, uint index1, uint index2);
void vectorElementSwap(uint64_t *vector, uint index1, uint index2);

// Device functions
__device__ direction computeNextDirection(uint64_t &seed);

__device__ direction checkBorder(int walker_px,
                                 int walker_py,
                                 direction &nextDirection,
                                 const int map_columns,
                                 const int map_rows);

__device__ void computeNextPosition(int &walker_px,
                                    int &walker_py,
                                    direction nextDirection,
                                    int &next_x,
                                    int &next_y);

__device__ bool checkNextPosition(int next_x,
                                  int next_y,
                                  const uint64_t *bitBlock,
                                  const int bitBlockColumns);

__device__ int findBlockIndex(int next_x, int next_y, int bitBlockColumns);
__device__ int findBitIndex(int next_x, int next_y);
__device__ bool checkIfBlockBitIsWall(uint64_t currentBlock, int currentBit);
__device__ uint64_t xorShift64(struct xorshift64_state *state);
__device__ uint convertLocalToGlobal_2D(uint _localPos, uint _shiftConverter);

#endif