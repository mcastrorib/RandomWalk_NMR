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

#include "myAllocator.h"

int* myAllocator::getIntArray(uint size)
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

uint* myAllocator::getUIntArray(uint size)
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

double* myAllocator::getDoubleArray(uint size)
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

uint64_t* myAllocator::getUInt64Array(uint size)
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