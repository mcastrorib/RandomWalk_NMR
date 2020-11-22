#ifndef RNG_H_
#define RNG_H_

#include <iostream>
#include <random>
#include <stdint.h>

uint64_t RNG_uint64()
{
	std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 4294967296); 

    // RNG warm up
    for(int i = 0; i < 1000; i++) dist(rng);

    return (dist(rng));
}

#endif