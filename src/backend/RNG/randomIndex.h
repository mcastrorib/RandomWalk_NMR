#ifndef RANDOMINDEX_H_
#define RANDOMINDEX_H_

#include <iostream>
#include <random>

class RandomIndex {
    std::mt19937 eng;
    std::uniform_int_distribution<> distr;
public:
    RandomIndex(int lower, int upper, int threadID = 0) 
        : eng(std::random_device()() * (1 + threadID))
        , distr(lower, upper)
    {}

    int operator()() { 
         return distr(eng);
    }
};

#endif