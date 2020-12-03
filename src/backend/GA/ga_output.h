#ifndef GA_OUTPUT_H
#define GA_OUTPUT_H

#include <vector>
#include "individual.h"
#include "population.h"

using namespace std;

class GA_output
{
public:
    Individual bestIndividual;
    vector<Population> population;

    // methods
    virtual ~GA_output()
    {
        if (population.size() > 0)
            population.clear();
    }
};

#endif