#ifndef GA_POPULATION_H
#define GA_POPULATION_H

#include <vector>
#include "individual.h"

using namespace std;

//
class Population
{
public:
    vector<Individual> individuals;

    // methods
    Population()
    {
        individuals = vector<Individual>();
    }
    Population(uint _popSize, uint _nvar)
    {
        individuals = vector<Individual>(_popSize, Individual(_nvar));
    };

    // copy constructor
    Population(const Population &otherPop)
    {
        this->individuals = otherPop.individuals;
    };
    virtual ~Population()
    {
        if (individuals.size() > 0)
            individuals.clear();
    }
};

#endif