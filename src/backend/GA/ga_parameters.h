#ifndef GA_PARAMETERS_H
#define GA_PARAMETERS_H

#include <iostream>
#include <string>
#include <vector>

#include "ga_defs.h"

using namespace std;

class GA_parameters
{

public:
    uint populationSize;
    double beta;
    double offspringProportion;
    double mutationRatio;
    double gamma;
    double mutationDeviation;
    double diversityRate;

    //methods
    GA_parameters()
    {
        this->populationSize = GA_POPULATION_SIZE;

        // The proportion of the next generation that is composed by new individuals
        // if this value is set 1.0, every generation is composed by new crossoved individuals        
        this->offspringProportion = GA_OFFSPRING_PROPORTION;
        this->gamma = GA_GAMMA;
        this->mutationRatio = GA_MUTATION_RATIO;
        this->mutationDeviation = GA_MUTATION_DEVIATION;

        // beta is used only for minimization problems
        // beta is a parameter used to construct the parent probability pool array
        this->beta = GA_BETA;

        this->diversityRate = GA_DIVERSITY;
    };
    virtual ~GA_parameters(){};

    void printInfo()
    {
        cout << "GA PARAMETERS: " << endl;
        cout << "population size: " << populationSize << endl;
        cout << "offspring proportion: " << offspringProportion << endl;
        cout << "gamma: " << gamma << endl;
        cout << "mutation ratio: " << mutationRatio << endl;
        cout << "mutation deviation: " << mutationDeviation << endl;
        cout << "beta: " << beta << endl;
        cout << "diversity rate: " << diversityRate << endl;
    }
};

#endif