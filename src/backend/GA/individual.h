#ifndef GA_INDIVIDUAL_H
#define GA_INDIVIDUAL_H

#include <vector>
#include "ga_defs.h"

using namespace std;

class Individual
{
public:
    vector<double> genotype;
    double fitness;

    // methods
    Individual(){};
    Individual(uint _nvar)
    {
        genotype = vector<double>(_nvar);
    }

    // copy constructor
    Individual(const Individual &otherIndividual)
    {
        this->genotype = otherIndividual.genotype;
        this->fitness = otherIndividual.fitness;
    }

    virtual ~Individual()
    {
        if (genotype.size() > 0)
            genotype.clear();
    }

    // mutate method
    void copyIndividual(const Individual &clone);
    void mutate(double mutationRatio, double mutationDeviation, vector<double> &varmin, vector<double> &varmax);
    double generateRandom(double minvalue, double maxvalue);
    void printInfo();

    // get methods
    double getGene(uint i)
    {
        return this->genotype[i];
    }
    double getFitness()
    {
        return this->fitness;
    }
};

#endif