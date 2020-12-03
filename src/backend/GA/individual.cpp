#include <iostream>
#include <vector>
#include <random>

#include <omp.h>
#include "individual.h"

using namespace std;

// Individual Class methods
void Individual::copyIndividual(const Individual &clone)
{
    if(this->genotype.size() > 0) this->genotype.clear();
    
    for(uint gene = 0; gene < clone.genotype.size(); gene++)
    {
        this->genotype.push_back(clone.genotype[gene]); 
    }
    this->fitness = clone.fitness;
}

void Individual::mutate(double mutationRatio, double mutationDeviation, vector<double> &varmin, vector<double> &varmax)
{

    uint size = this->genotype.size();
    vector<bool> flag(size);
    double random;

    // Identify Mutable Genes
    for (uint gene = 0; gene < size; gene++)
    {
        random = generateRandom(0.0, 1.0);
        flag[gene] = (random < mutationRatio);
    }

    // Apply Mutation
    for (uint gene = 0; gene < size; gene++)
    {
        if (flag[gene])
        {
            this->genotype[gene] += this->genotype[gene] * mutationDeviation * generateRandom(-1.0, 1.0);
        }
    }
}

double Individual::generateRandom(double minvalue, double maxvalue)
{
    int CPUfactor = 1;
    if(GA_OPENMP) CPUfactor += omp_get_thread_num();
    std::mt19937_64 myRNG;
    std::random_device device;    
    myRNG.seed(device() * CPUfactor * CPUfactor);
    std::uniform_real_distribution<double> double_dist;

    double range = maxvalue - minvalue;
    double random = (double_dist(myRNG) * range) + minvalue;
    return random;
}

void Individual::printInfo()
{
    cout << "ind: { ";
    for (uint var = 0; var < this->genotype.size(); var++)
    {
        cout << " " << this->genotype[var] << "  ";
    }
    cout << "}";
    cout << "\t fit: " << this->fitness << endl;
}

