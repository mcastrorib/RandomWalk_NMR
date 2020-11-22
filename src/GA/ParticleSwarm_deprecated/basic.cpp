#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <time.h>
#include <random>

#include "PSO.h"

using namespace std;

// basic functions
// basic functions
void ParticleSwarm::printParticle(Particle &_particle)
{
    cout << "ind: {";
    for (uint var = 0; var < _particle.position.size(); var++)
    {
        cout << " " << _particle.position[var] << ", ";
    }
    cout << "}";
    cout << "\t fitness: " << _particle.fitness << endl;
}

void ParticleSwarm::printState(vector<Particle> &population)
{
    cout << "population state:" << endl;

    for (uint id = 0; id < population.size(); id++)
    {
        cout << "ind " << id << ": {";
        for (uint var = 0; var < population[id].position.size(); var++)
        {
            cout << " " << population[id].position[var] << ", ";
        }
        cout << "}";
        cout << "\t fit: " << population[id].fitness << endl;
    }
    cout << endl
         << endl;
}

double ParticleSwarm::generateRandom(double minvalue, double maxvalue)
{

    std::mt19937_64 myRNG;
    std::random_device device;
    myRNG.seed(device());
    std::uniform_real_distribution<double> double_dist;

    double range = maxvalue - minvalue;
    double random = (double_dist(myRNG) * range) + minvalue;
    return random;
}

double ParticleSwarm::findMax(double &a, double &b)
{
    if (a > b)
        return a;
    else
        return b;
}

double ParticleSwarm::findMin(double &a, double &b)
{
    if (a < b)
        return a;
    else
        return b;
}
