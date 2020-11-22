#ifndef PARTICLE_SWARM_H
#define PARTICLE_SWARM_H

#include <iostream>
#include <string>
#include <vector>

#include "../NMR_Simulation/NMR_Simulation.h"

using namespace std;

class BestParticle
{
public:
    vector<double> position;
    double fitness;

    // methods
    BestParticle(){};

    // copy constructor
    BestParticle(const BestParticle &otherParticle)
    {
        this->position = otherParticle.position;
        this->fitness = otherParticle.fitness;
    }

    virtual ~BestParticle()
    {
        if (position.size() > 0)
            position.clear();
    }
};

class Particle
{
public:
    vector<double> position;
    vector<double> velocity;
    double fitness;
    BestParticle best;

    // methods
    Particle(){};
    Particle(uint _nvar)
    {
        position = vector<double>(_nvar);
        velocity = vector<double>(_nvar);
        this->best.position = vector<double>(_nvar);
    }

    // copy constructor
    Particle(const Particle &otherParticle)
    {
        this->position = otherParticle.position;
        this->velocity = otherParticle.velocity;
        this->fitness = otherParticle.fitness;
        this->best = otherParticle.best;
    }

    virtual ~Particle()
    {
        if (position.size() > 0)
            position.clear();
        if (velocity.size() > 0)
            velocity.clear();
    }
};

//
class PSO_population
{
public:
    vector<Particle> particles;

    // methods
    PSO_population(uint _popSize)
    {
        particles = vector<Particle>(_popSize);
    };

    // copy constructor
    PSO_population(const PSO_population &otherPop)
    {
        this->particles = otherPop.particles;
    };
    virtual ~PSO_population()
    {
        if (particles.size() > 0)
            particles.clear();
    }
};

class PSO_output
{
public:
    BestParticle best;
    vector<Particle> population;
    vector<double> bestFitness;

    // methods
    virtual ~PSO_output()
    {
        if (population.size() > 0)
            population.clear();

        if (bestFitness.size() > 0)
            bestFitness.clear();
    }
};

class PSO_problem
{

public:
    //NMR_Simulation NMR_obj;
    uint numberOfVariables;
    vector<double> minimumValue;
    vector<double> maximumValue;
    double (*fitnessFunction)(vector<double> &particle, NMR_Simulation &NMR);

    // methods
    PSO_problem(uint _nvar = 5) : numberOfVariables(_nvar)
    {
        minimumValue = vector<double>(_nvar);
        maximumValue = vector<double>(_nvar);
    };

    PSO_problem(uint _nvar,
                        double (*_fitnessFunction)(vector<double> &particle, NMR_Simulation &NMR)) : numberOfVariables(_nvar),
                                                                                                    fitnessFunction(_fitnessFunction)
    {
        minimumValue = vector<double>(_nvar);
        maximumValue = vector<double>(_nvar);
    };

    virtual ~PSO_problem()
    {
        if (minimumValue.size() > 0)
            minimumValue.clear();

        if (maximumValue.size() > 0)
            maximumValue.clear();
    }
};

class PSO_parameters
{

public:
    uint maxIterations;
    uint populationSize;
    double innertiaCoefficient;
    double innertiaDampingRatio;
    double personalAcceleration;
    double socialAcceleration;

    //methods

    PSO_parameters(){};
    virtual ~PSO_parameters(){};
};

class ParticleSwarm
{

public:
    PSO_problem problem;
    PSO_parameters parameters;

    ParticleSwarm(PSO_problem _problem, PSO_parameters _parameters) : problem(_problem),
                                                                              parameters(_parameters){};
    virtual ~ParticleSwarm(){};
    PSO_output run(NMR_Simulation &NMR);

    // private methods
    void applyBounds(vector<double> &attribute, vector<double> &minvalue, vector<double> &maxvalue);

    // basic function
    void printParticle(Particle &indvidual);
    void printState(vector<Particle> &population);
    double generateRandom(double minvalue, double maxvalue);
    double findMax(double &a, double &b);
    double findMin(double &a, double &b);
};

// app interface functions
void PSO_app(NMR_Simulation &NMR,
                      uint nvar = 5,
                      double (*NMRfitnessFunction)(vector<double> &, NMR_Simulation &) = NULL);

// some classic minimization functions
double sphere(vector<double> &position);
double distance(vector<double> &position);




#endif