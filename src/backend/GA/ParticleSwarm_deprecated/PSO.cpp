#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <time.h>
#include <random>
#include <algorithm>

#include "PSO.h"

#include "../NMR_Simulation/NMR_Simulation.h"

using namespace std;

PSO_output ParticleSwarm::run(NMR_Simulation &NMR)
{
    
    // Problem
    double (*fitnessFunction)(vector<double> &, NMR_Simulation &) = this->problem.fitnessFunction;
    uint numberOfVariables = this->problem.numberOfVariables;
    vector<double> minimumValue = this->problem.minimumValue;
    vector<double> maximumValue = this->problem.maximumValue;

    // Parameters
    uint maxIterations = this->parameters.maxIterations;
    uint populationSize = this->parameters.populationSize;
    double w = this->parameters.innertiaCoefficient;
    double wdamp = this->parameters.innertiaDampingRatio;
    double c1 = this->parameters.personalAcceleration;
    double c2 = this->parameters.socialAcceleration;

    // Velocity Limits
    vector<double> maxVelocity(numberOfVariables);
    vector<double> minVelocity(numberOfVariables);
    for (uint var = 0; var < numberOfVariables; var++)
    {
        maxVelocity[var] = 0.2 * (maximumValue[var] - minimumValue[var]);
        minVelocity[var] = -maxVelocity[var];
    }

    // Initialization

    // Template for Empty Particles
    Particle empty_particle(numberOfVariables);
    for (uint i = 0; i < empty_particle.position.size(); i++)
    {
        empty_particle.position[i] = 0.0;
        empty_particle.velocity[i] = 0.0;
        empty_particle.best.position[i] = 0.0;
    }
    empty_particle.fitness = 0.0;
    empty_particle.best.fitness = 0.0;

    // Best Solution Role Model
    BestParticle globalBest;
    globalBest.position = vector<double>(numberOfVariables);
    globalBest.fitness = numeric_limits<double>::min();

    // Initializing Population
    vector<Particle> particle(populationSize);
    for (uint id = 0; id < particle.size(); id++)
    {
        particle[id] = empty_particle;

        // Generate random solution
        for (uint var = 0; var < numberOfVariables; var++)
        {
            particle[id].position[var] = generateRandom(minimumValue[var], maximumValue[var]);
        }

        // Initialize Velocity
        for (uint var = 0; var < numberOfVariables; var++)
        {
            particle[id].velocity[var] = 0.0;
        }

        // Evaluate random solution
        particle[id].fitness = fitnessFunction(particle[id].position, NMR);

        // Update Personal Best
        particle[id].best.position = particle[id].position;
        particle[id].best.fitness = particle[id].fitness;

        // Update Global Best
        if (particle[id].fitness > globalBest.fitness)
        {
            globalBest = particle[id].best;
        }
    }

    // Print State
    printState(particle);

    // Best Fitness At Each Iteration
    vector<double> bestFitness(maxIterations);

    // PSO Main Loop
    for (uint iteration = 0; iteration < maxIterations; iteration++)
    {

        // Advance Each Particle
        for (uint id = 0; id < particle.size(); id++)
        {

            // Update Velocity
            for (uint var = 0; var < numberOfVariables; var++)
            {
                double r1 = generateRandom(0, 1);
                double r2 = generateRandom(0, 1);
                double inertia = w * particle[id].velocity[var];
                double personalXP = c1 * r1 * (particle[id].best.position[var] - particle[id].position[var]);
                double socialXP = c1 * r1 * (globalBest.position[var] - particle[id].position[var]);
                particle[id].velocity[var] = inertia + personalXP + socialXP;
            }

            // Apply Velocity Limits
            applyBounds(particle[id].velocity, minVelocity, maxVelocity);

            // Update Position
            for (uint var = 0; var < numberOfVariables; var++)
            {
                particle[id].position[var] += particle[id].velocity[var];
            }

            // Apply Position Bounds
            applyBounds(particle[id].position, minimumValue, maximumValue);

            // Evaluation
            particle[id].fitness = fitnessFunction(particle[id].position, NMR);

            // Update Personal Best
            if (particle[id].fitness > particle[id].best.fitness)
            {
                particle[id].best.position = particle[id].position;
                particle[id].best.fitness = particle[id].fitness;

                // Update Global Best
                if (particle[id].best.fitness > globalBest.fitness)
                {
                    globalBest = particle[id].best;
                }
            }
        }

        // Store the Best Fitness Value
        bestFitness[iteration] = globalBest.fitness;

        // Show Iteration Info
        cout << "Iteration " << iteration << ": Best Fitness = " << bestFitness[iteration] << endl;

        // Damping Inertia Coefficient
        w = w * wdamp;

        // Print State
        printState(particle);
    }

    // Output
    PSO_output output;
    output.population = particle;
    output.best = globalBest;
    output.bestFitness = bestFitness;

    return output;
}

void ParticleSwarm::applyBounds(vector<double> &attribute, vector<double> &minvalue, vector<double> &maxvalue)
{
    for (uint var = 0; var < minvalue.size(); var++)
    {
        attribute[var] = findMax(attribute[var], minvalue[var]);
        attribute[var] = findMin(attribute[var], maxvalue[var]);
    }
}

