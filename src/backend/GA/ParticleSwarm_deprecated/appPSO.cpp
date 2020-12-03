#include <iostream>
#include <string>
#include <vector>
#include <math.h>

#include "PSO.h"

#include "../NMR_Simulation/NMR_Simulation.h"

using namespace std;

void PSO_app(NMR_Simulation &NMR, uint nvar, double (*NMRfitnessFunction)(vector<double> &, NMR_Simulation &))
{
    // Problem Definition
    PSO_problem problem(nvar);

    //problem.NMR_obj = NMR;
    problem.fitnessFunction = NMRfitnessFunction;
    problem.maximumValue = {100.0, 100.0, 1.0, 100.0, 100.0, 100.0, 1.0, 100.0};
    problem.minimumValue = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // PSO Parameters

    // Constriction Coefficients
    double kappa = 1;
    double phi1 = 2.05;
    double phi2 = 2.05;
    double phi = phi1 + phi2;
    double chi = (2 * kappa) / abs(2 - phi - sqrt(phi * phi - 4 * phi));

    PSO_parameters parameters;
    parameters.maxIterations = 100;
    parameters.populationSize = 50;
    parameters.innertiaCoefficient = chi;
    parameters.innertiaDampingRatio = 0.95;
    parameters.personalAcceleration = chi * phi1;
    parameters.socialAcceleration = chi * phi2;

    // PSO Object
    ParticleSwarm PSO(problem, parameters);

    // Run PSO
    PSO_output output = PSO.run(NMR);

    // Results;
    cout << "Best individual is: " << endl;
    cout << "{";
    for (uint var = 0; var < output.best.position.size(); var++)
    {
        cout << " " << output.best.position[var] << ", ";
    }
    cout << "}";
    cout << "\t fitness: " << output.best.fitness << endl;
}

