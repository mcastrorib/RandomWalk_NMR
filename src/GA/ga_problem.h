#ifndef GA_PROBLEM_H
#define GA_PROBLEM_H

#include <iostream>
#include <string>
#include <vector>

#include "../NMR_Simulation/NMR_Simulation.h"
#include "ga_defs.h"

using namespace std;

class GA_problem
{

public:
    uint genotypeSize;
    vector<double> minimumValue;
    vector<double> maximumValue;
    double (*fitnessFunction)(vector<double> &individual, NMR_Simulation &NMR);

    // methods
    GA_problem()
    {
        genotypeSize = GA_GENOTYPE_SIZE;
        maximumValue = GA_MAX_VALUES;
        minimumValue = GA_MIN_VALUES;
    }

    GA_problem(double (*_fitnessFunction)(vector<double> &individual, NMR_Simulation &NMR)) : fitnessFunction(_fitnessFunction)
    {
        genotypeSize = GA_GENOTYPE_SIZE;
        maximumValue = GA_MAX_VALUES;
        minimumValue = GA_MIN_VALUES;
    };


    GA_problem(uint _nvar) : genotypeSize(_nvar)
    {
        minimumValue = vector<double>(_nvar);
        maximumValue = vector<double>(_nvar);        
    };

    GA_problem(uint _nvar,
               double (*_fitnessFunction)(vector<double> &individual, NMR_Simulation &NMR)) : genotypeSize(_nvar), 
                                                                                           fitnessFunction(_fitnessFunction)
    {
        minimumValue = vector<double>(_nvar);
        maximumValue = vector<double>(_nvar);
    };

    virtual ~GA_problem()
    {
        if (minimumValue.size() > 0)
            minimumValue.clear();

        if (maximumValue.size() > 0)
            maximumValue.clear();
    }

    void printInfo()
    {
        cout << "GA PROBLEM: " << endl;
        cout << "number of variables: " << genotypeSize << endl;
        cout << "max values: {"; printVector(maximumValue); cout << "}" << endl;
        cout << "min values: {"; printVector(minimumValue); cout << "}" << endl;   
    }

    void printVector(vector<double> &vec)
    {   
        for (uint i = 0; i < vec.size(); i++)
        {
            cout << vec[i] << " ";
        }
    }
};

#endif