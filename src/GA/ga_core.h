#ifndef GA_CORE_H
#define GA_CORE_H

#include <iostream>
#include <string>
#include <vector>

#include "../NMR_Simulation/NMR_Simulation.h"
#include "ga_defs.h"
#include "ga_parameters.h"
#include "ga_problem.h"
#include "individual.h"
#include "population.h"
#include "ga_output.h"

using namespace std;

struct Offspring
{
    Individual c1;
    Individual c2;
};

struct ParentsIndex
{
    uint p1;
    uint p2;
};

class GeneticAlgorithm
{

public:
    // double (*fitnessFunction)(vector<double> &, NMR_Simulation &);
    Population population;
    uint populationSize;
    vector<Offspring> offspring;
    Individual bestIndividual;
    NMR_Simulation &NMR;
    simulationMethod RW_method;
    

    // Output
    GA_output output;

    GeneticAlgorithm(NMR_Simulation &_NMR, uint _nvar);
    virtual ~GeneticAlgorithm()
    {
        // delete[] NMR;
        // NMR = NULL;
    };
    
    void readGAProblem(void);
    void readGAProblem(GA_problem &_problem);
    void readGAParameters(void);
    void readGAParameters(GA_parameters &_parameters);
    
    void run(uint _generations = 1);
    void save();

    // migration
    void importIndividual(Individual &_individual, uint _rank);
    Individual createEmptyIndividual();
    Individual getIndividual(uint _id);

    void results();
    void state();
    void times();
    void sortPopulation();

    // get methods for private attributes
    double getBestFitness() { return this->population.individuals[0].fitness; }
    double getRunTime(){ return this->run_time;}
    double getInitTime(){ return this->init_time; }
    double getSelectionTime(){ return this->selection_time; }
    double getReproductionTime(){ return this->reproduction_time; }
    double getEvaluationTime(){ return this->evaluation_time; }
    double getDiversificationTime(){ return this->diversity_time; }
    double getSaveTime() { return this->save_time; }
    simulationMethod getSimulationMethod() { return this->RW_method; }
    simulationMethod getImageBasedMethod();
    simulationMethod getHistogramBasedMethod();
    void setMethod(simulationMethod _method) { this->RW_method = _method; }

    // set mpi rank
    void setMPIRank(int _rank){ this->mpi_rank = _rank; }


    // private attributes and methods
private:
    // Optimization problem
    uint genotypeSize;
    vector<double> minimumValue;
    vector<double> maximumValue;

    // GA Parameters
    double offspringProportion;
    uint numberOfOffsprings;
    double beta;
    double gamma;
    double mutationRatio;
    double mutationDeviation;
    double diversityRate;
    double genotypeDeviation;
    bool incestuous;
    int mutationsImposed;

    uint currentGeneration;
    double run_time;
    double init_time;
    double selection_time;
    double reproduction_time;
    double evaluation_time;
    double diversity_time;
    double save_time;

    // mpi rank
    int mpi_rank;

    // private GA methods
    void initPopulation();
    void selection();
    void reproduction();
    void diversify();
    void writeState();

    // initialization
    void initBestIndividual();
    Population createNewPopulation();

    // reproduction
    void createProbabilitiesVector(vector<double> &_probabilities);
    void createParentsList(vector<ParentsIndex> &_parents, vector<double> &_probabilities);    
    void createOffspring(vector<ParentsIndex> &_parents);
    void applyMutation();
    void applyBounds();
    void evaluateOffspring();
    void evaluateFitness(Individual &_individual); 
    double fitNMRT2(vector<double> &_sigmoid);
    void runSimulation();

    // selection
    // void sortPopulation();
    void updateBest();
    void createNextGeneration();

    // diversify
    void checkGenotypeDeviation();
    void updateIncestuousity();
    void applyGeneDiversification();
    void resetPopulation();
    void imposeMutation();

    // // increment generation 
    void incrementGeneration()
    {           
        this->currentGeneration++;
    }

    // low-level reproduction
    void fillProbabilitiesArray(vector<double> &probabilities);
    uint rouletteWheelSelection(vector<double> &p);
    Offspring applyCrossover(const Individual &parent1, const Individual &parent2);
    void applyBounds(Individual &child, vector<double> &minvalue, vector<double> &maxvalue);

    // output file creation
    void createOutputFile();

    // basic print/debug methods
    void printParametersInfo()
    {
        cout << "GA PARAMETERS: " << endl;
        cout << "population size: " << populationSize << endl;
        cout << "offspring proportion: " << offspringProportion << endl;
        cout << "number of offsprings: " << numberOfOffsprings << endl;
        cout << "gamma: " << gamma << endl;
        cout << "mutation ratio: " << mutationRatio << endl;
        cout << "mutation deviation: " << mutationDeviation << endl;
        cout << "beta: " << beta << endl;
        cout << "diversity rate: " << diversityRate << endl;
    };

    void printProblemInfo()
    {
        cout << "GA PROBLEM: " << endl;
        cout << "number of variables: " << genotypeSize << endl;
        cout << "max values: {"; printVector(maximumValue); cout << "}" << endl;
        cout << "min values: {"; printVector(minimumValue); cout << "}" << endl;   
    }

    void printBest()
    {
        // Results;
        cout << "Current best individual: " << endl;
        this->population.individuals[0].printInfo();
    }

    void printGlobalBest()
    {
        // Results;
        cout << "Best individual: " << endl;
        this->bestIndividual.printInfo();
    }

    void printVector(vector<double> &vec)
    {
        for (uint i = 0; i < vec.size(); i++)
            {
                cout << vec[i] << " ";
            }
    }

    void printVector(vector<ParentsIndex> &couples)
    {
        for (uint i = 0; i < couples.size(); i++)
            {
                cout << "[" << couples[i].p1 << " , " << couples[i].p2 << "]";
            }
    }

    void printPopulation()
    {
        for (uint id = 0; id < population.individuals.size(); id++)
        {
            cout << "#" << id << "\t";
            population.individuals[id].printInfo();
        }
    }

    void printPopulation(vector<Individual> &population)
    {
        for (uint id = 0; id < population.size(); id++)
        {
            cout << "#" << id << " ";
            population[id].printInfo();
        }
    }

    void printOffspring(vector<Offspring> &offspring)
    {
        for(uint id = 0; id < offspring.size(); id++)
        {
            cout << "CHILDREN(" << id << "):" << endl;
            offspring[id].c1.printInfo();
            offspring[id].c2.printInfo();
        }
    }

    void printGeneration()
    {
        cout << "[" << this->mpi_rank << "]" << " Generation " << this->currentGeneration << ":" << endl;
    }

    void printState()
    {
        // cout << endl;
        (*this).printGeneration();
        (*this).printPopulation();
        (*this).printBest();
        cout << endl;
    }

    void updateRunTime()
    {
        this->run_time = this->init_time + 
                         this->selection_time +
                         this->reproduction_time +
                         this->evaluation_time +
                         this->diversity_time +
                         this->save_time;
    }

    void printTimes()
    {
        cout << "---------------------- GA EXECTIME ----------------------" << endl;
        cout << "initialization: \t" << init_time << " s" << endl;
        cout << "selection: \t\t" << selection_time << " s" << endl;
        cout << "reproduction: \t\t" << reproduction_time << " s" << endl;
        cout << "evaluation: \t\t" << evaluation_time << " s" << endl;
        cout << "diversification: \t" << diversity_time << " s" << endl;
        cout << "save: \t\t\t" << save_time << " s" << endl;
        cout << "runtime: \t\t" << run_time << " s" << endl;
        cout << "---------------------------------------------------------" << endl;
        cout << endl;
    }
    
    // basic vec/math functions
    double generateRandom(double minvalue, double maxvalue);
    int generateRandom(int minvalue, int maxvalue);
    double findMean(vector<double> &array);
    double findStdDev(vector<double> &array);
    double sum_elements(vector<double> &array);
    vector<double> cumsum(vector<double> &array);
    uint findIndex(double x, vector<double> &array);
    double findMax(double &a, double &b);
    double findMin(double &a, double &b);
    void get_multi_thread_loop_limits(const int tid, 
                                      const int num_threads, 
                                      const int num_loops, 
                                      int &start, 
                                      int &finish);
};

#endif