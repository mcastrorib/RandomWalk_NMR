#ifndef MPI_GA_ISLAND_H
#define MPI_GA_ISLAND_H

#include <iostream>
#include <string>
#include <vector>

#include "../NMR_Simulation/NMR_Simulation.h"
#include "ga_defs.h"
#include "ga_core.h"
#include "ga_parameters.h"
#include "ga_problem.h"
#include "individual.h"
#include "population.h"
#include "ga_output.h"

using namespace std;

// typedef
struct mpi_ga_individual
{
     double genotype[GA_GENOTYPE_SIZE];
     double fitness;
};

class GA_Island
{
public:
	GeneticAlgorithm GA;
	int mpi_rank;
	int islands;    

    GA_Island(NMR_Simulation &_NMR, 
    		  uint _nvar, 
    		  int _mpi_rank = 0, 
    		  int _mpi_processes = 1,
    		  bool _logFlag = false);
    virtual ~GA_Island(){};

    void run(uint _generations);
    void runAsync(uint _generations);
    void migrate(uint _migrations);
    void migrateAsync(uint _migrations);
    void postWorkAsync(uint _migrations);
    void state();
    void times();
    void log(bool _userFlag = false);
    void results();
    void save();

    // get methods
    double getMigrationTime() { return this->migration_time; }
    double getNotificationTime() { return this->notification_time; }
    double getSendTime() { return this->send_individuals_time; }
    double getReceiveTime() { return this->receive_individuals_time; }

    // set simulation method from GA core class
    simulationMethod getImageBasedMethod()
    {
        return this->GA.getImageBasedMethod();
    }

    simulationMethod getHistogramBasedMethod()
    {
        return this->GA.getHistogramBasedMethod();
    }

    void setMethod(simulationMethod _method)
    {
        this->GA.setMethod(_method);
    }


private:
    double targetBestFit;
	double migrationRate;
	uint generationsPerMigration;
	MPI_Datatype MPI_GA_individual;	
    bool loggingFlag;
	double migration_time;
	double notification_time;
	double send_individuals_time;
	double receive_individuals_time;

	void sendBestIndividuals(uint _migrations);
	vector<Individual> receiveIndividuals(uint _migrations);
	void notifyIslandState(int _state);
	void sendStateNotification(int _state);
	void receiveStateNotifications(int _state);
	void insertIndividualsIntoGA(vector<Individual> &_immigrant, uint _migrations);

    // methods used for async migration
    bool checkIfSolutionWasFound();
    bool checkGAImprovement();
    bool checkMigrations();
    bool checkMigrationQuality(vector<Individual> &_immigrant);
    void acceptMigrations(vector<Individual> &_immigrant);

    void updateTargetBestFitness()
    {
        double improvement = GA_MIGRATION_IMPROVEMENT * (1.0 - this->GA.population.individuals[0].fitness);
        this->targetBestFit = this->GA.population.individuals[0].fitness + improvement;
        if(this->targetBestFit > 1.0) 
            this->targetBestFit = 1.0;
    }

	void updateMigrationTime()
	{
		this->migration_time = this->notification_time + 
							   this->send_individuals_time + 
							   this->receive_individuals_time;
	}

	void printTimes()
    {
    	cout << "----------------------- EXECTIME ------------------------" << endl;
        cout << "---------------------- GA EXECTIME ----------------------" << endl;
        cout << "initialization: \t" << this->GA.getInitTime() << " s" << endl;
        cout << "selection: \t\t" << this->GA.getSelectionTime() << " s" << endl;
        cout << "reproduction: \t\t" << this->GA.getReproductionTime() << " s" << endl;
        cout << "evaluation: \t\t" << this->GA.getEvaluationTime() << " s" << endl;
        cout << "diversification: \t" << this->GA.getDiversificationTime() << " s" << endl;
        cout << "save: \t\t\t" << this->GA.getSaveTime() << " s" << endl;
        cout << "runtime: \t\t" << this->GA.getRunTime() << " s" << endl;
        cout << "--------------------- MPI EXECTIME ----------------------" << endl;
        cout << "notifications: \t\t" << this->notification_time << " s" << endl;
        cout << "send individuals: \t" << this->send_individuals_time << " s" << endl;
        cout << "receive individuals: \t" << this->receive_individuals_time << " s" << endl;
        cout << "GA migration: \t\t" << this->migration_time << " s" << endl;
        cout << "---------------------------------------------------------" << endl;
        cout << "TOTAL: \t\t\t" << this->migration_time + this->GA.getRunTime() << " s" << endl;
        cout << "---------------------------------------------------------" << endl;
        cout << endl;
    }

    void createOutputFile();
    void writeResults();

};

#endif