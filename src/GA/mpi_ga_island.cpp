#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <vector>
#include <limits>
#include <random>
#include <algorithm>

// include MPI API for multiprocesses
#include <mpi.h>
#include <omp.h>

#include "mpi_ga_island.h"
#include "../NMR_Simulation/NMR_Simulation.h"
#include "ga_core.h"
#include "individual.h"

using namespace std;

GA_Island::GA_Island(NMR_Simulation &_NMR, 
    		  		 uint _nvar,
    		  		 int _mpi_rank, 
    		  		 int _mpi_processes,
    		  		 bool _loggingFlag) : GA(GeneticAlgorithm(_NMR, _nvar)), 
									  	  mpi_rank(_mpi_rank), 
									  	  islands(_mpi_processes), 
									  	  loggingFlag(_loggingFlag),
									  	  migration_time(0.0), 
									  	  notification_time(0.0), 
									 	  send_individuals_time(0.0), 
										  receive_individuals_time(0.0)

{
	this->generationsPerMigration = GA_GEN_PER_MIGRATION;	
	this->migrationRate = GA_MIGRATION_RATE;

	// create output file for results
	(*this).createOutputFile();

	// set mpi rank in GA object
	this->GA.setMPIRank(this->mpi_rank); 

	// commit mpi datatype
	uint individual_size = GA_GENOTYPE_SIZE + 1;
	MPI_Type_contiguous(individual_size, MPI_DOUBLE, &MPI_GA_individual);
	MPI_Type_commit(&MPI_GA_individual);
}

void GA_Island::state()
{
	cout << endl;
	cout << "Island " << this->mpi_rank << endl;
	this->GA.state();
}

void GA_Island::log(bool _userFlag)
{
	if(this->loggingFlag || _userFlag)
	{
		sleep(this->mpi_rank);
		(*this).state();
		(*this).times();
	}
}

void GA_Island::results()
{
	sleep(this->mpi_rank + 2);
	cout << "Island " << this->mpi_rank << endl;
	this->GA.times();
	(*this).times();
}

void GA_Island::run(uint _generations)
{
	 // run GA for a few generations
     uint migrations = (uint) (this->migrationRate * ((double) this->GA.population.individuals.size()));
     uint GA_completeRuns = _generations / this->generationsPerMigration;
     uint generationsInLastRun = _generations % this->generationsPerMigration;     

    if(this->mpi_rank == 0)
    {

    	int GA_lastRun = 0;
    	if(generationsInLastRun > 0) GA_lastRun++; 

     	cout << "total generations: \t\t" << _generations << endl;
        cout << "migrationRate: \t\t\t" << this->migrationRate << endl;
        cout << "individual migrations: \t\t" << migrations << endl;
        cout << "generations per migration: \t" << this->generationsPerMigration << endl;
        cout << "GA runs: \t\t\t" << GA_completeRuns + GA_lastRun << endl;
        cout << "generations in last run: \t" << generationsInLastRun << endl;
        cout << endl;
    }

     // complete runs with migrations
     for(uint run = 0; run < GA_completeRuns; run++)
     {
		// run GA evolution 
		this->GA.run(generationsPerMigration);

		// log GA state after run
		(*this).log();

		// migrate individuals between GA islands
		(*this).migrate(migrations);		
     } 

     // last run (no migration needed)
     if(generationsInLastRun != 0)
     {
          this->GA.run(generationsInLastRun);
          (*this).log();
     }
}

void GA_Island::runAsync(uint _generations)
{   
	(*this).notifyIslandState(GA_ASYNC_READY_TAG);		
	cout << "[" << this->mpi_rank << "]" << " ~ Ready to run GA async" << endl;

	uint migrations = (uint) (this->migrationRate * ((double) this->GA.population.individuals.size()));
	if(this->mpi_rank == 0)
    {
     	cout << endl;
     	cout << "---------------------------------------------------" << endl;
     	cout << "total generations: \t\t" << _generations << endl;
        cout << "individual migrations: \t\t" << migrations << endl;
        cout << "improvement per migration \t" << GA_MIGRATION_IMPROVEMENT << endl;
        cout << "---------------------------------------------------" << endl;
        cout << endl;
    }

    (*this).updateTargetBestFitness();
	cout << "[" << this->mpi_rank << "]" << " ~ initial target fitness is " << this->targetBestFit << endl; 

	uint run = 0; 
    while(run < _generations && !(*this).checkIfSolutionWasFound())
    {
   		// increment run count
    	run++;

    	// run and evolve one generation in GA  
		this->GA.run(1);

		// perform migration protocol 
		if((run % this->generationsPerMigration) == 0) 
			(*this).migrateAsync(migrations); 
    }

    // check if GA exited because solution was found
    if((*this).checkIfSolutionWasFound())
    {
    	// send individuals to neighbor GA islands asynchronously
    	cout << "[" << this->mpi_rank << "]" << " ~ solution was found, sending to neighbor islands" << endl;
		(*this).sendBestIndividuals(migrations);  
    }

    // wait for other island results
    (*this).postWorkAsync(migrations); 
}

bool GA_Island::checkIfSolutionWasFound()
{
	double currentBestFit = this->GA.getBestFitness();
	if((1.0 - currentBestFit) < GA_SOLUTION_TOLERANCE)
	{
		return true;
	} else
	{
		return false;
	}
}

bool GA_Island::checkGAImprovement()
{
	double currentBestFit = this->GA.getBestFitness();
	if(currentBestFit > this->targetBestFit)
	{
		return true;
	}
	return false;
}

bool GA_Island::checkMigrations()
{	
	int source = this->mpi_rank - 1; if(source < 0) source = this->islands - 1;
   	MPI_Status status;
   	int flag = 0;
	int tag = 0; // tag will identify the 1st migration
	MPI_Iprobe(source, tag, MPI_COMM_WORLD, &flag, &status);
   	
   	if(flag)
   		return true;
   	else
   		return false;
}

bool GA_Island::checkMigrationQuality(vector<Individual> &_immigrant)
{	
	if(_immigrant.size() > 0 && this->GA.population.individuals.size() > 0)
	{
		uint worstID = this->GA.population.individuals.size() - 1;
		if(_immigrant[0].fitness > this->GA.population.individuals[worstID].fitness)
			return true;
	}

	return false;
}

void GA_Island::acceptMigrations(vector<Individual> &_immigrant)
{	
	// add new individuals
	int populationSize = this->GA.population.individuals.size();
	int migrationSize = _immigrant.size();
	for(int id = 0; id < migrationSize; id++)
	{
		this->GA.population.individuals.push_back(_immigrant[id]);
	}

	// resort population after new additions
	this->GA.sortPopulation();

	// select best individuals after migrations
	this->GA.population.individuals.resize(populationSize);
}

void GA_Island::migrateAsync(uint _migrations)
{
	// check if GA population evolved beyond the target fitness
	if((*this).checkGAImprovement())
	{	
		cout << "[" << this->mpi_rank << "]" << " ~ GA population has improved, sending best individuals" << endl;

		// send individuals to neighbor GA islands asynchronously
		(*this).sendBestIndividuals(_migrations); // sending only one individual 

		// set new target fitness
		(*this).updateTargetBestFitness();
		cout << "[" << this->mpi_rank << "]" << " ~ new target fitness is " << this->targetBestFit << endl;	
	}

	// check if other islands have sent individuals
	if((*this).checkMigrations())
	{
		
		// receive individuals from neighbor GA islands asynchronously
		vector<Individual> immigrant(_migrations, Individual()); // receiving only one individual
		immigrant = (*this).receiveIndividuals(_migrations); // receiving only one individual

		// check if individuals are good enough // checar de trás pra frente hehe
		if((*this).checkMigrationQuality(immigrant))
		{
			cout << "[" << this->mpi_rank << "] ~ migration received and accepted" << endl;

			// log state before migration
			(*this).log(); 

			// accept new individuals in population
			(*this).acceptMigrations(immigrant);

			// log state after migration
			(*this).log(); 

			// save state after migration
			this->GA.save();

			// set new target fitness
			(*this).updateTargetBestFitness();
			cout << "[" << this->mpi_rank << "]" << " ~ new target fitness is " << this->targetBestFit << endl;
		}
		else
		{
			cout << "[" << this->mpi_rank << "] ~ migration received but rejected" << endl;
		}
	}
}

void GA_Island::postWorkAsync(uint _migrations)
{
	cout << "[" << this->mpi_rank << "]" << " ~ GA work is done, waiting for late migrations" << endl;	
	(*this).notifyIslandState(GA_ASYNC_DONE_TAG);		
	

	// check if other islands have sent individuals
	vector<Individual> immigrant(_migrations, Individual()); // receiving only one individual
	int postMigrations = 0;
	while((*this).checkMigrations())
	{
		// account number of migrations in the after work queue
		postMigrations++;
		
		// receive individuals from neighbor GA islands 		
		immigrant = (*this).receiveIndividuals(_migrations); // receiving only one individual

	}

	// check if there are late migrations to incorporate
	if(postMigrations > 0)
	{ 
		
		// check if individuals are good enough // checar de trás pra frente hehe
		if((*this).checkMigrationQuality(immigrant))
		{
			cout << "[" << this->mpi_rank << "] ~ "<< postMigrations 
			<< " late migration(s) received and accepted" << endl;

			// log state before migration
			(*this).log(); 

			// accept new individuals in population
			(*this).acceptMigrations(immigrant);

			// log state after migration
			(*this).log(); 

			// save state after migration
			this->GA.save();		
		}
		else
		{
			cout << "[" << this->mpi_rank << "] ~ "<< postMigrations 
			<< " late migration(s) received but rejected" << endl;
		}
	}
	else
	{
		cout << "[" << this->mpi_rank << "] ~ no late migration received." << endl;
	}
	
}


void GA_Island::migrate(uint _migrations)
{
	// Start MPI island migrations
	sleep(this->islands);

	cout << "[" << this->mpi_rank << "]" << " ~ starting mpi communication..." << endl;
	(*this).notifyIslandState(MIGRATION_START_TAG);          

	cout << "[" << this->mpi_rank << "]" << " ~ starting migrations..." << endl;
	(*this).sendBestIndividuals(_migrations);       

	cout << "[" << this->mpi_rank << "]" << " ~ individuals sent, now ready to receive.." << endl;
	(*this).notifyIslandState(MIGRATION_READY_TAG);		

	cout << "[" << this->mpi_rank<< "]" << " ~ everyone is ready to receive..." << endl;
	vector<Individual> immigrant(_migrations, Individual());
	immigrant = (*this).receiveIndividuals(_migrations);

	cout << "[" << this->mpi_rank << "]" << " ~ received individuals..." << endl;
	(*this).notifyIslandState(MIGRATION_END_TAG);		

	cout << "[" << this->mpi_rank << "]" << " ~ migration is finished." << endl;
	(*this).insertIndividualsIntoGA(immigrant, _migrations);

	// log GA state after migration
	(*this).log();

	// population needs to be resorted after migrations
	(*this).GA.sortPopulation();
}

void GA_Island::sendBestIndividuals(uint _migrations)
{
	// set MPI destination
	int destination = ((this->mpi_rank + 1) % this->islands);
	if(destination != this->mpi_rank)
	{	
		// collect time
		double time = omp_get_wtime();

		mpi_ga_individual genotype[_migrations];
		for(uint id = 0; id < _migrations; id++)
	  	{
	  		Individual migrant = this->GA.getIndividual(id);
	  
	  		for(uint gene = 0; gene < GA_GENOTYPE_SIZE; gene++)
	  		{
	  			genotype[id].genotype[gene] = migrant.getGene(gene);
	  		}
	  		genotype[id].fitness = migrant.getFitness();
	  
	  
	  	// MPI send	  	
  		int tag = id; // tag will identify the migration
  		int data_size = 1;
  		MPI_Status status;
  		MPI_Send(&genotype[id],
  				 data_size,
  				 MPI_GA_individual,
  				 destination,
  				 tag,
  				 MPI_COMM_WORLD);
		}

		this->send_individuals_time += omp_get_wtime() - time;
	}  
}

    


vector<Individual>  GA_Island::receiveIndividuals(uint _migrations)
{	
	// MPI receive          
	int source = this->mpi_rank - 1; if(source < 0) source = this->islands - 1;

	vector<Individual> immigrant(_migrations, Individual());
	if(source != this->mpi_rank)
	{	
		// collect time
		double time = omp_get_wtime();

		// initialize receiving indiduals
		mpi_ga_individual individual_received;
		for(uint id = 0; id < _migrations; id++)
		{
		   immigrant[id] = this->GA.createEmptyIndividual(); 
		}

		for(uint id = 0; id < _migrations; id++)
		{
		  	// MPI receive          
		   	int source = this->mpi_rank - 1; if(source < 0) source = this->islands - 1;
		   	int tag = id; // tag will identify the migration
		  	int data_size = 1;
		  	MPI_Status status;
		  	MPI_Recv(&individual_received,
		             data_size,
		             MPI_GA_individual,
		             source,
		             tag,
		             MPI_COMM_WORLD,
		             &status);
		                 
			for(uint gene = 0; gene < GA_GENOTYPE_SIZE; gene++)
			{
				immigrant[id].genotype[gene] = individual_received.genotype[gene];                    
			}
			immigrant[id].fitness = individual_received.fitness; 	           
		}

		this->receive_individuals_time += omp_get_wtime() - time;
		return immigrant;
	}

	// if no message was exchange, return an empty list
	immigrant.clear();
	return immigrant;
}

void GA_Island::notifyIslandState(int _state)
{
	double time = omp_get_wtime();
	(*this).sendStateNotification(_state);
	(*this).receiveStateNotifications(_state);
	this->notification_time += omp_get_wtime() - time;
}

void GA_Island::sendStateNotification(int _state)
{
	for(uint proc = 0; proc < this->islands; proc++)
    {
        if(proc != this->mpi_rank)
        {
            // MPI send
            int msg = 0;
            int destination = proc;
            int tag = _state; 
            int data_size = 1;
            MPI_Send(&msg,
                     data_size,
                     MPI_INT,
                     destination,
                     tag,
                     MPI_COMM_WORLD);
		}
    }
}


void GA_Island::receiveStateNotifications(int _state)
{
	// receive notification that other islands are ready to receive migrations
	for(uint proc = 0; proc < this->islands; proc++)
	{
		if(proc != this->mpi_rank)
		{
			// MPI send
			int msg = 0;
			int source = proc;
			int tag = _state; 
			int data_size = 1;
			MPI_Status status;
			MPI_Recv(&msg,
					 data_size,
					 MPI_INT,
					 source,
					 tag,
					 MPI_COMM_WORLD,
					 &status);
		}
	}
}

void GA_Island::insertIndividualsIntoGA(vector<Individual> &_immigrant, uint _migrations)
{
	double time = omp_get_wtime();

	for(uint id = 0; id < _immigrant.size(); id++)
	{
		this->GA.importIndividual(_immigrant[id], id);
	}

	this->receive_individuals_time += omp_get_wtime() - time;
}

void GA_Island::times()
{
	(*this).updateMigrationTime();
	(*this).printTimes();
}


void GA_Island::save()
{
    cout << "[" << this->mpi_rank << "]" << " ~ saving results" << endl;
    (*this).writeResults();
}

void GA_Island::writeResults()
{
    string filename = this->GA.NMR.simulationDirectory + "/NMR_GA_results.txt";
    ofstream file;
    file.open(filename, ios::app);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    // update time cronometer
    (*this).updateMigrationTime();

    // write in file
    file.precision(8);
    file << "----------------------- EXECTIME ------------------------" << endl;
    file << "---------------------- GA EXECTIME ----------------------" << endl;
    file << "initialization: \t" << this->GA.getInitTime() << " s" << endl;
    file << "selection: \t\t\t" << this->GA.getSelectionTime() << " s" << endl;
    file << "reproduction: \t\t" << this->GA.getReproductionTime() << " s" << endl;
    file << "evaluation: \t\t" << this->GA.getEvaluationTime() << " s" << endl;
    file << "diversification: \t" << this->GA.getDiversificationTime() << " s" << endl;
    file << "save: \t\t\t\t" << this->GA.getSaveTime() << " s" << endl;
    file << "GA runtime: \t\t" << this->GA.getRunTime() << " s" << endl;
    file << "--------------------- MPI EXECTIME ----------------------" << endl;
    file << "notifications: \t\t" << (*this).getNotificationTime() << " s" << endl;
    file << "send genotype: \t\t" << (*this).getSendTime() << " s" << endl;
    file << "receive genotype: \t" << (*this).getReceiveTime() << " s" << endl;
    file << "GA migration: \t\t" << (*this).getMigrationTime() << " s" << endl;
    file << "---------------------------------------------------------" << endl;
    file << "TOTAL: \t\t\t\t" << (*this).getMigrationTime() + this->GA.getRunTime() << " s" << endl;
    file << "---------------------------------------------------------" << endl;
    file << endl;

    file << "--------------------------------------------------" << endl;
    file.precision(5);
    file << "Current best: { ";
    for (uint var = 0; var < this->GA.population.individuals[0].genotype.size(); var++)
    {
        file << " " << this->GA.population.individuals[0].genotype[var] << ",  ";
    }
    file << "}"; file.precision(12);
    file << "\t fit: " << this->GA.population.individuals[0].fitness << endl;
    file << "--------------------------------------------------" << endl;

    file << "Historic best: { ";
    file.precision(5);
    for (uint var = 0; var < this->GA.bestIndividual.genotype.size(); var++)
    {
        file << " " << this->GA.bestIndividual.genotype[var] << "  ";
    }
    file << "}"; file.precision(12);
    file << "\t fit: " << this->GA.bestIndividual.fitness << endl;
    file << "--------------------------------------------------" << endl;
      

    file << endl;
    file.close();
}

void GA_Island::createOutputFile()
{
    string filename = this->GA.NMR.simulationDirectory + "/NMR_GA_results.txt";
    ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    file << "GA Results" << endl;

    file.close();
}