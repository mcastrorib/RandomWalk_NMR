#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <limits>
#include <random>
#include <algorithm>

#include <omp.h>

#include "../NMR_Simulation/NMR_Simulation.h"
#include "ga_defs.h"
#include "ga_core.h"


using namespace std;

GeneticAlgorithm::GeneticAlgorithm(NMR_Simulation &_NMR,     
                                   uint _nvar) : NMR(_NMR), 
                                                 RW_method(HistogramBased),
                                                 population(Population()), 
                                                 currentGeneration(0),
                                                 incestuous(false),
                                                 mutationsImposed(0),
                                                 run_time(0.0),
                                                 init_time(0.0),
                                                 selection_time(0.0),
                                                 reproduction_time(0.0),
                                                 evaluation_time(0.0),
                                                 diversity_time(0.0),
                                                 save_time(0.0),
                                                 mpi_rank(0)
{
    // Initialize offspring vector
    vector<Offspring> offspring();

    // Set GA opt problem 
    (*this).readGAProblem();

    // Set GA parameters
    (*this).readGAParameters();

    // Initialize a new population
    (*this).initPopulation();

    // create empty output file
    if(GA_SAVE_MODE) 
    {
        (*this).createOutputFile();
        (*this).save();
    }
}

void GeneticAlgorithm::readGAProblem()
{
    // Problem definitions
    GA_problem problem;

    this->genotypeSize = problem.genotypeSize;
    this->minimumValue = problem.minimumValue;
    this->maximumValue = problem.maximumValue;
}

void GeneticAlgorithm::readGAProblem(GA_problem &_problem)
{
    // Problem definitions
    this->genotypeSize = _problem.genotypeSize;
    this->minimumValue = _problem.minimumValue;
    this->maximumValue = _problem.maximumValue;
}

void GeneticAlgorithm::readGAParameters()
{
    // Parameters
    GA_parameters parameters;

    this->populationSize = parameters.populationSize;
    this->offspringProportion = parameters.offspringProportion;
    this->numberOfOffsprings = (offspringProportion * (populationSize / 2)) * 2;
    this->beta = parameters.beta;
    this->gamma = parameters.gamma;
    this->mutationRatio = parameters.mutationRatio;
    this->mutationDeviation = parameters.mutationDeviation;
    this->diversityRate = parameters.diversityRate;
}

void GeneticAlgorithm::readGAParameters(GA_parameters &_parameters)
{
    // Parameters
    this->populationSize = _parameters.populationSize;
    this->offspringProportion = _parameters.offspringProportion;
    this->numberOfOffsprings = (offspringProportion * (populationSize / 2)) * 2;
    this->beta = _parameters.beta;
    this->gamma = _parameters.gamma;
    this->mutationRatio = _parameters.mutationRatio;
    this->mutationDeviation = _parameters.mutationDeviation;
    this->diversityRate = _parameters.diversityRate;
}

void GeneticAlgorithm::initPopulation()
{
    double time = omp_get_wtime();
    this->population = createNewPopulation();
    (*this).initBestIndividual();
    (*this).sortPopulation();
    this->init_time += omp_get_wtime() - time;
}

void GeneticAlgorithm::run(uint _generations)
{
    // GA main loop
    for(uint i = 0; i < _generations; i++)
    {
        // increment generation            
        (*this).incrementGeneration();
        (*this).printGeneration();

        // population reproduction and selection
        (*this).reproduction();
        (*this).selection();

        // diversify genotype
        (*this).diversify(); 

        // save state in disc
        (*this).save();
    }

    (*this).updateRunTime();
}

void GeneticAlgorithm::reproduction()
{
    double time = omp_get_wtime();  
    vector<double> probabilities;
    (*this).createProbabilitiesVector(probabilities);

    vector<ParentsIndex> parentsList;
    (*this).createParentsList(parentsList, probabilities);
    (*this).createOffspring(parentsList);   
    (*this).applyMutation();
    (*this).applyBounds();
    this->reproduction_time += omp_get_wtime() - time;
    
    time = omp_get_wtime();
    (*this).evaluateOffspring(); 
    this->evaluation_time += omp_get_wtime() - time;
    
}

void GeneticAlgorithm::selection()
{
    double time = omp_get_wtime();
    (*this).createNextGeneration();
    (*this).sortPopulation();
    this->selection_time += omp_get_wtime() - time;
}

void GeneticAlgorithm::diversify()
{
    double time = omp_get_wtime();
    (*this).checkGenotypeDeviation();
    (*this).applyGeneDiversification();
    this->diversity_time += omp_get_wtime() - time;
}

Individual GeneticAlgorithm::createEmptyIndividual()
{
    // // Template for Empty Individuals
    Individual empty_individual(this->genotypeSize);
    for (uint gene = 0; gene < empty_individual.genotype.size(); gene++)
    {
        empty_individual.genotype[gene] = 0.0;
    }
    empty_individual.fitness = 0.0;

    return empty_individual;
}


Population GeneticAlgorithm::createNewPopulation()
{
    double time;
    Individual empty_individual = createEmptyIndividual();
    Population newPopulation(populationSize, genotypeSize); 

    for (uint id = 0; id < newPopulation.individuals.size(); id++)
    {
        
        newPopulation.individuals[id] = empty_individual;

        // Generate random genotype
        for (uint gene = 0; gene < genotypeSize; gene++)
        {
            newPopulation.individuals[id].genotype[gene] = generateRandom(minimumValue[gene], maximumValue[gene]);
        }

        // Evaluate random genotype
        time = omp_get_wtime();
        (*this).evaluateFitness(newPopulation.individuals[id]);
        time = omp_get_wtime() - time;

        // discount time related with fitness evalutions
        this->evaluation_time += time;
        this->init_time -= time;       
    }

    return newPopulation;
}


void GeneticAlgorithm::initBestIndividual()
{
    Individual empty_individual = createEmptyIndividual();
    this->bestIndividual = empty_individual;
    this->bestIndividual.fitness = numeric_limits<double>::min();
}


void GeneticAlgorithm::updateBest()
{
    if(this->bestIndividual.fitness < population.individuals[0].fitness)
    {
        this->bestIndividual = population.individuals[0];
    }
}



void GeneticAlgorithm::createProbabilitiesVector(vector<double> &_probabilities)
{
    // Probabilities Of Parents Selection Array
    for (uint i = 0; i < populationSize; i++)
    {
        _probabilities.push_back(population.individuals[i].fitness);
    }
    double sum = sum_elements(_probabilities);
    if (sum != 0)
    {
        for (uint i = 0; i < populationSize; i++)
        {
            _probabilities[i] = _probabilities[i] / sum;
        }
    }

}

void GeneticAlgorithm::createParentsList(vector<ParentsIndex> &_parents, vector<double> &_probabilities)
{

    const int num_cpu_threads = omp_get_max_threads();
    const int numberOfCouples = (numberOfOffsprings / 2);
    int c_start, c_finish;
    ParentsIndex newCouple;

    #pragma omp parallel if(GA_OPENMP) shared(_probabilities, _parents) private(c_start, c_finish, newCouple) 
    {
        const int thread_id = omp_get_thread_num();
                

        // set thread loop limits based in the thread id 
        if(GA_OPENMP)
        { 
            get_multi_thread_loop_limits(thread_id, 
                                         num_cpu_threads, 
                                         numberOfCouples, 
                                         c_start, 
                                         c_finish);        
        } 
        else    // if GA_OPENMP == false, it uses a single thread 
        {
            get_multi_thread_loop_limits(thread_id, 
                                         1, 
                                         numberOfCouples, 
                                         c_start, 
                                         c_finish);
        }

        for (int couple = c_start; couple < c_finish; couple++)
        {             
            // Select parents couple     
            newCouple.p1 = rouletteWheelSelection(_probabilities);
            do
            {
                newCouple.p2 = rouletteWheelSelection(_probabilities);
            }while(newCouple.p2 == newCouple.p1);
    
            // add to parents list
            if(GA_OPENMP)
            {
                #pragma omp critical 
                {
                    _parents.push_back(newCouple);
                }
            } else
            {
                _parents.push_back(newCouple);
            }
        }
    }
}

void GeneticAlgorithm::createOffspring(vector<ParentsIndex> &_parents)
{
    // clear last offspring
    if(this->offspring.size() > 0) this->offspring.clear();
    const int num_cpu_threads = omp_get_max_threads();
    const int size = this->numberOfOffsprings / 2;
    int id_start, id_finish;

    #pragma omp parallel if(GA_OPENMP) shared(_parents) private(id_start, id_finish) 
    {
        const int thread_id = omp_get_thread_num();        

        // set thread loop limits based in the thread id 
        if(GA_OPENMP) 
        {
            get_multi_thread_loop_limits(thread_id, 
                                         num_cpu_threads, 
                                         size, 
                                         id_start, 
                                         id_finish);
        }
        else    // if GA_OPENMP == false, it uses a single thread
        {
            get_multi_thread_loop_limits(thread_id, 
                                         1, 
                                         size, 
                                         id_start, 
                                         id_finish);
        }

        for (int id = id_start; id < id_finish; id++)
        {
            // Define 1st and 2nd parents
            const Individual parent1 = population.individuals[_parents[id].p1];
            const Individual parent2 = population.individuals[_parents[id].p2];        

            // Perform Crossover
            Offspring children = (*this).applyCrossover(parent1, parent2);

            // Add offspring to Offspring population
            if(GA_OPENMP)
            {
                #pragma omp critical 
                {
                    this->offspring.push_back(children);
                }
            } else
            {
                this->offspring.push_back(children);
            }
            
        }
    }
}

void GeneticAlgorithm::applyMutation()
{
    const int num_cpu_threads = omp_get_max_threads();
    const int offspring_size = this->offspring.size();
    int id_start, id_finish;

    #pragma omp parallel if(GA_OPENMP) shared(offspring) private(id_start, id_finish)  
    {
        const int thread_id = omp_get_thread_num();        

        // set thread loop limits based in the thread id 
        if(GA_OPENMP) 
        {
            get_multi_thread_loop_limits(thread_id, 
                                         num_cpu_threads, 
                                         offspring_size, 
                                         id_start, 
                                         id_finish);
        }
        else    // if GA_OPENMP == false, it uses a single thread
        {
            get_multi_thread_loop_limits(thread_id, 
                                         1, 
                                         offspring_size, 
                                         id_start, 
                                         id_finish);
        }

        for (int id = id_start; id < id_finish; id++)
        {
            // Perform Mutation on 1st child
            this->offspring[id].c1.mutate(this->mutationRatio, 
                                          this->mutationDeviation, 
                                          this->minimumValue, 
                                          this->maximumValue);

            // Perform Mutation on 2nd child
            this->offspring[id].c2.mutate(this->mutationRatio, 
                                          this->mutationDeviation, 
                                          this->minimumValue, 
                                          this->maximumValue);
        }
    }
}

void GeneticAlgorithm::applyBounds()
{   
    const int num_cpu_threads = omp_get_max_threads();
    const int offspring_size = this->offspring.size();
    int id_start, id_finish;

    #pragma omp parallel if(GA_OPENMP) shared(offspring) private(id_start, id_finish) 
    {
        const int thread_id = omp_get_thread_num();
        

        // set thread loop limits based in the thread id 
        if(GA_OPENMP) 
        {
            get_multi_thread_loop_limits(thread_id, 
                                         num_cpu_threads, 
                                         offspring_size, 
                                         id_start, 
                                         id_finish);
        }
        else    // if GA_OPENMP == false, it uses a single thread
        {
            get_multi_thread_loop_limits(thread_id, 
                                         1, 
                                         offspring_size, 
                                         id_start, 
                                         id_finish);
        }

        for (int id = id_start; id < id_finish; id++)
        {
            // Apply Bounds
            (*this).applyBounds(this->offspring[id].c1, this->minimumValue, this->maximumValue);
            (*this).applyBounds(this->offspring[id].c2, this->minimumValue, this->maximumValue);
        }
    }
}

void GeneticAlgorithm::evaluateOffspring()
{
   for (uint id = 0; id < this->offspring.size(); id++)
    {
        // Evaluate First Offspring
        (*this).evaluateFitness(this->offspring[id].c1);
        if (this->offspring[id].c1.fitness > this->bestIndividual.fitness)
            this->bestIndividual = this->offspring[id].c1;

        // Evaluate Second Offspring
        (*this).evaluateFitness(this->offspring[id].c2);
        if (this->offspring[id].c2.fitness > this->bestIndividual.fitness)
            this->bestIndividual = this->offspring[id].c2;
    } 
}

void GeneticAlgorithm::evaluateFitness(Individual &_individual)
{
    _individual.fitness = fitNMRT2(_individual.genotype);       
}

double GeneticAlgorithm::fitNMRT2(vector<double> &sigmoid)
{
    // Update walkers superficial relaxativity from the candidate solution vector
    if(this->RW_method == ImageBased) 
        this->NMR.updateWalkersRelaxativity(sigmoid);

    // Create new histogram' penalties vector
    if(this->RW_method == HistogramBased) 
        this->NMR.createPenaltiesVector(sigmoid);

    // Perform walk simulation to get Global Energy Decay  
    (*this).runSimulation();

    // Perform Laplace Inversion
    this->NMR.applyLaplaceInversion();

    // Find correlation value between simulated and input T2 distributions
    // this value is elevated to Nth potency so that correlation intervals can be amplified
    // this tactic may benefit the search heuristics adopted - it is optional
    double correlation = this->NMR.leastSquaresT2();
    correlation *= correlation;

    // reset GE and T2 vectors for next simulations
    // this->NMR.resetGlobalEnergy();
    // this->NMR.resetT2Distribution();

    return correlation;
}

void GeneticAlgorithm::runSimulation()
{
    if(this->RW_method == ImageBased)
    {
        cout << "[imgbsd] ";
        this->NMR.walkSimulation();
        return;
    }

    if(this->RW_method == HistogramBased)
    {
        cout << "[hstbsd] ";
        this->NMR.fastSimulation();
        return;
    }
}


void GeneticAlgorithm::sortPopulation()
{
    // Sort island individuals by fitness
    sort(population.individuals.begin(), 
         population.individuals.end(), 
         [](Individual const &a, Individual &b) 
         { return a.fitness > b.fitness; });

    // update best individual
    (*this).updateBest();
}

void GeneticAlgorithm::createNextGeneration()
{
    // Select best individuals in population
    population.individuals.resize(this->population.individuals.size() - (2 * this->offspring.size()));

    // Merge offspring with selected population
    for(uint id = 0; id < this->offspring.size(); id++)
    {
        population.individuals.push_back(this->offspring[id].c1);
        population.individuals.push_back(this->offspring[id].c2); 
    }
}

void GeneticAlgorithm::checkGenotypeDeviation()
{
    double dev;
    double variation;
    double norm_dev = 0.0;

    // find standard deviation for each gene
    for(uint gene = 0; gene < this->genotypeSize; gene++)
    {
        vector<double> genes;

        // collect specific gene in population
        for(uint id = 0; id < this->population.individuals.size(); id++)
        {    
            genes.push_back(this->population.individuals[id].genotype[gene]);    
        }

        if(GA_MEAN_DEVIATION)  
            variation = (*this).findMean(genes);
        else 
            variation = this->maximumValue[gene] - this->minimumValue[gene];

        dev = (*this).findStdDev(genes);
        norm_dev += (dev / variation);
    }

    // find mean gene deviation in population
    this->genotypeDeviation = norm_dev / ((double) this->genotypeSize);    
    (*this).updateIncestuousity();
}

void GeneticAlgorithm::updateIncestuousity()
{
    double threshold = 1.0 - this->diversityRate;
    if(this->genotypeDeviation < threshold) 
    {    
        this->incestuous = true;
    }
}

void GeneticAlgorithm::applyGeneDiversification()
{
    if(this->incestuous)
    {
        cout << "[" << this->mpi_rank << "]" << " ~ mean dev is " << this->genotypeDeviation << endl;
        cout << "[" << this->mpi_rank << "]" << " ~ individuals are too similar... ";
        if(this->mutationsImposed == GA_MUTATIONS_PER_RESET)
        {
            cout << "reseting population." << endl;
            (*this).resetPopulation();
        } else
        {
            cout << "imposing mutation." << endl;
            (*this).imposeMutation();
        }
    
    (*this).sortPopulation();
    this->incestuous = false;
    }

}

void GeneticAlgorithm::resetPopulation()
{    
    // Select best individuals in population
    int top = (int) ceil(GA_TOP_SIZE * this->populationSize);
    int reset_size = this->populationSize - top;
    population.individuals.resize(top);
    
    for (uint id = 0; id < reset_size; id++)
    {
        
        Individual newIndividual = createEmptyIndividual();

        // Generate random solution
        for (uint gene = 0; gene < this->genotypeSize; gene++)
        {
            newIndividual.genotype[gene] = generateRandom(minimumValue[gene], maximumValue[gene]);
        }

        // Evaluate new individual
        double time = omp_get_wtime();
        (*this).evaluateFitness(newIndividual);
        time = omp_get_wtime() - time;

        // discount time related with fitness evalutions
        this->evaluation_time += time;
        this->diversity_time -= time; 
        
        // Add new individual to population
        population.individuals.push_back(newIndividual);
    }

    // reset mutations counter
    this->mutationsImposed = 0;
}

void GeneticAlgorithm::imposeMutation()
{
    // Select individuals in population to apply mutation
    int id_start = 1;
    int id_end = this->population.individuals.size() / 2;

    for (uint id = id_start; id < id_end; id++)
    {
        this->population.individuals[id].mutate(1.0, /* 1.0 value imposes mutation for every gene */
                                                this->mutationDeviation, 
                                                this->minimumValue, 
                                                this->maximumValue);

        // Evaluate mutant individual
        double time = omp_get_wtime();
        (*this).evaluateFitness(this->population.individuals[id]);
        time = omp_get_wtime() - time;

        // discount time related with fitness evalutions
        this->evaluation_time += time;
        this->diversity_time -= time; 
    }

    // increment mutations counter
    this->mutationsImposed++;
}

void GeneticAlgorithm::importIndividual(Individual &_individual, uint _rank)
{
    if(!(this->population.individuals.size() < _rank))
    {
        this->population.individuals[_rank] = _individual;
    } else
    {
        cout << "[" << this->mpi_rank << "]" << "GA failed to import individual" << endl;
    }
}

Individual GeneticAlgorithm::getIndividual(uint _id)
{
    Individual clone;
    if(_id < this->population.individuals.size())
    {
        clone = this->population.individuals[_id];
        return clone;
    } else
    {
        clone = createEmptyIndividual();
        return clone;
    }
}


void GeneticAlgorithm::results()
{
    (*this).printBest();
}

void GeneticAlgorithm::state()
{
    (*this).printState();
}

void GeneticAlgorithm::times()
{
    (*this).updateRunTime();
    (*this).printTimes();
}

// void GeneticAlgorithm::sort()
// {
//     (*this).sortPopulation();
// }

void GeneticAlgorithm::fillProbabilitiesArray(vector<double> &probabilities)
{
    for (uint id = 0; id < probabilities.size(); id++)
    {
        probabilities[id] = exp((-1.0) * this->beta * probabilities[id]);
    }
}

uint GeneticAlgorithm::rouletteWheelSelection(vector<double> &p)
{

    vector<double> c = cumsum(p);
    double r = generateRandom(0.0, 1.0);
    uint index =  findIndex(r, c);    

    return index;
}

Offspring GeneticAlgorithm::applyCrossover(const Individual &parent1, const Individual &parent2)
{

    Offspring children;
    children.c1.copyIndividual(parent1);
    children.c2.copyIndividual(parent2);

    for (uint gene = 0; gene < children.c1.genotype.size(); gene++)
    {
        
        // Standard Recommendation
        // double alpha = generateRandom(-gamma, 1.0 + gamma);

        // Paper Recommendation
        double alpha = generateRandom(0.0, this->gamma);

        children.c1.genotype[gene] = alpha * parent1.genotype[gene] + (1.0 - alpha) * parent2.genotype[gene];
        children.c2.genotype[gene] = alpha * parent2.genotype[gene] + (1.0 - alpha) * parent1.genotype[gene];
    }

    return children;
}

void GeneticAlgorithm::applyBounds(Individual &child, vector<double> &minvalue, vector<double> &maxvalue)
{
    for (uint var = 0; var < minvalue.size(); var++)
    {
        child.genotype[var] = findMax(child.genotype[var], minvalue[var]);
        child.genotype[var] = findMin(child.genotype[var], maxvalue[var]);
    }
}


// basic math and vector functions
double GeneticAlgorithm::generateRandom(double minvalue, double maxvalue)
{

    int CPUfactor = this->mpi_rank + 1;
    if(GA_OPENMP) CPUfactor += omp_get_thread_num();
    std::mt19937_64 myRNG;
    std::random_device device;    
    myRNG.seed(device() * CPUfactor * CPUfactor);
    std::uniform_real_distribution<double> double_dist;

    double range = maxvalue - minvalue;
    double random = (double_dist(myRNG) * range) + minvalue;
    return random;
}

int GeneticAlgorithm::generateRandom(int minvalue, int maxvalue)
{

    int CPUfactor = this->mpi_rank + 1;
    if(GA_OPENMP) CPUfactor += omp_get_thread_num();
    std::mt19937_64 myRNG;
    std::random_device device;    
    myRNG.seed(device() * CPUfactor * CPUfactor);
    std::uniform_int_distribution<std::mt19937::result_type> dist(minvalue, maxvalue);

    return dist(myRNG);
}

double GeneticAlgorithm::findMean(vector<double> &array)
{
    double sum = 0;
    double size = array.size();

    for (uint id = 0; id < size; id++)
    {
        sum += array[id];
    }

    return (sum / size);
}

double GeneticAlgorithm::findStdDev(vector<double> &array)
{
    double mean = (*this).findMean(array);
    double sum = 0.0;
    int size = array.size();

    for(uint idx = 0; idx < size; idx++)
    {
        sum += (array[idx] - mean) * (array[idx] - mean); 
    }

    return sqrt(sum/((double) size));
}

double GeneticAlgorithm::sum_elements(vector<double> &array)
{

    double sum = 0.0;
    for (uint i = 0; i < array.size(); i++)
    {
        sum += array[i];
    }

    return sum;
}

vector<double> GeneticAlgorithm::cumsum(vector<double> &array)
{
    vector<double> cumSumArray(array.size());
    cumSumArray[0] = array[0];
    if (array.size() > 0)
    {
        for (uint i = 1; i < array.size(); i++)
        {
            cumSumArray[i] = array[i] + cumSumArray[i - 1];
        }
    }

    return cumSumArray;
}

// function to find first element of array that is
// greater than a value x
uint GeneticAlgorithm::findIndex(double x, vector<double> &array)
{
    uint index = 0;

    while (x > array[index] && index < array.size())
    {
        index++;
    }

    return index;
}

double GeneticAlgorithm::findMax(double &a, double &b)
{
    if (a > b)
        return a;
    else
        return b;
}

double GeneticAlgorithm::findMin(double &a, double &b)
{
    if (a < b)
        return a;
    else
        return b;
}

void GeneticAlgorithm::get_multi_thread_loop_limits(const int tid, 
                                                    const int num_threads, 
                                                    const int num_loops, 
                                                    int &start, 
                                                    int &finish)
{
    const int loops_per_thread = num_loops / num_threads;
    start = tid * loops_per_thread;
    if (tid == (num_threads - 1)) { finish = num_loops; }
    else { finish = start + loops_per_thread; }
}

void GeneticAlgorithm::save()
{
    if(GA_SAVE_MODE) 
    {
        double time = omp_get_wtime();
        (*this).writeState();
        this->save_time += omp_get_wtime() - time;
    }
}

void GeneticAlgorithm::writeState()
{
    string filename = this->NMR.simulationDirectory + "/NMR_GA_evolution.txt";
    ofstream file;
    file.open(filename, ios::app);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    file << "[" << this->mpi_rank << "]" << " Generation " << this->currentGeneration << ":" << endl;
    for(int id = 0; id < this->population.individuals.size(); id++)
    {
            file << "#" << id << "\tind: { ";
            file.precision(5);
            for (uint var = 0; var < this->population.individuals[id].genotype.size(); var++)
            {
                file << " " << this->population.individuals[id].genotype[var] << ",  ";
            }
            file << "}";
            file.precision(12);
            file << "\t fit: " << this->population.individuals[id].fitness << endl;
    }    
    file << endl;

    file.close();
}

void GeneticAlgorithm::createOutputFile()
{
    string filename = this->NMR.simulationDirectory + "/NMR_GA_evolution.txt";
    ofstream file;
    file.open(filename, ios::out);
    if (file.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    file << "new GA" << endl;

    file.close();
}

simulationMethod GeneticAlgorithm::getImageBasedMethod()
{
    simulationMethod method = ImageBased;
    return method;
}

simulationMethod GeneticAlgorithm::getHistogramBasedMethod()
{
    simulationMethod method = HistogramBased;
    return method;
}