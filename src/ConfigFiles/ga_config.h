#ifndef ga_CONFIG_H_
#define ga_CONFIG_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class ga_config
{
public:
    // -- T2 REFERENCE CURVE
    string T2_PATH;

    // -- GENOTYPE
    uint GENOTYPE_SIZE;
    vector<double> GENOTYPE_MAX;
    vector<double> GENOTYPE_MIN;

    // -- GA ESSENTIALS
    double TOLERANCE;
    uint POPULATION_SIZE;
    double OFFSPRING_PROPORTION;
    double GAMMA;
    double MUTATION_RATIO;
    double MUTATION_DEVIATION;
    uint MUTATION_PER_RESET;
    double RESET_PROPORTION;
    double ELITE_SIZE;
    double BETA;
    double DIVERSITY;
    bool MEAN_DEVIATION;
    bool RESET_POPULATION;
    
    // -- MIGRATION & MPI 
    uint GEN_PER_MIGRATION;
    double MIGRATION_RATE;
    double MIGRATION_IMPROVEMENT;
    uint MIGRATION_START_TAG;
    uint MIGRATION_READY_TAG;
    uint MIGRATION_END_TAG;
    uint ASYNC_READY_TAG;
    uint ASYNC_DONE_TAG;    
    uint SOLUTION_FOUND_TAG;
        
    // --- GA SAVE. 
    bool SAVE_GA;  

    // default constructors
    ga_config(const string configFile);

    //copy constructors
    ga_config(const ga_config &otherConfig);

    // default destructor
    virtual ~ga_config()
    {
        // cout << "OMPLoopEnabler object destroyed succesfully" << endl;
    } 

    void readConfigFile(const string configFile);
    
    // -- 
    void readT2Path(string s);
    // -- 
    void readGenotypeSize(string s);
    void readGenotypeMax(string s);
    void readGenotypeMin(string s);
    // --
    void readTolerance(string s);
    void readPopulationSize(string s);
    void readOffspringProportion(string s);
    void readGamma(string s);
    void readMutationRatio(string s);
    void readMutatioDeviation(string s);
    void readMutatioPerReset(string s);
    void readResetProportion(string s);
    void readEliteSize(string s);
    void readBeta(string s);
    void readDiversity(string s);
    void readMeanDeviation(string s);
    void readResetPopulation(string s);
    // --
    void readGenPerMigration(string s);
    void readMigrationRate(string s);
    void readMigrationImprovement(string s);
    void readMigrationStartTag(string s);
    void readMigrationReadyTag(string s);
    void readMigrationEndTag(string s);
    void readAsyncReadyTag(string s);
    void readAsyncDoneTag(string s);
    void readSolutionFoundTag(string s);
    // --  
    void readSaveGA(string s);  
};

#endif