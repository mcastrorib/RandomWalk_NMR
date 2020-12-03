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
    string config_filepath;
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
    ga_config(){};
    ga_config(const string configFile);

    //copy constructors
    ga_config(const ga_config &otherConfig);

    // default destructor
    virtual ~ga_config()
    {
        // cout << "OMPLoopEnabler object destroyed succesfully" << endl;
    } 

    void readConfigFile(const string configFile);
    
    // -- Read methods
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

    // -- Get methods
    string getConfigFilepath() {return this->config_filepath; }
    string getT2Path() { return this->T2_PATH; }
    // -- 
    uint getGenotypeSize() { return this->GENOTYPE_SIZE ; }
    vector<double> getGenotypeMax() { return this->GENOTYPE_MAX ; }
    vector<double> getGenotypeMin() { return this->GENOTYPE_MIN ; }
    // --
    double getTolerance() { return this->TOLERANCE ; }
    uint getPopulationSize() { return this->POPULATION_SIZE ; }
    double getOffspringProportion() { return this->OFFSPRING_PROPORTION ; }
    double getGamma() { return this->GAMMA ; }
    double getMutationRatio() { return this->MUTATION_RATIO ; }
    double getMutatioDeviation() { return this->MUTATION_DEVIATION ; }
    uint getMutatioPerReset() { return this->MUTATION_PER_RESET ; }
    double getResetProportion() { return this->RESET_PROPORTION ; }
    double getEliteSize() { return this->ELITE_SIZE ; }
    double getBeta() { return this->BETA ; }
    double getDiversity() { return this->DIVERSITY ; }
    double getMeanDeviation() { return this->MEAN_DEVIATION ; }
    bool getResetPopulation() { return this->RESET_POPULATION ; }
    // --
    uint getGenPerMigration() { return this->GEN_PER_MIGRATION ; }
    double getMigrationRate() { return this->MIGRATION_RATE ; }
    double getMigrationImprovement() { return this->MIGRATION_IMPROVEMENT ; }
    uint getMigrationStartTag() { return this->MIGRATION_START_TAG ; }
    uint getMigrationReadyTag() { return this->MIGRATION_READY_TAG ; }
    uint getMigrationEndTag() { return this->MIGRATION_END_TAG ; }
    uint getAsyncReadyTag() { return this->ASYNC_READY_TAG ; }
    uint getAsyncDoneTag() { return this->ASYNC_DONE_TAG ; }
    uint getSolutionFoundTag() { return this->SOLUTION_FOUND_TAG ; }
    // --  
    bool getSaveGA() { return this->SAVE_GA ; }  
};

#endif