// include C++ standard libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>

#include "ga_config.h"

using namespace std;

// default constructors
ga_config::ga_config(const string configFile) : config_filepath(configFile)
{
    vector<double> GENOTYPE_MIN();
    vector<double> GENOTYPE_MAX();
    
    string default_dirpath = CONFIG_ROOT;
    string default_filename = GA_CONFIG_DEFAULT;
    (*this).readConfigFile(default_dirpath + default_filename);
	if(configFile != (default_dirpath + default_filename)) (*this).readConfigFile(configFile);
}

//copy constructors
ga_config::ga_config(const ga_config &otherConfig)
{
    this->config_filepath = otherConfig.config_filepath;
    // -- T2 REFERENCE CURVE
    this->T2_PATH = otherConfig.T2_PATH;    

    // -- GENOTYPE
    this->GENOTYPE_SIZE = otherConfig.GENOTYPE_SIZE;
    this->GENOTYPE_MAX = otherConfig.GENOTYPE_MAX;
    this->GENOTYPE_MIN = otherConfig.GENOTYPE_MIN;

    // -- GA ESSENTIALS
    this->TOLERANCE = otherConfig.TOLERANCE;
    this->POPULATION_SIZE = otherConfig.POPULATION_SIZE;
    this->OFFSPRING_PROPORTION = otherConfig.OFFSPRING_PROPORTION;
    this->GAMMA = otherConfig.GAMMA;
    this->MUTATION_RATIO = otherConfig.MUTATION_RATIO;
    this->MUTATION_DEVIATION = otherConfig.MUTATION_DEVIATION;
    this->MUTATION_PER_RESET = otherConfig.MUTATION_PER_RESET;
    this->RESET_PROPORTION = otherConfig.RESET_PROPORTION;
    this->ELITE_SIZE = otherConfig.ELITE_SIZE;
    this->BETA = otherConfig.BETA;
    this->DIVERSITY = otherConfig.DIVERSITY;
    this->MEAN_DEVIATION = otherConfig.MEAN_DEVIATION;
    this->RESET_POPULATION = otherConfig.RESET_POPULATION;
    
    // -- MIGRATION & MPI 
    this->GEN_PER_MIGRATION = otherConfig.GEN_PER_MIGRATION;
    this->MIGRATION_RATE = otherConfig.MIGRATION_RATE;
    this->MIGRATION_IMPROVEMENT = otherConfig.MIGRATION_IMPROVEMENT;
    this->MIGRATION_START_TAG = otherConfig.MIGRATION_START_TAG;
    this->MIGRATION_READY_TAG = otherConfig.MIGRATION_READY_TAG;
    this->MIGRATION_END_TAG = otherConfig.MIGRATION_END_TAG;
    this->ASYNC_READY_TAG = otherConfig.ASYNC_READY_TAG;
    this->ASYNC_DONE_TAG = otherConfig.ASYNC_DONE_TAG;    
    this->SOLUTION_FOUND_TAG = otherConfig.SOLUTION_FOUND_TAG;
        
    // --- GA SAVE. 
    this->SAVE_GA = otherConfig.SAVE_GA;  
}

// read config file
void ga_config::readConfigFile(const string configFile)
{
    ifstream fileObject;
    fileObject.open(configFile, ios::in);
    if (fileObject.fail())
    {
        cout << "Could not open ga config file from disc." << endl;
        exit(1);
    }

    string line;
    while(fileObject)
    {
    	getline(fileObject, line);

    	string s = line;
    	string delimiter = ": ";
		size_t pos = 0;
		string token, content;
    	while ((pos = s.find(delimiter)) != std::string::npos) 
    	{
			token = s.substr(0, pos);
			content = s.substr(pos + delimiter.length(), s.length());
			s.erase(0, pos + delimiter.length());

			if(token == "T2_PATH") (*this).readT2Path(content);
            else if(token == "GENOTYPE_SIZE") (*this).readGenotypeSize(content);
            else if(token == "GENOTYPE_MAX") (*this).readGenotypeMax(content);
            else if(token == "GENOTYPE_MIN") (*this).readGenotypeMin(content);
            else if(token == "TOLERANCE") (*this).readTolerance(content);
            else if(token == "POPULATION_SIZE") (*this).readPopulationSize(content);
            else if(token == "OFFSPRING_PROPORTION") (*this).readOffspringProportion(content);
            else if(token == "GAMMA") (*this).readGamma(content);
            else if(token == "MUTATION_RATIO") (*this).readMutationRatio(content);
            else if(token == "MUTATION_DEVIATION") (*this).readMutatioDeviation(content);
            else if(token == "MUTATIONS_PER_RESET") (*this).readMutatioPerReset(content);
            else if(token == "RESET_PROPORTION") (*this).readResetProportion(content);
            else if(token == "ELITE_SIZE") (*this).readEliteSize(content);        
            else if(token == "BETA") (*this).readBeta(content);
            else if(token == "DIVERSITY") (*this).readDiversity(content);
            else if(token == "MEAN_DEVIATION") (*this).readMeanDeviation(content);
            else if(token == "RESET_POPULATION") (*this).readResetPopulation(content);
            else if(token == "GEN_PER_MIGRATION") (*this).readGenPerMigration(content);
            else if(token == "MIGRATION_RATE") (*this).readMigrationRate(content);
            else if(token == "MIGRATION_IMPROVEMENT") (*this).readMigrationImprovement(content);
            else if(token == "MIGRATION_START_TAG") (*this).readMigrationStartTag(content);
            else if(token == "MIGRATION_READY_TAG") (*this).readMigrationReadyTag(content);
            else if(token == "MIGRATION_END_TAG") (*this).readMigrationEndTag(content);
			else if(token == "ASYNC_READY_TAG") (*this).readAsyncReadyTag(content);
            else if(token == "ASYNC_DONE_TAG") (*this).readAsyncDoneTag(content);
            else if(token == "SOLUTION_FOUND_TAG") (*this).readSolutionFoundTag(content);
            else if(token == "SAVE_GA") (*this).readSaveGA(content);

		}
    } 

    fileObject.close();
}

void ga_config::readT2Path(string s)
{
    this->T2_PATH = s;
}

void ga_config::readGenotypeSize(string s)
{
    this->GENOTYPE_SIZE = std::stoi(s);
}

void ga_config::readGenotypeMax(string s)
{
    // parse vector
    if(s.compare(0, 1, "{") == 0 and s.compare(s.length() - 1, 1, "}") == 0)
    {
        string strvec = s.substr(1, s.length() - 2);
        string delimiter = ",";
        size_t pos = 0;
        string token, content;
        while ((pos = strvec.find(delimiter)) != std::string::npos) 
        {
            token = strvec.substr(0, pos);
            content = strvec.substr(pos + delimiter.length(), strvec.length());
            strvec.erase(0, pos + delimiter.length());

            // add value to RHO attribute
            this->GENOTYPE_MAX.push_back(std::stod(token));
        }
        // add value to RHO attribute
        this->GENOTYPE_MAX.push_back(std::stod(strvec));
    }
}

void ga_config::readGenotypeMin(string s)
{
    // parse vector
    if(s.compare(0, 1, "{") == 0 and s.compare(s.length() - 1, 1, "}") == 0)
    {
        string strvec = s.substr(1, s.length() - 2);
        string delimiter = ",";
        size_t pos = 0;
        string token, content;
        while ((pos = strvec.find(delimiter)) != std::string::npos) 
        {
            token = strvec.substr(0, pos);
            content = strvec.substr(pos + delimiter.length(), strvec.length());
            strvec.erase(0, pos + delimiter.length());

            // add value to RHO attribute
            this->GENOTYPE_MIN.push_back(std::stod(token));
        }
        // add value to RHO attribute
        this->GENOTYPE_MIN.push_back(std::stod(strvec));
    }
}

void ga_config::readTolerance(string s)
{
    this->TOLERANCE = std::stod(s);
}

void ga_config::readPopulationSize(string s)
{
    this->POPULATION_SIZE = std::stoi(s);
}

void ga_config::readOffspringProportion(string s)
{
    this->OFFSPRING_PROPORTION = std::stod(s);
}

void ga_config::readGamma(string s)
{
    this->GAMMA = std::stod(s);
}

void ga_config::readMutationRatio(string s)
{
    this->MUTATION_RATIO = std::stod(s);
}

void ga_config::readMutatioDeviation(string s)
{
    this->MUTATION_DEVIATION = std::stod(s);
}

void ga_config::readMutatioPerReset(string s)
{
    this->MUTATION_PER_RESET = std::stoi(s);
}

void ga_config::readResetProportion(string s)
{
    this->RESET_PROPORTION = std::stod(s);
}

void ga_config::readEliteSize(string s)
{
    this->ELITE_SIZE = std::stod(s);
}

void ga_config::readBeta(string s)
{
    this->BETA = std::stod(s);
}

void ga_config::readDiversity(string s)
{
    this->DIVERSITY = std::stod(s);
}

void ga_config::readMeanDeviation(string s)
{
    if(s == "true") this->MEAN_DEVIATION = true;
    else this->MEAN_DEVIATION = false;
}

void ga_config::readResetPopulation(string s)
{
    if(s == "true") this->RESET_POPULATION = true;
    else this->RESET_POPULATION = false;
}

void ga_config::readGenPerMigration(string s)
{
    this->GEN_PER_MIGRATION = stoi(s);
}

void ga_config::readMigrationRate(string s)
{
    this->MIGRATION_RATE = std::stod(s);
}

void ga_config::readMigrationImprovement(string s)
{
    this->MIGRATION_IMPROVEMENT = std::stod(s);
}

void ga_config::readMigrationStartTag(string s)
{
    this->MIGRATION_START_TAG = std::stoi(s);
}

void ga_config::readMigrationReadyTag(string s)
{
    this->MIGRATION_READY_TAG = std::stoi(s);
}

void ga_config::readMigrationEndTag(string s)
{
    this->MIGRATION_END_TAG = std::stoi(s);
}

void ga_config::readAsyncReadyTag(string s)
{
    this->ASYNC_READY_TAG = std::stoi(s);
}

void ga_config::readAsyncDoneTag(string s)
{
    this->ASYNC_DONE_TAG = std::stoi(s);
}

void ga_config::readSolutionFoundTag(string s)
{
    this->SOLUTION_FOUND_TAG = std::stoi(s);
}

void ga_config::readSaveGA(string s)
{
    if(s == "true") this->SAVE_GA = true;
    else this->SAVE_GA = false;
}  