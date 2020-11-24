// include C++ standard libraries
#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <cmath>

// just to get limits
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

// include MPI API for multiprocesses
#include <mpi.h>

// include OpenMP for multicore implementation
#include <omp.h>

// include CMake Configuration file
#include "NMR_RWConfig.h"
#include "rwnmr_config.h"
#include "uct_config.h"
#include "pfgse_config.h"
#include "cpmg_config.h"
#include "ga_config.h"


// include project files
#include "NMR_Simulation.h"
#include "walker.h"
#include "myRNG.h"
#include "NMR_Network.h"
#include "NMR_pfgse.h"
#include "fileHandler.h"
#include "consoleInput.h"
#include "mpi_ga_island.h"
#include "ga_core.h"
#include "Vector3D.h"
#include "OMPLoopEnabler.h"

using namespace std;
using namespace cv;

// Interface fuctions
int pfgse_old();
ConsoleInput input(string _name);
NMR_Simulation setNMR(ConsoleInput _input);
void GA_NMR_T2(NMR_Simulation &NMR, int myRank, int mpi_processes);
void save_GA_solution_data(vector<double> &sigmoid, NMR_Simulation &NMR);

// Main Program
int main(int argc, char *argv[])
{    
     // -- Read NMR essentials config files
     // -- set path to config files dir
     string config_root = "/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/config/";     
     // -- rwnmr config
     string rw_config_path = config_root + "rwnmr.config";
     rwnmr_config rwNMR_Config(rw_config_path);      
     // -- uct image config
     string uct_config_path = config_root + "uct.config";
     uct_config uCT_Config(uct_config_path); 
     cout << "config files read" << endl << endl;
     // -----

     // -- Build NMR_Simulation essentials
     cout << "-- creating RWNMR object" << endl;
     NMR_Simulation NMR(rwNMR_Config, uCT_Config);
     cout << "- RWNMR object created succesfully." << endl;
     cout << "-- reading uCT-image" << endl;
     NMR.readImage();
     cout << "- uCT-image read succesfully." << endl;     
     cout << "-- setting walkers" << endl;
     NMR.setWalkers();
     cout << "- walkers set." << endl;
     cout << "-- saving uCT-image info" << endl;
     NMR.save();
     cout << "- saved succesfully." << endl << endl;
     NMR.info();
     // -----

     // -- Read PFGSE routine config files
     cout << "-- performing pfgse sequence" << endl;
     pfgse_config pfgse_Config(config_root + "pfgse.config");
     NMR_PFGSE pfgse(NMR, pfgse_Config);
     pfgse.run_sequence();
     cout << "- pfgse sequence performed succesfully" << endl << endl;
     // -----
     return 0;
}

int pfgse_old()
{    
     // Parameters
     int myRank = 0;
     string walker_count = "10k";
     uint numberOfWalkers = 10000;
     uint shift = 0;
     uint deviation = 1;      // for walker's restricted placement
     bool restrictWalkersPlacement = true;
     
     // PFGSE NMR
     // time sampling
     int timeSamples = 10;
     double logspace_min = -1.0;
     double logspace_max = 1.0;
     double length_A = 10.0;    // only this value needs to change 

     // experiment parameters
     double maxGradient = 50.0;    // 0 -> 10 (?)
     Vector3D vecGradient(maxGradient, 0.0, 0.0); 
     double pulseWidth = 0.1;       // ideally, lim->0, but in lab is around ~4ms 
     double gamma = 42.576;        // giromagnetic ratio of Hidrogen spin
     int gradientSamples = 200;
     double threshold = 0.8;

     // Initialize NMR_Simulations object for GA optimization
     NMR_Simulation NMR = setNMR(input("PFGSE_NMR_FreeMedia_rho=0.0_res=1.0_shift=0_w=" + walker_count));

     // --
     // Load and read rock image and NMR T2 data 
     NMR.info(); 
     NMR.readImage();
     NMR.saveInfo();
     NMR.readInputT2();                 

     // --
     // get 3D points to restrict walker placement zone 
     Point3D point1(NMR.getImageWidth()/2 - deviation, 
                    NMR.getImageWidth()/2 - deviation,
                    NMR.getImageDepth()/2 - deviation);

     Point3D point2(NMR.getImageWidth()/2 + deviation, 
                    NMR.getImageWidth()/2 + deviation,
                    NMR.getImageDepth()/2 + deviation);
     // bool random = true;

     // --
     // Set and place walkers
     if(restrictWalkersPlacement) NMR.setWalkers(point1, point2, numberOfWalkers);          
     else NMR.setWalkers(numberOfWalkers);

     // --
     // Apply voxel subdivison 
     if(shift != 0)
     {
          NMR.applyVoxelDivision(shift);
          cout << endl;
          cout << "image-voxel resolution: " << NMR.getImageVoxelResolution() << endl;
          cout << "steps per echo: " << NMR.getStepsPerEcho() << endl;
     }

     // --
     // PFGSE time sampling
     cout << endl << "time samples: " << endl;
     double D_bulk = NMR.getDiffusionCoefficient();
     double time_T = (length_A*length_A)/D_bulk;
     vector<double> time_sampling;
     time_sampling = logspace(logspace_min, logspace_max, timeSamples);     
     vector<double> exposureTime;
     for(int sample = 0; sample < timeSamples; sample++)
     {
          exposureTime.push_back(time_sampling[sample] * time_T);
          cout << "t[" << sample << "] = " << exposureTime[sample] << " ms" << endl;
     }

     // --
     // PFGSE procedure parameters 
     double pi = 0.5 * (TWO_PI);
     double SVrelation;       
     int runs = timeSamples;     
     for(int run = 0; run < runs; run++)
     {
          // run experiment simulation
          cout << endl << "running PFGSE simulation:" << endl;
          NMR_PFGSE pfgse(NMR, 
                          vecGradient, 
                          gradientSamples, 
                          exposureTime[run], 
                          pulseWidth, 
                          gamma);

          pfgse.run();
     
          pfgse.setThresholdFromLHSValue(threshold);
          pfgse.recoverD("stejskal");
          pfgse.recoverD("msd");

          // recover S/V for short observation times (see ref. Sorland)
          SVrelation = (1.0 - (pfgse.getD_msd() / NMR.getDiffusionCoefficient()));
          SVrelation *= 2.25 * sqrt(pi / (NMR.getDiffusionCoefficient() * exposureTime[run]));
          cout << "S/V ~= " << SVrelation << endl;
          
          // save results in disc
          pfgse.save();  
     }
     
     // NMR.mapSimulation();
     // NMR.walkSimulation();
     // NMR.save();   
     return 0;
}

ConsoleInput input(string _name)
{
     // User Input
     ConsoleInput input;
     input.simulationName = _name; 
     input.steps = 40000;
     input.occupancy = 1.0;
     input.numberOfImages = 200;
     input.seed = myRNG::RNG_uint64();
     input.imagePath.path = "/home/matheus/Documentos/doutorado_ic/tese/NMR/Images/Synthetic/free_media/free_space/imgs/";
     input.imagePath.filename = "free_media_";
     input.imagePath.fileID = 0;
     // input.imagePath.digits = 3;
     input.updateCompletePath();
     input.T2path = "/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr/db/input_data/tiny_3D/NMR_T2.txt";
     input.use_GPU = true;

     return input;
}

NMR_Simulation setNMR(ConsoleInput input)
{
     
     // Declare NMR_Simulation Object
     NMR_Simulation NMR(input.simulationName);
     // NMR.setSimulation(input.occupancy, input.seed, input.use_GPU);
     NMR.setGPU(input.use_GPU);
     NMR.setImageOccupancy(input.occupancy);
     NMR.setInitialSeed(input.seed);
     NMR.setTimeFramework(input.steps);
     NMR.setImage(input.imagePath, input.numberOfImages);     
     NMR.setInputT2(input.T2path);       

     return NMR;
}

void GA_NMR_T2(NMR_Simulation &NMR, int myRank, int mpi_processes)
{
     // Initialize time count
     double time = omp_get_wtime();

     // Initialize Island Model Genetic Algorithm over NMR Simulation object     
     uint sigmoidParameters = GA_GENOTYPE_SIZE;
     uint generations = 5;
     bool verbose = true;

     GA_Island iGA(NMR, sigmoidParameters, myRank, mpi_processes, verbose);
     iGA.runAsync(generations);  

     iGA.setMethod(iGA.getImageBasedMethod());
     iGA.runAsync(generations);

     iGA.save();

     // get best individual data
     vector<double> sigmoid;
     sigmoid = iGA.GA.bestIndividual.genotype;
     save_GA_solution_data(sigmoid, NMR);

     // log procedure runtime 
     time = omp_get_wtime() - time;
     sleep(mpi_processes - myRank);
     cout << "[" << myRank << "]" << " ~ procedure finished in " << time << " s."  << endl;
}


void save_GA_solution_data(vector<double> &sigmoid, NMR_Simulation &NMR)
{
     // Update walkers superficial relaxativity from the candidate solution vector
     NMR.updateWalkersRelaxativity(sigmoid);  

     // Perform walk simulation to get Global Energy Decay
     // and save GE decay over time in disc     
     NMR.walkSimulation();

     // Perform Laplace Inversion
     NMR.applyLaplaceInversion();

     // Save simulation data
     NMR.save();
}


