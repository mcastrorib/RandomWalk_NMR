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
#include "NMR_pfgse.h"
#include "NMR_cpmg.h"

using namespace std;
using namespace cv;

// Main Program
int main(int argc, char *argv[])
{    
     // -- Read NMR essentials config files
     // -- set path to config files dir
     string config_root = "/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/config/";     
     // -- rwnmr & uct image config
     rwnmr_config rwNMR_Config(config_root + "rwnmr.config");      
     uct_config uCT_Config(config_root + "uct.config"); 
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
     pfgse.run();
     cout << "- pfgse sequence performed succesfully" << endl << endl;
     // -----

     // -- Read CPMG routine config files
     cout << "-- performing cpmg sequence" << endl;
     cpmg_config cpmg_Config(config_root + "cpmg.config");
     NMR_cpmg cpmg(NMR, cpmg_Config);
     cpmg.run();
     cout << "- cpmg sequence performed succesfully" << endl << endl;
     // -----

     return 0;
}

