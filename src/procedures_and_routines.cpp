// Main Program
int displacement_test(int argc, char *argv[])
{    
     // Initialize NMR_Simulations object for GA optimization
     int myRank = 0;
     NMR_Simulation NMR = initNMR("PFGSE_NMR");

     // --
     // Load and read rock image and NMR T2 data 
     // (only master process have access to these files) 
     NMR.info(); 
     NMR.readImage();
     NMR.readInputT2();                 

     // --
     // Set and place walkers
     NMR.setWalkers(256);
     NMR.placeWalkersInSamePoint(3,4,5000);
     NMR.setTimeFramework(2000.0);
     cout << "steps = " << NMR.simulationSteps << endl;
     cout << "time/step = " << NMR.timeInterval << endl;
     NMR.mapSimulation();
     NMR.save();

     uint max_displacement = 0;
     uint wid = 0;
     for(uint id = 0; id < NMR.walkers.size(); id++)
     {
          uint displacement = abs((int) NMR.walkers[id].position_z - 
                                  (int) NMR.walkers[id].initialPosition.z);

          if(displacement > max_displacement)
          {
               max_displacement = displacement;
               wid = id;
          }
     }

     cout << "walker[" << wid << "]: " << max_displacement << endl;

}

// Main Program
int PGFSE_proc(int argc, char *argv[])
{    
     // Initialize NMR_Simulations object for GA optimization
     int myRank = 0;
     NMR_Simulation NMR = initNMR("PFGSE_NMR");

     // --
     // Load and read rock image and NMR T2 data 
     // (only master process have access to these files) 
     NMR.info(); 
     NMR.readImage();
     NMR.readInputT2();                 

     // --
     // Set and place walkers
     NMR.setWalkers(256);
     NMR.placeWalkersInSamePoint(3,4,5000);

     // --
     // Run PFGSE simulation parameters
     double exposureTime = 50.0;      // PFG Exposure time in ms (needs to be at least 10x tinyDelta)
     double timeIncrement = 50.0;
     int runs = 0;
     double minGradient = 0.0;     // 0 -> 10 (?) 
     double maxGradient = 40.0;    // 0 -> 10 (?) 
     double tinyDelta = 4.0;       // ideally, lim->0, but in lab is around ~4ms 
     double gamma = 42.576;        // giromagnetic ratio of Hidrogen spin
     int gradientSamples = 128;
     double threshold = 0.01;

     for(int run = 0; run < runs; run++)
     {
          // run experiment simulation
          cout << endl << "running PFGSE simulation:" << endl;
          NMR_PFGSE pfgse(NMR, 
                          maxGradient, 
                          gradientSamples, 
                          minGradient, 
                          exposureTime, 
                          tinyDelta, 
                          gamma);
          pfgse.run();
          pfgse.setThresholdFromLHSValue(threshold);
          pfgse.recover();
          pfgse.save();     

          // increment exposure time
          exposureTime += timeIncrement;
     }     
     
     return 0;
}

void GA_NMR_T2(NMR_Simulation &NMR, int myRank, int mpi_processes)
{
     // Initialize time count
     double time = omp_get_wtime();

     // Initialize Island Model Genetic Algorithm over NMR Simulation object     
     uint sigmoidParameters = GA_GENOTYPE_SIZE;
     uint generations = 5;
     bool verbose = true;

     GA_Island iGA(NMR, RW_NMR_T2, sigmoidParameters, myRank, mpi_processes, verbose);
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

double RW_NMR_T2(vector<double> &sigmoid, NMR_Simulation &NMR)
{
     // Update walkers superficial relaxativity from the candidate solution vector
     NMR.updateWalkersRelaxativity(sigmoid);
     
     // Perform walk simulation to get Global Energy Decay
     // and save GE decay over time in disc     
     NMR.walkSimulation();

     // Perform Laplace Inversion
     NMR.applyLaplaceInversion();
     
     // Find correlation value between simulated and input T2 distributions
     // this value is elevated to 4th potency so that correlation intervals can be amplified
     // this tatic may benefit the search heuristics adopted - it is optional
     double correlation;
     correlation = NMR.leastSquaresT2();
     // correlation = NMR.correlateT2();
     correlation *= correlation;
     
     // reset GE and T2 vectors for next simulations
     NMR.resetGlobalEnergy();
     NMR.resetT2Distribution();

     return correlation;
}

void generateInputT2(NMR_Simulation &NMR)
{
     vector<double> sigmoid(8);
     // {K1, A1, eps1, B1, K2, A2, eps2, B2}
     sigmoid = {35.0, 0.5, 0.25, 2.0, 35.0, 0.5, 0.5, 2.0}; // theoretical 
     // sigmoid = {27.9406, 29.1872, 0.119314, 88.6019, 17.8748, 17.0588, 0.994237, 51.4817};    

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

void mpiPattern(int argc, char *argv[])
{
     // --
     // Initialize MPI variables
     int myRank;
     int mpi_processes;

     // --
     // Initialize MPI interface
     MPI_Init(&argc, &argv);
     MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
     MPI_Comm_size(MPI_COMM_WORLD, &mpi_processes);

     cout << "rank " << myRank << " is alive!" << endl;

     // --
     // Finalize MPI interface
     MPI_Finalize();
}

void histSimulation_debug(int myRank, int mpi_processes)
{
     // --
     // Set relaxativity distributions
     vector<double> sigmoid(8);
     double rho;
     if(myRank == 0)
     {    
          sigmoid = {30.0,2.5,0.3,45.0,15.0,2.5,0.6,50.0};
          rho = 40.0;     
     } 
     if(myRank == 1)
     {
          sigmoid = {2.5,15.0,0.6,100.0,2.5,15.0,0.35,50.0};
          rho = 20.0;      
     }
     if(myRank == 2)
     {    
          sigmoid = {1.0,20.0,0.8,10.0,10.0,1.0,0.2,10.0};
          rho = 10.0;  
     }
     if(myRank == 3)
     {     
          sigmoid = {35.0,1.0,0.7,10.0,1.0,15.0,0.2,35.0};
          rho = 5.0;  
     }

     // --
     // Histogram based simulations
     // --
     // Initialize NMR_Simulations object for GA optimization
     NMR_Simulation NMR_hst = initNMR("NMR_hstbsd_" + std::to_string(myRank));

     // --
     // Load and read rock image and NMR T2 data 
     // (only master process have access to these files) 
     if(myRank == 0)
     {
          NMR_hst.info(); 
          NMR_hst.readImage();
          NMR_hst.readInputT2();                 
     }

     // --
     // Start MPI Communication to send rock image 
     // and NMR T2 data to other processes
     NMR_Network network_hst(NMR_hst, myRank, mpi_processes);
     network_hst.transfer();

     // --
     // Set and place walkers
     NMR_hst.setWalkers();

     // --
     // Run map simulation
     NMR_hst.mapSimulation();

     // --
     // Run walk simulation - histogram based
     NMR_hst.createHistogram();
     NMR_hst.createPenaltiesVector(sigmoid);
     // NMR_hst.createPenaltiesVector(rho);
     NMR_hst.fastSimulation();

     // Perform Laplace Inversion
     NMR_hst.applyLaplaceInversion();
     NMR_hst.save();
     NMR_hst.clear();



     // --
     // Image based simulations
     // --
     // Initialize NMR_Simulations object for GA optimization
     NMR_Simulation NMR_img = initNMR("NMR_imgbsd_" + std::to_string(myRank)); 

     // --
     // Load and read rock image and NMR T2 data 
     // (only master process have access to these files) 
     if(myRank == 0)
     {
          NMR_img.info(); 
          NMR_img.readImage();
          NMR_img.readInputT2();                 
     }

     // --
     // Start MPI Communication to send rock image 
     // and NMR T2 data to other processes
     NMR_Network network_img(NMR_img, myRank, mpi_processes);
     network_img.transfer();

     // --
     // Set and place walkers
     NMR_img.setWalkers();

     // --
     // Run map simulation
     NMR_img.mapSimulation();

     // --
     // Run walk simulation - image based
     NMR_img.updateWalkersRelaxativity(sigmoid);
     // NMR_img.updateWalkersRelaxativity(rho);
     NMR_img.walkSimulation();

     // Perform Laplace Inversion
     NMR_img.applyLaplaceInversion();
     NMR_img.save();
     NMR_img.clear();
}