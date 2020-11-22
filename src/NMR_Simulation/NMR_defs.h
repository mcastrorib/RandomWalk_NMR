#ifndef NMR_DEFS_H
#define NMR_DEFS_H

#define DATA_PATH "/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr/db/"



#ifndef DEFAULT_SIM_PARAM
#define DEFAULT_SIM_PARAM
#define DEFAULT_RELAXATIVITY 0.0		// in um/s
#define DIFFUSION_COEFFICIENT 2.5		// in um²/ms
#define STEPS_PER_ECHO 5
#define IMAGE_RESOLUTION 1.0
#define VOXEL_DIVISIONS 1			// number of material voxels inside an image voxel per dimension
#define INITIAL_SEED 27108652
#endif

// SAVE MODE
#define NMR_SAVE_IMAGE_INFO true
#define NMR_SAVE_COLLISIONS true
#define NMR_SAVE_DECAY true
#define NMR_SAVE_HISTOGRAM true
#define NMR_SAVE_HISTOGRAM_LIST false
#define NMR_SAVE_T2 false
#define NMR_SAVE_BINIMAGE false

// HISTOGRAM SIZE
#define NMR_HISTOGRAMS 1	
#define NMR_HISTOGRAM_SIZE 1024

// OPENMP FLAG
#define NMR_OPENMP true

// CUDA GLOBAL VARIABLES
// kernel configuration
// (sugested: blocks 1024, threads 512, echoes 16, reduce TRUE, max_rwsteps 65536)
#define BLOCKS 4096
#define THREADSPERBLOCK 1024
#define ECHOESPERKERNEL 16
#define REDUCE_IN_GPU true
#define MAX_RWSTEPS 65536

// NMR COMMUNICATION
#define BITBLOCKS_BATCHES_SIZE 100000
#define BITBLOCK_PROP_SIZE 7
#define NMR_T2_SIZE 128
#define NMR_START_TAG 1000
#define NMR_BITBLOCK_TAG 2000
#define NMR_BATCH_TAG 3000
#define NMR_T2_TAG 4000
#define NMR_END_TAG 5000

// PFG NMR SIMULATION
#define TWO_PI (6.2831853071795864769252867665590057683943)
#define GIROMAGNETIC_RATIO 42.576
#define PFGSE_TINY_DELTA 1.0
#define PFGSE_BIG_DELTA 40.0
#define DEFAULT_GRADIENT 0.0
#define PFGSE_USE_TWOPI false

#endif