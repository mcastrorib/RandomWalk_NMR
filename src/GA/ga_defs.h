#ifndef GA_DEFS_H
#define GA_DEFS_H

typedef enum SimulationMethod
{
    ImageBased = 0,
    HistogramBased = 1
} simulationMethod;

// openMP flag
#define GA_OPENMP false

// GA tolerance
#define GA_SOLUTION_TOLERANCE 0.000001

// GA genotype
#define GA_GENOTYPE_SIZE 8
#define GA_MAX_VALUES {50.0, 50.0, 1.0, 100.0, 50.0, 50.0, 1.0, 100.0}
#define GA_MIN_VALUES {0.1, 0.1, 0.0, 1.0, 0.1, 0.1, 0.0, 1.0}

// GA core
#define GA_POPULATION_SIZE 24
#define GA_OFFSPRING_PROPORTION 0.5
#define GA_GAMMA 0.5
#define GA_MUTATION_RATIO 0.1
#define GA_MUTATION_DEVIATION 1.0
#define GA_MUTATIONS_PER_RESET 5
#define GA_RESET_PROPORTION 0.75
#define GA_TOP_SIZE 0.25
#define GA_BETA 1.0
#define GA_DIVERSITY 0.9
#define GA_MEAN_DEVIATION true
#define GA_RESET_POP true
#define GA_SAVE_MODE true

// MPI GA island
#define GA_GEN_PER_MIGRATION 5
#define GA_MIGRATION_RATE 0.25
#define GA_MIGRATION_IMPROVEMENT 0.4
#define MIGRATION_START_TAG 1000
#define MIGRATION_READY_TAG 2000
#define MIGRATION_END_TAG 3000
#define GA_ASYNC_READY_TAG 8000
#define GA_ASYNC_DONE_TAG 8002
#define GA_SOLUTION_FOUND_TAG 9999

#endif