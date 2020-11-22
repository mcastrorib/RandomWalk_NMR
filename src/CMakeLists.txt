cmake_minimum_required(VERSION 3.17 FATAL_ERROR)


# set the project name
project(NMR_RW_simulation VERSION 1.1 LANGUAGES C CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# specify mpi compilers
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

option(USE_NMR_SIMULATION "Use provided RMN_Simulation2D class implementation" ON)
option(USE_WALKER "Use provided walker class implementation" ON)
option(USE_BITBLOCK "Use provided blockMap class implementation" ON)
option(USE_RNG "Use provided RNG implementations for random number generation" ON)
option(USE_LAPLACE "Use provided Tikhonov solver for Laplace inverse transform" ON)
option(USE_GA "Use provided Genetic Algorithm implementation for optimization" ON)
option(USE_FILEHANDLER "Use provided implementation for handling external files" ON)
option(USE_CONSOLEINPUT "Use provided implementation for reading user's console input" ON)
option(USE_MATH "Use provided implementation for math and linear algebra computations" ON)
option(USE_UTILS "Use provided implementation for system utils" ON)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file(NMR_RWConfig.h.in NMR_RWConfig.h)

# Require MPI package
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
include_directories(MPI_INCLUDE_PATH)

# include OpenCV package
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# include CUDA package
find_package(CUDAToolkit REQUIRED)


# include OpenMP package
find_package( OpenMP REQUIRED )

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#include Eigen package
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(/usr/include/eigen3)


# list libraries
if(USE_NMR_SIMULATION)
  add_subdirectory(NMR_Simulation)
  list(APPEND EXTRA_LIBS NMR_Simulation)
endif()

if(USE_WALKER)
  add_subdirectory(Walker)
  list(APPEND EXTRA_LIBS Walker)
endif()

if(USE_BITBLOCK)
  add_subdirectory(BitBlock)
  list(APPEND EXTRA_LIBS BitBlock)
endif()

if(USE_RNG)
  add_subdirectory(RNG)
  list(APPEND EXTRA_LIBS RNG)
endif()

if(USE_LAPLACE)
  add_subdirectory(Laplace)
  list(APPEND EXTRA_LIBS Laplace)
endif()

if(USE_GA)
  add_subdirectory(GA)
  list(APPEND EXTRA_LIBS GA)
endif()

if(USE_FILEHANDLER)
  add_subdirectory(FileHandler)
  list(APPEND EXTRA_LIBS FileHandler)
endif()

if(USE_CONSOLEINPUT)
  add_subdirectory(ConsoleInput)
  list(APPEND EXTRA_LIBS ConsoleInput)
endif()

if(USE_MATH)
  add_subdirectory(Math)
  list(APPEND EXTRA_LIBS Math)
endif()

if(USE_UTILS)
  add_subdirectory(Utils)
  list(APPEND EXTRA_LIBS Utils)
endif()

# add the executable
add_executable(RWNMR main.cpp)


target_link_libraries(RWNMR PUBLIC ${EXTRA_LIBS} ${MPI_LIBRARIES} ${OpenCV_LIBS} OpenMP::OpenMP_CXX Eigen3::Eigen)
target_link_libraries(RWNMR PRIVATE CUDA::cudart)


# add the binary tree to the search path for include files
# so that we will find RandomWalkConfig.h
target_include_directories(RWNMR PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )

install(TARGETS RWNMR DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/NMR_RWConfig.h"
  DESTINATION include
  )
