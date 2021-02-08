#!/bin/bash
# This scrip builds the Random-Walk NMR Simulator program usin CMAKE
# Before build, all requirements must be installed in the system
# Check "rwnmr_requirements_installation_guide" in "requirements" directory for a complete guidance

# Get system available processors
procs=$(nproc --all)
echo "Building RWNMR with ${procs} processors..."

# Source code path
SRC_DIR='./src/backend'

## BUILDING RELEASE VERSION
# Build directories path
RELEASE_DIR='./build/release'
# Configure the build
cmake -S ${SRC_DIR} -B ${RELEASE_DIR} -D CMAKE_BUILD_TYPE=Release
# Actually build the binaries
cmake --build build/release -j${procs}
# Create symbolic link in root directory
ln -sf ${RELEASE_DIR}/RWNMR RWNMR

## BUILDING DEBUG VERSION
# Build directories path
DEBUG_DIR='./build/debug'
# Configure the build
cmake -S ${SRC_DIR} -B ${DEBUG_DIR} -D CMAKE_BUILD_TYPE=Debug
# Actually build the binaries
cmake --build build/debug -j${procs}



