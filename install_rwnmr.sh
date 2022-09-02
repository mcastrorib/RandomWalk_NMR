#!/bin/bash
# This scrip builds the Random-Walk NMR Simulator program usin CMAKE
# Before build, all requirements must be installed in the system
# Check "rwnmr_requirements_installation_guide" in "requirements" directory for a complete guidance

# Get system available processors
procs=$(nproc --all)

# Source code path
SRC_DIR='./src/backend'

# Assign compilation mode from command line args
TARGET_MODE="Release"
TARGET_DIR="./build/release"

for i in "$@"; do
	case $1 in
		-r|--release)
			TARGET_MODE="Release"
			TARGET_DIR="./build/release"
			shift
			;;
		-d|--debug)
			TARGET_MODE="Debug"
			TARGET_DIR="./build/debug"
			shift
			;;
		*)
			TARGET_MODE="Release"
			TARGET_DIR="./build/release"
			;;
	esac
done

## BUILDING RWNMR
echo "Building RWNMR (${TARGET_MODE}) with ${procs} processors..."
# Configure the build
cmake -S ${SRC_DIR} -B ${TARGET_DIR} -D CMAKE_BUILD_TYPE=${TARGET_MODE}
# Actually build the binaries
cmake --build ${TARGET_DIR} -j${procs}

if [ ${TARGET_MODE} == "Release" ] 
then
	# Create symbolic link in root directory
	echo "Creating symbolic link to executable RWNMR"
	ln -sf ${TARGET_DIR}/RWNMR RWNMR
fi


# ## BUILDING RELEASE VERSION
# # Build directories path
# RELEASE_DIR='./build/release'
# # Configure the build
# cmake -S ${SRC_DIR} -B ${RELEASE_DIR} -D CMAKE_BUILD_TYPE=Release
# # Actually build the binaries
# cmake --build build/release -j${procs}
# # Create symbolic link in root directory
# ln -sf ${RELEASE_DIR}/RWNMR RWNMR

# ## BUILDING DEBUG VERSION
# # Build directories path
# DEBUG_DIR='./build/debug'
# # Configure the build
# cmake -S ${SRC_DIR} -B ${DEBUG_DIR} -D CMAKE_BUILD_TYPE=Debug
# # Actually build the binaries
# cmake --build build/debug -j${procs}



