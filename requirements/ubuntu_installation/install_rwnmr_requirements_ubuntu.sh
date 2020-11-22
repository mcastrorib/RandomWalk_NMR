#!/bin/bash
# This scrip installs all the dependecies of RWNMR program for Ubuntu 20.04 linux distro

# Update system
sudo apt update

# Install CMAKE
cd ./cmake
./install_cmake.sh
cd ..

# Install OpenMPI
cd ./mpi
./install_mpi.sh
cd ..

# Install EIGEN3
cd ./eigen
./install_eigen.sh
cd ..

# Install OpenCV
cd ./opencv
./install_opencv.sh
cd ..

# Install CUDA 
cd ./cuda
./install_cuda.sh
cd ..

echo "RWNMR dependencies installed."
