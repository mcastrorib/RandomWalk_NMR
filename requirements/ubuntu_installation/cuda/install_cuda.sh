#!/bin/bash
# This script installs CUDA 11.1 in ubuntu 20.04 

## CUDA (nvidia toolkit for gpu acceleration)
# for ubuntu 20.04:
ubuntu_version=2004
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu_version}/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu${ubuntu_version}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu${ubuntu_version}-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu${ubuntu_version}-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu${ubuntu_version}-11-1-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# for other distros, check the step-by-step at nvidia developers page:
# https://developer.nvidia.com/CUDA-toolkit


# - now to compile/run CUDA files, CUDA paths must be added to 
# - the enviroment variables everytime a new terminal window is open: 

# $ export PATH=$PATH:/usr/local/cuda/bin
# $ export CUDADIR=/usr/local/cuda
# $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# - this procedure can be automatic, by coping the file named 'cuda.sh' to /etc/profile.d/ 
sudo cp cuda.sh /etc/profile.d/cuda_paths.sh


