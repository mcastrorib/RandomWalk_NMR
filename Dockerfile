# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.0.3-devel-ubuntu20.04
WORKDIR /app

# Remove geographic region apt interaction
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt install -y tcl
RUN apt update && apt install -y libssl-dev wget 

# Install cmake (latest)
RUN apt update && apt install -y software-properties-common lsb-release 
RUN apt clean all
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt update && apt install -y kitware-archive-keyring 
RUN rm /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt update && apt install -y cmake

# install dependencies
RUN apt update && apt install -y libopenmpi-dev openmpi-bin 
RUN apt update && apt install -y libeigen3-dev 
RUN apt update && apt install -y libopencv-dev python3-opencv

# Add CUDA environment variables
ENV CUDADIR=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Remove cache from apt
RUN rm -rf /var/cache/apt/*

# Copy source code to container workspace
COPY . .