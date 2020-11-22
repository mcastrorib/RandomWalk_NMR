#!/bin/bash
# This install latest version of CMAKE using snap 

# CMAKE 3.17 or greater (project build)
# if an older version of cmake is already installed via apt, first uninstall it 
sudo apt --purge autoremove cmake
sudo snap install cmake --classic

