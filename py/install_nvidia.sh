#!/bin/bash

if ! lspci | grep -i 'nvidia' > /dev/null; then
    echo -e "\e[31mThis script requires an NVIDIA GPU.\e[0m"
    exit 1
fi

# Install Conda with packages
echo -e "\e[33mBeginning installation"
echo -e "Installing Miniforge3 with packages...\e[0m"
if command -v conda >/dev/null 2>&1; then
    echo -e "\e[33mMiniforge3 already installed, skipping...\e[0m"
else
    cd ~
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
    rm Miniforge3-Linux-x86_64.sh
    ~/miniforge3/bin/conda init
    echo 'export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
    echo 'export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.2.0/include:$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.2.0/include/c++/:$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.2.0/include/c++/x86_64-conda-linux-gnu:$CPLUS_INCLUDE_PATH"' >> ~/.bashrc
    source ./.bashrc
fi

conda env create --name python-gpu-env --file environment.yml

# Install Cuda 12.8.1
echo -e "\e[33mInstalling Cuda...\e[0m"
if [ -d "$HOME/.local/cuda-12.8.1/" ]; then
    echo -e "\e[33mCUDA 12.8.1 already installed, skipping...\e[0m"
else
    cd ~
    wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
    chmod +x ./cuda_12.8.1_570.124.06_linux.run
    ./cuda_12.8.1_570.124.06_linux.run --toolkit --silent --override --toolkitpath=$HOME/.local/cuda-12.8.1
    rm ~/cuda_12.8.1_570.124.06_linux.run
    echo 'export LD_LIBRARY_PATH="$HOME/.local/cuda-12.8.1/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
    echo 'export PATH=$HOME/cuda-12.8/bin:$PATH' >> ~/.bashrc
fi

echo -e "\e[33mFinished installation\e[0m"

# Installation check
if command -v conda >/dev/null 2>&1; then
    echo -e "\e[32mMiniforge3 SUCCESSFULLY installed\e[0m"
else
    echo -e "\e[31mMiniforge3 FAILED to install\e[0m"
fi

if [ -d "$HOME/.local/cuda-12.8.1/" ]; then
    echo -e "\e[32mCUDA 12.8.1 SUCCESSFULLY installed\e[0m"
else
    echo -e "\e[31mCUDA 12.8.1 FAILED to install\e[0m"
fi
