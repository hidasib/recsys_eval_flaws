#!/bin/sh
eval "$(conda shell.bash hook)"
conda config --append channels nvidia
conda env create -f conda_gru4rec_theano_gpu.yml
conda activate gru4rec_theano_gpu
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
touch $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
touch $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
activate_script='#!/bin/sh
export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export ORIGINAL_CUDA_HOME=$CUDA_HOME
export ORIGINAL_CUDA_ROOT=$CUDA_ROOT
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
export CUDA_HOME="${CONDA_PREFIX}/lib"
export CUDA_ROOT="${CONDA_PREFIX}/lib"'
deactivate_script='#!/bin/sh
export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH
export CUDA_HOME=$ORIGINAL_CUDA_HOME
export CUDA_ROOT=$ORIGINAL_CUDA_ROOT
unset ORIGINAL_LD_LIBRARY_PATH
unset ORIGINAL_CUDA_HOME
unset ORIGINAL_CUDA_ROOT'
printf %b "$activate_script" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
printf %b "$deactivate_script" > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
conda deactivate