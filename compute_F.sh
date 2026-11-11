#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32G   # memory per CPU core
#SBATCH --job-name=spectral_NMF_marcs
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export PYTHONUNBUFFERED=1

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
uv run construct_big_nmf.py
