#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32G   # memory per CPU core
#SBATCH --job-name=spectral_NMF_marcs
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

# Set OpenMP threads from the node-level CPU count.
# The Python program runs once; numerical libraries handle threading.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export PYTHONUNBUFFERED=1

# Train or load the production stellar and telluric bases.
uv run construct_big_nmf.py
