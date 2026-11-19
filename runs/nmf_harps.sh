#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=256G   # total memory; independent of CPU count
#SBATCH --job-name=spectral_NMF_marcs
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

# Use processes for raw spectra. The Python script sets NMF threads separately.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NMF_THREADS=${NMF_THREADS:-8}
export PYTHONUNBUFFERED=1

uv run ./runs/nmf_harps.py "$@"
