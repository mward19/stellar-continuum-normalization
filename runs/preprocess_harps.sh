#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=1   # one parent process writes the shared cache
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --partition=m12
#SBATCH --mem=512G
#SBATCH --job-name=preprocess_harps
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

# Worker processes handle raw spectra; keep numerical libraries single-threaded.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

uv run ./runs/preprocess_harps.py "$@"
