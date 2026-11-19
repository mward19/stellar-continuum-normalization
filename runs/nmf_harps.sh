#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=m12
#SBATCH --mem=512G
#SBATCH --job-name=nmf_harps
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

# Coordinate descent benefits from a moderate numerical thread count.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NMF_THREADS=${NMF_THREADS:-16}
export PYTHONUNBUFFERED=1

uv run ./runs/nmf_harps.py "$@"
