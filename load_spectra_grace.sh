#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # one process; it forks its own workers
#SBATCH --nodes=1   # workers share the main process's cache, so one node only
#SBATCH --cpus-per-task=48   # worker processes for computing spectra
#SBATCH --mem-per-cpu=8G   # per worker; check actual use with `seff <jobid>` and tune
#SBATCH -J "process bosz spectra"   # job name
#SBATCH --output=logs/load_spectra_%j.out   # %j = job id; logs/ must exist before sbatch

# Each worker uses one core; don't let numpy/scipy oversubscribe with threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Worker count is taken from SLURM_CPUS_PER_TASK.
# Safe to resubmit if the job times out: cached spectra are skipped and at most
# the last partial group (< --spectra-per-file spectra) is recomputed.
# Extra arguments passed to sbatch are appended and override the ones below.
uv run load_spectra.py \
    --root /home/mward19/nobackup/autodelete/bosz-spectra/data \
    --cache-dir /home/mward19/nobackup/autodelete/bosz-spectra/processed-harps \
    --R-in 300000 \
    --R-out 3707.32 \
    --D 3000 \
    --wave-min 3000 \
    --wave-max 7000 \
    --spectra-per-file 1000 \
    "$@"
