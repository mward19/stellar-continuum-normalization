"""Prepare and cache the HARPS stellar training matrix."""
import argparse
import os

from src.preparation import prepare_bosz
from harps import load_harps_training_data


parser = argparse.ArgumentParser(description="Preprocess the HARPS stellar training matrix")
parser.add_argument(
    "--matrix-format",
    choices=("dense", "sparse"),
    default="dense",
    help="Training matrix representation (default: dense float32)"
)
parser.add_argument(
    "--row-stride",
    type=int,
    default=1,
    help="Keep every n-th MARCS grid spectrum (default: 1, use all selected spectra)",
)
parser.add_argument(
    "--n-workers",
    type=int,
    default=-1,
    help="Raw-spectrum workers; -1 uses the Slurm CPU allocation (default: -1)"
)
args = parser.parse_args()

if args.n_workers == 0 or args.n_workers < -1:
    parser.error("--n-workers must be -1 or a positive integer")
if args.row_stride < 1:
    parser.error("--row-stride must be positive")
spectra, selection, trainer, cache = load_harps_training_data(
    args.matrix_format, row_stride=args.row_stride
)
print(f"Selected {len(selection):,} MARCS spectra (row stride {args.row_stride})", flush=True)

n_workers = None if args.n_workers == -1 else args.n_workers
prepared = prepare_bosz(
    spectra,
    selection,
    threshold=trainer.config['sparse_tol'],
    floor=trainer.config['sparse_floor'],
    n_workers=n_workers,
    matrix_cache=cache,
    matrix_format=args.matrix_format,
)
print(f"Prepared matrix cached in {cache.directory}", flush=True)
print(f"Shape: {prepared.optical_depth.shape}; dtype: {prepared.optical_depth.dtype}", flush=True)
