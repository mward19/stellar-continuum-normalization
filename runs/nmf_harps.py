"""Fit the HARPS stellar NMF basis from the prepared matrix cache."""
import argparse
import os

from threadpoolctl import threadpool_limits

from src.nmf import PreparedSpectra
from src.nmf_cache import CacheMiss
from harps import BASIS_CACHE, load_harps_training_data, prepared_matrix_params


parser = argparse.ArgumentParser(description="Fit the HARPS stellar NMF basis")
parser.add_argument(
    "--matrix-format",
    choices=("dense", "sparse"),
    default="dense",
    help="Prepared matrix representation (default: dense float32)"
)
parser.add_argument(
    "--row-stride",
    type=int,
    default=1,
    help="Keep every n-th MARCS grid spectrum; must match preprocessing (default: 1)",
)
args = parser.parse_args()
if args.row_stride < 1:
    parser.error("--row-stride must be positive")

spectra, selection, stellar, matrix_cache = load_harps_training_data(
    args.matrix_format, row_stride=args.row_stride
)
params = prepared_matrix_params(spectra, selection, stellar)
try:
    matrix = matrix_cache.load_matrix(params, storage=args.matrix_format)
except CacheMiss as exc:
    raise SystemExit(
        f"Prepared {args.matrix_format} HARPS matrix is missing from {matrix_cache.directory}. "
        "Run sbatch runs/preprocess_harps.sh first with the same --matrix-format."
    ) from exc

prepared = PreparedSpectra(
    optical_depth=matrix,
    wavelengths=spectra.wave_out,
    preprocessing=params['preprocessing'],
    source_manifest=params['source_manifest'],
)
nmf_threads = int(os.environ.get("NMF_THREADS", "16"))
print(
    f"Fitting {len(selection):,} spectra (row stride {args.row_stride}) "
    f"from {matrix_cache.directory} "
    f"with {args.matrix_format} storage and {nmf_threads} numerical threads",
    flush=True
)
with threadpool_limits(limits=nmf_threads):
    stellar.fit(prepared, cache=BASIS_CACHE)
print(stellar.cache_path)
print(stellar.diagnostics)
