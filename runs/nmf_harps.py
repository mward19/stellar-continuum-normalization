import argparse
import os
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.load_spectra import BOSZSpectra
from src.model import SpectralNMF
from src.preparation import select_bosz

parser = argparse.ArgumentParser(description="Train the HARPS stellar NMF basis")
parser.add_argument("--matrix-format", choices=("dense", "sparse"), default="dense",
                    help="Training matrix representation (default: dense float32)")
args = parser.parse_args()

HOME_DIR = Path('/home/mward19')

print("Beginning processing")

spectra = BOSZSpectra(
    root=HOME_DIR / Path("nobackup/autodelete/bosz-spectra/data"),
    R_in=300_000,
    R_out=115_000,
    D=None,  # Automatically sample R_out at three points per FWHM.
    wave_min=3800,
    wave_max=4200,
    # debug_max_F_spectra=1000,
    cache_dir=HOME_DIR / Path("nobackup/autodelete/bosz-spectra/processed-harps/cache"),
)
selection = select_bosz(spectra)  # MARCS atmospheres, Teff <= 8000 K, capped by debug limit.
print(f"Selected {len(selection):,} spectra for NMF", flush=True)

stellar = SpectralNMF(
    n_components=16, max_iter=500, tol=1e-4, sparse_tol=0.0,
    matrix_format=args.matrix_format, verbose=1,
)
allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
n_workers = min(64, allocated_cpus)
nmf_threads = min(allocated_cpus, int(os.environ.get("NMF_THREADS", "8")))
if min(n_workers, nmf_threads) < 1:
    raise ValueError("CPU allocation and NMF_THREADS must be positive")
print(f"Preparing spectra with {n_workers} workers; NMF uses {nmf_threads} numerical threads "
      f"and {args.matrix_format} storage", flush=True)
with threadpool_limits(limits=nmf_threads):
    stellar.fit(spectra, selection=selection, cache="cache/stellar", n_workers=n_workers)
print(stellar.cache_path)
print(stellar.diagnostics)
