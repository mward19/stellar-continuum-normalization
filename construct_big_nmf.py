from pathlib import Path

from src.load_spectra import BOSZSpectra
from src.load_telluric import TelluricSpectra
from src.model import ContinuumNormalizer

import numpy as np
from matplotlib import pyplot as plt

bosz_root = Path("/home/mward19/nobackup/autodelete/bosz-spectra")
telluric_root = Path("/home/mward19/nobackup/autodelete/telluric")

# Same parameters as the caches the NMFs were built from, so nothing is recomputed
bosz = BOSZSpectra(
    bosz_root / Path("data"),
    300_000,
    115_000,
    D=3000,
    wave_min=3000,
    wave_max=7000,
    cache_dir=bosz_root / Path("processed-harps"),
    # debug_max_F_spectra=10000,
    spectra_per_file=1000,
)
telluric = TelluricSpectra(
    telluric_root / Path("TellPCA_3000_10500_R120000.fits"),
    D=3000,
    wave_min=3000,
    wave_max=7000,
    R_out=115_000,
    n_pca=4,
    t_floor=1e-10,
    cache_dir=telluric_root / Path("processed"),
)

wave = bosz.wave_out
# Match Casey et al.'s stellar NMF setup: MARCS spectra, K=16, coordinate descent, and 10,000
# iterations. tol=0 disables early stopping so max_iter is the exact iteration count.
cn = ContinuumNormalizer(
    bosz,
    telluric,
    flux_errors=1e-8 * np.ones(3000),
    order_wavelengths=wave,
    F_filter=dict(atmos=("mp", "ms"), teff_max=None),
    F_fit_method="get_F_full",
    F_kwargs=dict(
        max_iter=500,
        seed=0,
        tol=1e-4,
        init="nndsvda",
        sparse_tol=1e-2,
    ),
)

