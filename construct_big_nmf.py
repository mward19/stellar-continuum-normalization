"""Explicitly load compatible NMF artifacts or train them in a Slurm job."""

from pathlib import Path

from src.load_spectra import BOSZSpectra
from src.load_telluric import TelluricSpectra
from src.model import SpectralNMF, TelluricNMF, NMFCache, CacheMiss
from src.preparation import select_bosz


def load_or_fit(model, data, cache, **kwargs):
    params = model.parameters_for(data, **kwargs)
    try:
        fitted = type(model).load(cache=cache, params=params)
        print(f'Loaded {fitted.role} basis: {fitted.cache_path}', flush=True)
        return fitted
    except CacheMiss:
        print(f'Training {model.role} basis', flush=True)
        fitted = model.fit(data, cache=cache, matrix_cache=cache, **kwargs)
        print(f'Cached {fitted.role} basis: {fitted.cache_path}', flush=True)
        print(f'Diagnostics: {fitted.diagnostics}', flush=True)
        return fitted


def main():
    bosz_root = Path('/home/mward19/nobackup/autodelete/bosz-spectra')
    telluric_root = Path('/home/mward19/nobackup/autodelete/telluric')

    # These settings match the existing production cache.
    # sandbox/nmf_subset.ipynb explores finer sampling in a separate output directory.
    bosz = BOSZSpectra(
        bosz_root / 'data',
        300_000,
        115_000,
        D=3000,
        wave_min=3000,
        wave_max=7000,
        cache_dir=bosz_root / 'processed-harps',
        spectra_per_file=1000
    )
    selection = select_bosz(bosz, atmos=('mp', 'ms'), teff_max=None)
    stellar = load_or_fit(
        SpectralNMF(
            max_iter=500,
            seed=0,
            tol=1e-4,
            init='nndsvda',
            sparse_tol=1e-2,
            verbose=1
        ),
        bosz,
        NMFCache(bosz.cache_dir),
        selection=selection
    )

    # The telluric basis has its own processed-data and NMF caches.
    telluric = TelluricSpectra(
        telluric_root / 'TellPCA_3000_10500_R120000.fits',
        D=3000,
        wave_min=3000,
        wave_max=7000,
        R_out=115_000,
        n_pca=4,
        t_floor=1e-10,
        cache_dir=telluric_root / 'processed'
    )
    terrestrial = load_or_fit(TelluricNMF(), telluric, NMFCache(telluric.cache_dir))
    return stellar, terrestrial


if __name__ == '__main__':
    main()
