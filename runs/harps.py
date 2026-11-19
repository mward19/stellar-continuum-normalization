"""Shared HARPS grid, selection, trainer, and prepared-matrix cache settings."""
from pathlib import Path
import numpy as np

from src.load_spectra import BOSZSpectra
from src.model import SpectralNMF
from src.nmf_cache import NMFCache
from src.preparation import describe_bosz, select_bosz


HOME = Path('/home/mward19')
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('cache-high-res')
RAW_SPECTRA = HOME / 'nobackup/autodelete/bosz-spectra/data'
PROCESSED_SPECTRA = HOME / 'nobackup/autodelete/bosz-spectra/processed-harps' / CACHE
PREPARED_MATRIX = REPO_ROOT / CACHE / 'harps_prepared'
BASIS_CACHE = REPO_ROOT / CACHE / 'stellar'


def load_harps_training_data(matrix_format='dense', row_stride=1):
    """Return the BOSZ loader, selected MARCS rows, trainer, and matrix cache."""
    if not isinstance(row_stride, int) or isinstance(row_stride, bool) or row_stride < 1:
        raise ValueError('row_stride must be a positive integer')
    spectra = BOSZSpectra(
        root=RAW_SPECTRA,
        R_in=300_000,
        R_out=115_000,
        D=None,
        wave_min=3000,
        wave_max=7000,
        cache_dir=PROCESSED_SPECTRA,
    )
    selection = select_bosz(spectra)
    # The paper explicitly allows skipping every n-th grid spectrum when memory
    # is limited. Sort by the full BOSZ identifier first so the same subset is
    # selected in preprocessing and training, independent of filesystem order.
    selection = np.asarray(
        sorted(selection, key=lambda row: tuple(vars(spectra.ids[int(row)]).values())),
        dtype=np.int64,
    )[::row_stride]
    trainer = SpectralNMF(
        n_components=16,
        max_iter=10_000,
        tol=1e-4,
        sparse_tol=0.0,
        matrix_format=matrix_format,
        verbose=1,
    )
    return spectra, selection, trainer, NMFCache(PREPARED_MATRIX)


def prepared_matrix_params(spectra, selection, trainer):
    """Return the exact preprocessing-cache key used by prepare_bosz()."""
    _, preprocessing, source_manifest, _ = describe_bosz(
        spectra,
        selection,
        threshold=trainer.config['sparse_tol'],
        floor=trainer.config['sparse_floor'],
        matrix_format=trainer.config['matrix_format'],
    )
    return dict(preprocessing=preprocessing, source_manifest=source_manifest)
