"""Selection and preparation of NMF inputs; no optimizers or inference here."""

from dataclasses import asdict, dataclass
import hashlib
import os

import numpy as np
from scipy import sparse
from tqdm import tqdm

from .nmf_cache import CacheMiss, NMFCache, array_digest, canonical

PREPARATION_VERSION = 1


@dataclass
class PreparedSpectra:
    optical_depth: object
    wavelengths: np.ndarray
    preprocessing: dict
    source_manifest: dict


def _indices(n, selection):
    if selection is None:
        return np.arange(n, dtype=np.int64)
    rows = np.asarray(selection)
    if rows.dtype.kind == 'b':
        if rows.shape != (n,):
            raise ValueError('Selection mask must have one entry per spectrum')
        rows = np.flatnonzero(rows)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu':
        raise ValueError('Selection must be a one-dimensional integer index array or boolean mask')
    if len(rows) == 0 or np.any(rows < 0) or np.any(rows >= n) or len(np.unique(rows)) != len(rows):
        raise ValueError('Selection must be nonempty, in range, and contain no duplicate rows')
    return rows.astype(np.int64)


def _limit_bosz_rows(spectra, rows):
    """Apply the loader's debug cap to the actual NMF input, without changing its IDs."""
    limit = getattr(spectra, 'debug_max_F_spectra', None)
    return rows if limit is None else rows[:limit]


def select_bosz(spectra, *, atmos=('mp', 'ms'), teff_max=8000, teff_min=None, indices=None):
    """Filter row indices, then apply debug_max_F_spectra; preserve loader ordering."""
    keep = []
    for row in _indices(len(spectra), indices):
        identifier = spectra.ids[row]
        teff = float(identifier.teff.removeprefix('t'))
        if atmos is not None and identifier.atmos not in atmos:
            continue
        if teff_max is not None and teff > teff_max:
            continue
        if teff_min is not None and teff < teff_min:
            continue
        keep.append(row)
    if not keep:
        raise ValueError('BOSZ selection contains no spectra')
    return _limit_bosz_rows(spectra, np.asarray(keep, dtype=np.int64))


def describe_bosz(spectra, selection=None, *, threshold=.01, floor=1e-10, matrix_format='dense'):
    if matrix_format not in ('dense', 'sparse'):
        raise ValueError("matrix_format must be 'dense' or 'sparse'")
    rows = _limit_bosz_rows(spectra, _indices(len(spectra), selection))
    # Row order is part of cache identity, not just membership in the selection.
    digest = hashlib.sha256()
    identifiers = [] if len(rows) <= 1000 else None
    for row in rows:
        identifier = asdict(spectra.ids[row])
        digest.update(canonical(identifier).encode() + b'\n')
        if identifiers is not None:
            identifiers.append(identifier)
    manifest = dict(
        kind='bosz',
        count=len(rows),
        ordered_ids_sha256=digest.hexdigest(),
        root=str(spectra.root)
    )
    # Preserve an inspectable selection for small runs, without huge basis sidecars for full grids.
    if identifiers is not None:
        manifest['ids'] = identifiers
    preprocessing = dict(
        spectra._processing_params(),
        optical_depth_threshold=float(threshold),
        flux_floor=float(floor),
        optical_depth_dtype='float32',
        optical_depth_storage=matrix_format,
        preparation_version=PREPARATION_VERSION,
        wavelength_unit='Angstrom',
        wavelength_convention='vacuum',
        reference_frame='stellar_rest'
    )
    return np.asarray(spectra.wave_out, dtype=np.float64), preprocessing, manifest, rows


def describe_telluric(spectra, selection=None):
    rows = _indices(len(spectra.M), selection)
    preprocessing = dict(
        spectra._params(),
        preparation_version=PREPARATION_VERSION,
        wavelength_unit='Angstrom',
        wavelength_convention='vacuum',
        reference_frame='observer'
    )
    path = spectra.pca_file
    manifest = dict(
        kind='pypeit',
        count=len(rows),
        rows_sha256=array_digest(rows),
        source=str(path)
    )
    if path.is_file():
        stat = path.stat()
        manifest.update(source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns)
    return np.asarray(spectra.wave_out, dtype=np.float64), preprocessing, manifest, rows


def describe_matrix(matrix, wavelengths, *, preprocessing=None, source_ids=None, selection=None):
    if len(matrix.shape) != 2:
        raise ValueError('Optical-depth training input must be a 2-D matrix')
    rows = _indices(matrix.shape[0], selection)
    wave = np.asarray(wavelengths, dtype=np.float64)
    if wave.ndim != 1 or matrix.shape[1] != len(wave):
        raise ValueError('Training matrix columns must match wavelengths')
    digest = hashlib.sha256()
    digest.update(canonical(matrix.shape).encode())
    # Bound temporary allocations when fingerprinting prepared dense or sparse matrices.
    for start in range(0, len(rows), 1024):
        batch = matrix[rows[start:start + 1024]]
        if sparse.issparse(batch):
            batch = batch.tocsr()
            digest.update(array_digest(batch.data).encode())
            digest.update(array_digest(batch.indices).encode())
            digest.update(array_digest(batch.indptr).encode())
        else:
            digest.update(array_digest(batch).encode())
    manifest = dict(kind='prepared_optical_depth', count=len(rows), data_sha256=digest.hexdigest())
    if source_ids is not None:
        if len(source_ids) != matrix.shape[0]:
            raise ValueError('source_ids must have one entry per training row')
        manifest['ids'] = [source_ids[int(row)] for row in rows]
    settings = {
        'wavelength_unit': 'Angstrom',
        'wavelength_convention': 'unspecified',
        'reference_frame': 'unspecified',
        **(preprocessing or {})
    }
    return wave, settings, manifest, rows


def prepare_bosz(
    spectra,
    selection=None,
    *,
    threshold=.01,
    floor=1e-10,
    chunk_size=1000,
    n_workers=1,
    matrix_cache=None,
    matrix_format='dense'
):
    """Build selected optical depths as float32, dense by default or explicitly CSR.

    Workers prepare missing processed flux first. Dense assembly fills one final
    training array, rather than retaining chunks and concatenating another copy.
    """
    if n_workers is None:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))
    if threshold < 0 or not 0 < floor <= 1 or chunk_size < 1 or n_workers < 1:
        raise ValueError('Invalid optical-depth preparation or worker settings')
    wave, settings, manifest, rows = describe_bosz(
        spectra,
        selection,
        threshold=threshold,
        floor=floor,
        matrix_format=matrix_format
    )
    params = dict(preprocessing=settings, source_manifest=manifest)
    cache = matrix_cache if isinstance(matrix_cache, NMFCache) else NMFCache(matrix_cache) if matrix_cache is not None else None
    if cache is not None:
        try:
            matrix = cache.load_matrix(params, storage=matrix_format)
            if matrix.dtype != np.float32:
                raise ValueError('Prepared BOSZ cache must contain float32 optical depth')
            tqdm.write(f'Loaded {matrix_format} float32 BOSZ matrix: {matrix.shape[0]:,} spectra, {matrix.shape[1]:,} pixels')
            return PreparedSpectra(matrix, wave, settings, manifest)
        except CacheMiss:
            pass
    if n_workers > 1:
        if not spectra.use_cache:
            raise ValueError('Parallel BOSZ preparation requires a cache_dir')
        # Compute only missing selected flux in workers; the parent owns all cache writes.
        spectra.process_spectra([spectra.ids[int(i)] for i in rows], n_workers=n_workers)

    # Dense mode fills one allocation; only sparse mode retains CSR chunks.
    matrix = np.empty((len(rows), len(wave)), dtype=np.float32) if matrix_format == 'dense' else None
    chunks = []
    if matrix is not None:
        tqdm.write(f'Preparing dense float32 BOSZ matrix ({matrix.nbytes / 1024**3:.2f} GiB)')
    with tqdm(total=len(rows), desc='Preparing selected BOSZ spectra', unit='spectrum') as progress:
        for start in range(0, len(rows), chunk_size):
            batch = rows[start:start + chunk_size]
            optical_depth = spectra.optical_depth(
                batch,
                tol=threshold,
                floor=floor,
                progress=progress.update
            )
            if matrix is not None:
                # Assignment casts only this batch; no intermediate sparse matrix is built.
                matrix[start:start + len(batch)] = optical_depth
            else:
                chunks.append(sparse.csr_matrix(optical_depth, dtype=np.float32))
    if matrix is None:
        matrix = sparse.vstack(chunks, format='csr')
    if cache is not None:
        cache.save_matrix(matrix, params)
    return PreparedSpectra(matrix, wave, settings, manifest)
