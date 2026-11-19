"""Versioned NMF artifacts, independent of training data and optimizers."""

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile

import numpy as np
import scipy
from scipy import sparse
import sklearn

SCHEMA_VERSION = 2


class CacheMiss(FileNotFoundError):
    """No complete artifact matches the requested parameters."""


class CacheValidationError(ValueError):
    """An explicit artifact is incomplete, corrupt, or incompatible."""


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical(value):
    return json.dumps(value, sort_keys=True, default=_json_default, allow_nan=False)


def array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(canonical(array.shape).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_basis(basis, wavelengths):
    basis, wavelengths = np.asarray(basis), np.asarray(wavelengths)
    if basis.ndim != 2 or min(basis.shape) < 1 or wavelengths.ndim != 1:
        raise CacheValidationError('Expected a nonempty (components, pixels) basis and 1-D wavelengths')
    if basis.dtype.kind != 'f' or wavelengths.dtype.kind != 'f':
        raise CacheValidationError('Basis and wavelengths must have floating-point dtypes')
    if basis.shape[1] != len(wavelengths) or len(wavelengths) < 2:
        raise CacheValidationError('Basis shape does not match wavelength grid')
    if not np.isfinite(basis).all() or np.any(basis < 0):
        raise CacheValidationError('Basis must be finite and non-negative')
    if not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0) or np.any(np.diff(wavelengths) <= 0):
        raise CacheValidationError('Wavelengths must be positive, finite, and strictly increasing')


@dataclass
class NMFArtifact:
    basis: np.ndarray
    wavelengths: np.ndarray
    metadata: dict
    path: Path


class NMFCache:
    """Job-ID filenames for basis artifacts; exact parameter checks for reuse.

    Constructors and lookups do not create directories. Saving uses a directory lock,
    atomic file replacements, checksums, and a JSON completion record written last.
    """

    def __init__(self, directory):
        self.directory = Path(directory)

    @staticmethod
    def _read_json(path):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise CacheValidationError(f'Cannot read metadata: {path}') from exc
        if not isinstance(value, dict):
            raise CacheValidationError(f'Metadata must be an object: {path}')
        return value

    @staticmethod
    def _params(metadata):
        value = metadata['params'] if 'schema_version' in metadata else metadata
        if not isinstance(value, dict):
            raise CacheValidationError('Artifact parameters must be a dictionary')
        return value

    def _paths(self, name, params, suffix='.npy', *, use_job_id=True):
        if not re.fullmatch(r'[A-Za-z0-9_]+', name):
            raise ValueError('Cache name must contain only letters, numbers, and underscores')
        # Matching parameters may reuse a filename; other configurations get a suffix.
        job_id = os.environ.get('SLURM_JOB_ID', '').strip()
        if use_job_id and suffix == '.npy' and re.fullmatch(r'[0-9]+', job_id):
            index = 1
            while True:
                tag = job_id if index == 1 else f'{job_id}_{index}'
                path = self.directory / f'{name}_{tag}.npy'
                sidecar = path.with_suffix('.json')
                wave_path = path.with_suffix('.wavelengths.npy')
                if not any(p.exists() for p in (path, sidecar, wave_path)):
                    return path, sidecar
                try:
                    if canonical(self._params(self._read_json(sidecar))) == canonical(params):
                        return path, sidecar
                except CacheValidationError:
                    pass
                index += 1
        key = hashlib.sha1(canonical(params).encode()).hexdigest()[:10]
        return self.directory / f'{name}_{key}{suffix}', self.directory / f'{name}_{key}.json'

    def _candidates(self, name, params):
        preferred, _ = self._paths(name, params)
        # Include pre-refactor hashes (json.dumps used default separators, as canonical does).
        yield preferred
        for sidecar in sorted(self.directory.glob(f'{name}_*.json')):
            tag = sidecar.stem[len(name) + 1:]
            if re.fullmatch(r'(?:[0-9]+(?:_[0-9]+)?|[0-9a-f]{10})', tag):
                yield sidecar.with_suffix('.npy')

    @staticmethod
    def _atomic(path, writer):
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            # Replace only after the complete file has been flushed to disk.
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def save(
        self,
        name,
        basis,
        wavelengths,
        *,
        params,
        role,
        preprocessing=None,
        source_manifest=None,
        diagnostics=None,
        artifact_kind='basis'
    ):
        if role not in ('stellar', 'telluric') or artifact_kind not in ('basis', 'checkpoint'):
            raise ValueError('Unknown role or artifact kind')
        basis, wavelengths = np.asarray(basis), np.asarray(wavelengths, dtype=np.float64)
        validate_basis(basis, wavelengths)
        if not isinstance(params, dict):
            raise ValueError('Artifact parameters must be a dictionary')
        canonical(params)  # Fail before touching files if settings cannot be serialized.
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / '.nmf_cache.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path, sidecar = self._paths(name, params)
            wave_path = path.with_suffix('.wavelengths.npy')
            self._atomic(path, lambda f: np.save(f, basis, allow_pickle=False))
            self._atomic(wave_path, lambda f: np.save(f, wavelengths, allow_pickle=False))
            arrays = {}
            for label, array, filename in [
                ('basis', basis, path),
                ('wavelengths', wavelengths, wave_path)
            ]:
                arrays[label] = dict(
                    filename=filename.name,
                    shape=list(array.shape),
                    dtype=str(array.dtype),
                    sha256=file_digest(filename)
                )
            metadata = dict(
                schema_version=SCHEMA_VERSION,
                artifact_kind=artifact_kind,
                role=role,
                params=params,
                arrays=arrays,
                preprocessing=preprocessing or {},
                source_manifest=source_manifest or {},
                diagnostics=diagnostics or {},
                provenance=dict(
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    thread_environment={
                        key: os.environ.get(key)
                        for key in (
                            'OMP_NUM_THREADS',
                            'OPENBLAS_NUM_THREADS',
                            'MKL_NUM_THREADS',
                            'SLURM_CPUS_PER_TASK',
                            'NMF_THREADS'
                        )
                    },
                    versions=dict(
                        python=platform.python_version(),
                        numpy=np.__version__,
                        scipy=scipy.__version__,
                        sklearn=sklearn.__version__
                    )
                )
            )
            payload = json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
                default=_json_default,
                allow_nan=False
            ).encode()
            # The JSON sidecar is the completion record, so publish it last.
            self._atomic(sidecar, lambda f: f.write(payload))
        return path

    def _read_artifact(self, path, *, role=None, params=None, allow_checkpoint=False):
        path = Path(path)
        if path.suffix != '.npy' or not path.is_file():
            raise CacheValidationError(f'Missing basis array: {path}')
        metadata = self._read_json(path.with_suffix('.json'))
        try:
            saved_params = self._params(metadata)
        except KeyError as exc:
            raise CacheValidationError(f'Missing artifact parameters: {path}') from exc
        if params is not None and canonical(saved_params) != canonical(params):
            raise CacheValidationError(f'Parameters do not match {path}')
        try:
            if 'schema_version' not in metadata:
                # Legacy F/G sidecars stored only parameters and assumed a logarithmic grid.
                saved_role = 'stellar' if 'bosz' in metadata else 'telluric' if 'telluric' in metadata else None
                if saved_role is None or (role is not None and role != saved_role):
                    raise CacheValidationError(f'Cannot identify legacy basis role: {path}')
                grid = metadata['bosz' if saved_role == 'stellar' else 'telluric']
                wave = np.geomspace(grid['wave_min'], grid['wave_max'], grid['D'])
                basis = np.load(path, allow_pickle=False)
                metadata = dict(
                    schema_version=0,
                    artifact_kind='basis',
                    role=saved_role,
                    params=saved_params,
                    preprocessing=grid,
                    source_manifest={},
                    diagnostics={},
                    provenance=dict(legacy=True, wavelengths_inferred=True)
                )
            else:
                if metadata['schema_version'] != SCHEMA_VERSION:
                    raise CacheValidationError(f'Unsupported artifact schema: {metadata["schema_version"]}')
                for field in (
                    'params',
                    'arrays',
                    'preprocessing',
                    'source_manifest',
                    'diagnostics',
                    'provenance'
                ):
                    if not isinstance(metadata.get(field), dict):
                        raise CacheValidationError(f'Missing or invalid artifact field: {field}')
                if metadata['role'] not in ('stellar', 'telluric') or (role is not None and metadata['role'] != role):
                    raise CacheValidationError(f'Basis role does not match {path}')
                if metadata['artifact_kind'] != 'basis' and not (allow_checkpoint and metadata['artifact_kind'] == 'checkpoint'):
                    raise CacheValidationError(f'Artifact is not a completed basis: {path}')
                loaded = {}
                for label in ('basis', 'wavelengths'):
                    info = metadata['arrays'][label]
                    array_path = path.parent / info['filename']
                    if array_path.parent != path.parent or array_path.name != info['filename']:
                        raise CacheValidationError('Array filenames must be local to the artifact directory')
                    if label == 'basis' and array_path != path:
                        raise CacheValidationError('Metadata refers to a different basis file')
                    if file_digest(array_path) != info['sha256']:
                        raise CacheValidationError(f'Checksum mismatch: {array_path}')
                    array = np.load(array_path, allow_pickle=False)
                    if list(array.shape) != info['shape'] or str(array.dtype) != info['dtype']:
                        raise CacheValidationError(f'Array shape/dtype mismatch: {array_path}')
                    loaded[label] = array
                basis, wave = loaded['basis'], loaded['wavelengths']
            validate_basis(basis, wave)
            expected_components = saved_params.get('training', {}).get('n_components')
            if expected_components is not None and basis.shape[0] != expected_components:
                raise CacheValidationError('Component count does not match training metadata')
            expected_wave = saved_params.get('wavelength_sha256')
            if expected_wave is not None and array_digest(wave) != expected_wave:
                raise CacheValidationError('Wavelength grid does not match training metadata')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, CacheValidationError):
                raise
            raise CacheValidationError(f'Invalid artifact: {path}: {exc}') from exc
        return NMFArtifact(basis, wave, metadata, path)

    def load(self, filepath=None, *, name=None, params=None, role=None, allow_checkpoint=False):
        if filepath is not None:
            return self._read_artifact(
                filepath,
                role=role,
                params=params,
                allow_checkpoint=allow_checkpoint
            )
        if name is None or params is None:
            raise ValueError('Cache search requires a name and complete matching parameters')
        failures = []
        for path in dict.fromkeys(self._candidates(name, params)):
            if not path.is_file():
                continue
            try:
                return self._read_artifact(
                    path,
                    role=role,
                    params=params,
                    allow_checkpoint=allow_checkpoint
                )
            except CacheValidationError as exc:
                failures.append(str(exc))
        detail = f'; last rejection: {failures[-1]}' if failures else ''
        raise CacheMiss(f'No compatible {name} artifact in {self.directory}{detail}')

    def save_matrix(self, matrix, params):
        """Cache prepared input by settings/selection hash, independent of job ID."""
        self.directory.mkdir(parents=True, exist_ok=True)
        is_sparse = sparse.issparse(matrix)
        name, suffix = ('sparse_S_log', '.npz') if is_sparse else ('dense_S_log', '.npy')
        path, sidecar = self._paths(name, params, suffix, use_job_id=False)
        if is_sparse:
            self._atomic(path, lambda f: sparse.save_npz(f, matrix, compressed=False))
        else:
            matrix = np.asarray(matrix)
            self._atomic(path, lambda f: np.save(f, matrix, allow_pickle=False))
        metadata = dict(
            params=params,
            shape=list(matrix.shape),
            dtype=str(matrix.dtype),
            sha256=file_digest(path)
        )
        self._atomic(sidecar, lambda f: f.write(canonical(metadata).encode()))
        return path

    def load_matrix(self, params, *, storage=None):
        """Load dense input as a read-only memory map, or an explicitly sparse cache.

        With storage=None, prefer dense input and fall back to older sparse caches.
        BOSZ preparation requests storage='dense' to avoid reintroducing CSR data.
        """
        if storage not in (None, 'dense', 'sparse'):
            raise ValueError("storage must be 'dense', 'sparse', or None")
        formats = ['dense', 'sparse'] if storage is None else [storage]
        for kind in formats:
            name, suffix = ('dense_S_log', '.npy') if kind == 'dense' else ('sparse_S_log', '.npz')
            path, sidecar = self._paths(name, params, suffix, use_job_id=False)
            if not path.is_file() or not sidecar.is_file():
                continue
            metadata = self._read_json(sidecar)
            if canonical(metadata.get('params')) != canonical(params) or file_digest(path) != metadata.get('sha256'):
                raise CacheValidationError(f'Prepared-matrix metadata/checksum mismatch: {path}')
            matrix = np.load(path, mmap_mode='r', allow_pickle=False) if kind == 'dense' else sparse.load_npz(path).tocsr()
            if list(matrix.shape) != metadata['shape'] or str(matrix.dtype) != metadata['dtype']:
                raise CacheValidationError(f'Prepared-matrix shape/dtype mismatch: {path}')
            return matrix
        raise CacheMiss(f'No prepared {storage or "dense or sparse"} matrix in {self.directory}')
