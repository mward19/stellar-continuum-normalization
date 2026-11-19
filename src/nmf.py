"""Explicit fitting/loading of stellar and telluric optical-depth bases."""

import time
import warnings

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from .nmf_cache import NMFCache, array_digest, validate_basis
from .preparation import (
    PreparedSpectra,
    describe_bosz,
    describe_telluric,
    describe_matrix,
    prepare_bosz,
    select_bosz
)


class _AbsorptionNMF:
    role = None

    def __init__(
        self,
        n_components,
        *,
        solver='cd',
        max_iter=10_000,
        tol=None,
        init='nndsvda',
        seed=0,
        shuffle=True,
        max_epochs=1000,
        batch_size=256,
        tol_batches=3,
        check_interval=50,
        sparse_tol=None,
        sparse_floor=1e-10,
        matrix_format='dense',
        verbose=0
    ):
        if solver not in ('cd', 'minibatch'):
            raise ValueError("solver must be 'cd' or 'minibatch'")
        if matrix_format not in ('dense', 'sparse'):
            raise ValueError("matrix_format must be 'dense' or 'sparse'")

        # Resolve defaults before recording parameters for cache lookup.
        if tol is None:
            tol = 0. if solver == 'cd' else 1e-4
        counts = (n_components, max_iter, max_epochs, batch_size, tol_batches, check_interval)
        if any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or value < 1
            for value in counts
        ) or not np.isfinite(tol) or tol < 0:
            raise ValueError('Component counts, iteration limits, and intervals must be positive; tol >= 0')
        if sparse_tol is None:
            sparse_tol = .01 if solver == 'cd' else 1e-6
        if sparse_tol < 0 or not 0 < sparse_floor <= 1:
            raise ValueError('Invalid sparse optical-depth settings')
        self.config = dict(
            n_components=int(n_components),
            solver=solver,
            max_iter=int(max_iter),
            tol=float(tol),
            init=init,
            seed=int(seed),
            shuffle=bool(shuffle),
            max_epochs=int(max_epochs),
            batch_size=int(batch_size),
            tol_batches=int(tol_batches),
            check_interval=int(check_interval),
            sparse_tol=float(sparse_tol),
            sparse_floor=float(sparse_floor),
            matrix_format=matrix_format,
            verbose=int(verbose)
        )

        # Fitted state is populated by fit() or load().
        self.components_ = None
        self.wavelengths = None
        self.params = None
        self.preprocessing = {}
        self.source_manifest = {}
        self.diagnostics = {}
        self.cache_path = None
        self.metadata = None

    @property
    def n_components(self):
        return self.config['n_components']

    @property
    def cache_name(self):
        if self.role == 'telluric':
            return 'nmf_G' if self.config['solver'] == 'cd' else 'nmf_G_batch'
        return 'nmf_F_full' if self.config['solver'] == 'cd' else 'nmf_F'

    def _describe(
        self,
        data,
        wavelengths=None,
        *,
        selection=None,
        preprocessing=None,
        source_ids=None
    ):
        if isinstance(data, PreparedSpectra):
            if selection is not None or wavelengths is not None or preprocessing is not None or source_ids is not None:
                raise ValueError('PreparedSpectra already defines its selection, wavelengths and provenance')
            return data.wavelengths, data.preprocessing, data.source_manifest, None
        if hasattr(data, 'ids') and hasattr(data, 'wave_out'):
            if self.role != 'stellar':
                raise ValueError('BOSZ inputs require SpectralNMF')
            return describe_bosz(
                data,
                selection,
                threshold=self.config['sparse_tol'],
                floor=self.config['sparse_floor'],
                matrix_format=self.config['matrix_format']
            )
        if hasattr(data, 'M') and hasattr(data, 'wave_out'):
            if self.role != 'telluric':
                raise ValueError('Telluric inputs require TelluricNMF')
            return describe_telluric(data, selection)
        if wavelengths is None:
            raise ValueError('Prepared optical-depth matrices require explicit wavelengths')
        return describe_matrix(
            data,
            wavelengths,
            preprocessing=preprocessing,
            source_ids=source_ids,
            selection=selection
        )

    def parameters_for(self, data, wavelengths=None, **kwargs):
        """Compute complete cache identity without fitting or preparing raw BOSZ flux."""
        wave, preprocessing, manifest, _ = self._describe(data, wavelengths, **kwargs)
        return self._parameters(wave, preprocessing, manifest)

    def _parameters(self, wave, preprocessing, manifest):
        return dict(
            role=self.role,
            algorithm_version=2,
            training={k: v for k, v in self.config.items() if k != 'verbose'},
            wavelength_sha256=array_digest(np.asarray(wave, dtype=np.float64)),
            preprocessing=preprocessing,
            source_manifest=manifest
        )

    def fit(
        self,
        data,
        wavelengths=None,
        *,
        selection=None,
        preprocessing=None,
        source_ids=None,
        cache=None,
        matrix_cache=None,
        n_workers=1,
        residual_samples=20_000
    ):
        """Train explicitly; dense/sparse matrix inputs are already negative-log flux.

        BOSZ and TelluricSpectra inputs are prepared using their own processing settings.
        matrix_format controls dense/CSR computation for every input type.
        Matrix values are used as supplied: thresholding belongs to preprocessing.
        Configured caches save results; this method never silently substitutes a cached fit.
        """
        wave, settings, manifest, rows = self._describe(
            data,
            wavelengths,
            selection=selection,
            preprocessing=preprocessing,
            source_ids=source_ids
        )
        params = self._parameters(wave, settings, manifest)
        if isinstance(data, PreparedSpectra):
            X = data.optical_depth
        elif hasattr(data, 'ids') and hasattr(data, 'wave_out'):
            X = prepare_bosz(
                data,
                rows,
                threshold=self.config['sparse_tol'],
                floor=self.config['sparse_floor'],
                n_workers=n_workers,
                matrix_cache=matrix_cache,
                matrix_format=self.config['matrix_format']
            ).optical_depth
        elif hasattr(data, 'M') and hasattr(data, 'wave_out'):
            X = data.M if selection is None else data.M[rows]
        else:
            X = data if selection is None else data[rows]
        if self.config['matrix_format'] == 'sparse':
            X = sparse.csr_matrix(X)
        else:
            X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        if X.ndim != 2 or min(X.shape) < 1 or X.shape[1] != len(wave):
            raise ValueError('Training requires a finite non-negative matrix matching the wavelength grid')

        # Scan bounded blocks rather than allocating a matrix-sized validity mask.
        values = X.data if sparse.issparse(X) else X
        block_size = 1_000_000 if sparse.issparse(X) else 1024
        for start in range(0, len(values), block_size):
            block = values[start:start + block_size]
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError('Training requires a finite non-negative matrix matching the wavelength grid')
        validate_basis(np.zeros((1, len(wave))), np.asarray(wave, dtype=np.float64))
        if X.dtype not in (np.dtype('float32'), np.dtype('float64')):
            X = X.astype(np.float64)
        if self.config['init'] in ('nndsvd', 'nndsvda', 'nndsvdar') and self.n_components > min(X.shape):
            raise ValueError('NNDSVD requires n_components <= min(training rows, pixels)')
        if residual_samples < 1:
            raise ValueError('residual_samples must be positive')
        self.components_ = None
        self.params, self.preprocessing, self.source_manifest = params, settings, manifest
        self.wavelengths = np.asarray(wave, dtype=np.float64).copy()
        self.metadata, self.cache_path = None, None
        started = time.perf_counter()

        # Retain the fitted weights so CD diagnostics do not launch a second solve.
        training_weights = None
        if self.config['solver'] == 'cd':
            model = NMF(
                n_components=self.n_components,
                init=self.config['init'],
                solver='cd',
                max_iter=self.config['max_iter'],
                tol=self.config['tol'],
                random_state=self.config['seed'],
                shuffle=self.config['shuffle'],
                verbose=self.config['verbose']
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                training_weights = model.fit_transform(X)
            iteration_limit = any(issubclass(item.category, ConvergenceWarning) for item in caught)
            self.diagnostics = dict(
                n_iter=int(model.n_iter_),
                iteration_limit_reached=bool(model.n_iter_ >= self.config['max_iter']),
                convergence_warning=iteration_limit,
                converged=bool(model.n_iter_ < self.config['max_iter'])
            )
            for item in caught:
                warnings.warn(item.message, item.category, stacklevel=2)
        else:
            model, diagnostics = self._fit_minibatch(X, cache)
            self.diagnostics = diagnostics
        self.components_ = model.components_.copy()
        validate_basis(self.components_, self.wavelengths)

        # Keep reconstruction quality separate from the optimizer's stopping status.
        self.diagnostics['fit_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        self.diagnostics.update(self._residuals(X, residual_samples, weights=training_weights))
        self.diagnostics['evaluation_seconds'] = time.perf_counter() - started
        if cache is not None:
            self.save(cache)
        return self

    def _fit_minibatch(self, X, cache):
        cfg = self.config
        # Match the previous partial_fit procedure: initialization defaults to sklearn's choice.
        model = MiniBatchNMF(
            n_components=self.n_components,
            forget_factor=1,
            random_state=cfg['seed'],
            init=cfg['init'],
            verbose=False
        )
        rng = np.random.default_rng(cfg['seed'])

        # Stop only after several consecutive checks find small component changes.
        previous = None
        batches_seen = consecutive = 0
        converged = False
        for epoch in tqdm(
            range(cfg['max_epochs']),
            desc=f'{self.role} NMF',
            unit='epoch',
            disable=not cfg['verbose']
        ):
            order = rng.permutation(X.shape[0])
            for start in range(0, len(order), cfg['batch_size']):
                rows = np.sort(order[start:start + cfg['batch_size']])
                model.partial_fit(X[rows])
                if previous is None:
                    previous = model.components_.copy()
                batches_seen += 1
                if batches_seen % cfg['check_interval']:
                    continue
                # Preserve RNG consumption from the old convergence diagnostic.
                rng.choice(X.shape[0], size=min(cfg['batch_size'], X.shape[0]), replace=False)
                components = model.components_
                change = np.linalg.norm(components - previous) / max(
                    np.linalg.norm(components),
                    np.finfo(components.dtype).eps
                )
                previous = components.copy()
                consecutive = consecutive + 1 if change <= cfg['tol'] else 0
                if consecutive >= cfg['tol_batches']:
                    converged = True
                    break
            self.components_ = model.components_.copy()
            if cache is not None:
                self.save(
                    cache,
                    name=self.cache_name + '_last_epoch',
                    artifact_kind='checkpoint',
                    diagnostics=dict(
                        epochs=epoch + 1,
                        batches_seen=batches_seen,
                        converged=converged
                    )
                )
            if converged:
                break
        return model, dict(
            epochs=epoch + 1,
            batches_seen=batches_seen,
            converged=converged,
            iteration_limit_reached=not converged
        )

    def _require_fitted(self):
        if self.components_ is None:
            raise ValueError('Call fit() or load() before using this basis')

    def evaluate(self, wavelengths):
        """Interpolate optical-depth basis rows at fixed effective resolution; no broadening."""
        self._require_fitted()
        wave = np.asarray(wavelengths, dtype=np.float64)
        if wave.ndim != 1 or not np.isfinite(wave).all():
            raise ValueError('Evaluation wavelengths must be a finite 1-D array')
        if np.any(wave < self.wavelengths[0]) or np.any(wave > self.wavelengths[-1]):
            raise ValueError('Evaluation wavelengths extend beyond the trained basis')
        return np.stack([np.interp(wave, self.wavelengths, f) for f in self.components_])

    def transform(self, optical_depth, *, wavelengths=None, max_iter=1000, tol=1e-6):
        """Solve non-negative weights with the basis fixed, without retraining it."""
        self._require_fitted()
        basis = self.components_ if wavelengths is None else self.evaluate(wavelengths)
        X = optical_depth.astype(np.float64) if sparse.issparse(optical_depth) else np.asarray(optical_depth, dtype=np.float64)
        if self.config['matrix_format'] == 'sparse':
            X = sparse.csr_matrix(X)
        elif sparse.issparse(X):
            X = X.toarray()
        if X.ndim != 2 or X.shape[1] != basis.shape[1]:
            raise ValueError('Optical-depth input columns must match evaluated basis')
        weights, _, _ = non_negative_factorization(
            X,
            H=np.asarray(basis, dtype=np.float64),
            n_components=len(basis),
            update_H=False,
            solver='cd',
            max_iter=max_iter,
            tol=tol,
            random_state=self.config['seed'],
            shuffle=self.config['shuffle']
        )
        return weights

    def _residuals(self, X, n_samples, *, weights=None):
        self._require_fitted()
        # Use the same sampled rows when comparing fits with the same seed.
        rng = np.random.default_rng(self.config['seed'])
        rows = np.sort(rng.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False))
        error = norm = 0.
        for start in range(0, len(rows), 1024):
            batch = X[rows[start:start + 1024]]
            target = batch.toarray().astype(np.float64) if sparse.issparse(batch) else np.asarray(batch, dtype=np.float64)
            # CD already fitted W and H together: evaluate that reconstruction directly.
            # Minibatch training does not retain weights for the complete training set.
            batch_weights = self.transform(batch) if weights is None else weights[rows[start:start + 1024]]
            residual = target - np.asarray(batch_weights, dtype=np.float64) @ self.components_.astype(np.float64)
            error += float(np.sum(residual**2))
            norm += float(np.sum(target**2))
        return dict(
            residual_samples=len(rows),
            residual_weight_source='fixed_basis_transform' if weights is None else 'training_fit',
            relative_frobenius_error=float(np.sqrt(error / norm)) if norm else None,
            rms_optical_depth=float(np.sqrt(error / (len(rows) * X.shape[1])))
        )

    def save(self, cache, *, name=None, artifact_kind='basis', diagnostics=None):
        self._require_fitted()
        cache = cache if isinstance(cache, NMFCache) else NMFCache(cache)
        path = cache.save(
            name or self.cache_name,
            self.components_,
            self.wavelengths,
            params=self.params,
            role=self.role,
            preprocessing=self.preprocessing,
            source_manifest=self.source_manifest,
            diagnostics=self.diagnostics if diagnostics is None else diagnostics,
            artifact_kind=artifact_kind
        )
        if artifact_kind == 'basis':
            self.cache_path = path
        return path

    @classmethod
    def load(cls, filepath=None, *, cache=None, params=None, allow_checkpoint=False):
        if allow_checkpoint and filepath is None:
            raise ValueError('Checkpoint loading requires an explicit filepath')
        if filepath is None and (cache is None or params is None):
            raise ValueError('Automatic load requires cache and complete matching params')
        if not isinstance(cache, NMFCache):
            from pathlib import Path
            cache = NMFCache(cache if cache is not None else Path(filepath).parent)
        training = params.get('training', {}) if params is not None else {}
        if filepath is None:
            solver = training.get('solver', params.get('solver', 'cd'))
            name = ('nmf_F' if solver == 'minibatch' else 'nmf_F_full') if cls.role == 'stellar' else ('nmf_G_batch' if solver == 'minibatch' else 'nmf_G')
        else:
            name = None
        artifact = cache.load(
            filepath,
            name=name,
            params=params,
            role=cls.role,
            allow_checkpoint=allow_checkpoint
        )
        saved = artifact.metadata['params']
        config = dict(saved.get('training', {}))
        if config:
            model = cls(**config)
        else:
            # Legacy sidecars have flat solver settings; preserve every setting they recorded.
            allowed = {
                'max_iter',
                'tol',
                'init',
                'seed',
                'shuffle',
                'max_epochs',
                'batch_size',
                'tol_batches',
                'check_interval',
                'sparse_tol',
                'sparse_floor'
            }
            config = {key: value for key, value in saved.items() if key in allowed}
            config['solver'] = 'minibatch' if saved.get('solver') == 'minibatch' else 'cd'
            model = cls(n_components=artifact.basis.shape[0], **config)
        model.components_, model.wavelengths = artifact.basis, artifact.wavelengths
        model.params, model.preprocessing = saved, artifact.metadata['preprocessing']
        model.source_manifest, model.diagnostics = artifact.metadata['source_manifest'], artifact.metadata['diagnostics']
        model.cache_path, model.metadata = artifact.path, artifact.metadata
        return model


class SpectralNMF(_AbsorptionNMF):
    role = 'stellar'
    select = staticmethod(select_bosz)

    def __init__(self, n_components=16, **kwargs):
        super().__init__(n_components, **kwargs)


class TelluricNMF(_AbsorptionNMF):
    role = 'telluric'

    def __init__(self, n_components=4, **kwargs):
        defaults = dict(tol=1e-4, init=None, shuffle=False)
        defaults.update(kwargs)
        super().__init__(n_components, **defaults)
