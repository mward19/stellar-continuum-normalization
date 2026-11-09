print("Importing packages")
try:
    from .load_spectra import BOSZSpectra
    from .load_telluric import TelluricSpectra
except ImportError:  # run as a script
    from load_spectra import BOSZSpectra
    from load_telluric import TelluricSpectra

import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from sklearn.decomposition import NMF, MiniBatchNMF
from tqdm import tqdm, trange
print("End importing")

_BUILD_STATE = None # (bosz_spectra, tol, floor), set before forking so workers can reach it


def _build_sparse_chunk(bounds):
    bosz_spectra, tol, floor = _BUILD_STATE
    return bosz_spectra.sparse(slice(*bounds), tol=tol, floor=floor).astype(np.float32)


C_KMS = 299792.458 # speed of light [km/s]
SPARSE_VERSION = 1 # bump when the sparse -log(flux) matrix is built differently
NMF_VERSION = 1 # bump when the NMF procedure changes in a way the params below don't capture


class ContinuumNormalizer:
    def __init__(
            self, 
            bosz_spectra: BOSZSpectra, 
            telluric_spectra: TelluricSpectra,
            flux_errors,
            order_wavelengths,
            debug_max_files=None,
            K=16, # number of stellar absorbtion basis vectors
            L=4,  # number of telluric absorbtion basis vectors
            n_modes=9, # Fourier continuum modes per row of order_wavelengths
            F_kwargs=None, # passed to get_F, e.g. to match an existing cache
            G_kwargs=None, # passed to get_G
            debug_max_F_spectra=None,
        ):
        self.bosz_spectra = bosz_spectra         # Matrix S
        self.telluric_spectra = telluric_spectra # Matrix M

        self.bosz_wave_out = bosz_spectra.wave_out # wavelengths F lives on
        self.telluric_wave_out = telluric_spectra.wave_out # wavelengths G lives on

        self.K = K
        self.L = L

        self.flux_errors = flux_errors
        self.order_wavelengths = order_wavelengths

        if debug_max_F_spectra is not None and debug_max_F_spectra < 1:
            raise ValueError("debug_max_F_spectra must be a positive integer or None")
        if debug_max_F_spectra is None:
            debug_max_F_spectra = getattr(bosz_spectra, "debug_max_F_spectra", None)
        self.debug_max_F_spectra = None if debug_max_F_spectra is None else int(debug_max_F_spectra)

        self.S_log = None # sparse -log(flux) matrix, built on demand (see get_sparse_S_log)
        self._S_log_key = None # (tol, floor) that S_log was built with
        self.train_rows = None # rows of S_log to train on; None means all (see filter_spectra)
        self.train_filter = None

        self.get_F(**(F_kwargs or {}))
        self.get_G(**(G_kwargs or {}))
        self.get_H(n_modes)

    def filter_spectra(self, atmos=("mp", "ms"), teff_max=8000):
        """
        Optional: restricts the spectra that get_F and fit_full train on. Call it before them.
        The defaults follow Casey et al. (2026): MARCS atmospheres only (BOSZ's "mp" and "ms";
        "ap" is ATLAS), and Teff <= 8000 K. Pass atmos=None or teff_max=None to skip either.
        """
        keep = []
        for row, id in enumerate(self.bosz_spectra.ids[:self._n_F_spectra()]):
            if atmos is not None and id.atmos not in atmos:
                continue
            if teff_max is not None and int(id.teff.lstrip("t")) > teff_max:
                continue
            keep.append(row)

        self.train_rows = np.array(keep, dtype=np.int64)
        self.train_filter = {
            "atmos": None if atmos is None else sorted(atmos),
            "teff_max": None if teff_max is None else float(teff_max),
        }
        tqdm.write(f"Training on {len(keep)} of {self._n_F_spectra()} selected spectra ({self.train_filter})")

    def _report_residuals(self, nmf, rows, n_samples, seed):
        """
        Reports how well S_log ~ W F on a random subset of `rows`, with W solved for the final F.
        This is computed directly in float64 on batches. (sklearn's reconstruction_err_ expands
        |X|^2 + |WH|^2 - 2<X,WH> in float32, which cancels catastrophically on a matrix this big
        and can come out as nan.)
        """
        rng = np.random.default_rng(seed)
        sample = np.sort(rng.choice(rows, size=min(n_samples, len(rows)), replace=False))
        squared_residual = squared_norm = 0.0
        for a in trange(0, len(sample), 1024, unit="batch", desc="Computing NMF residuals", leave=False):
            X = self.S_log[sample[a:a + 1024]]
            residual = X.toarray().astype(np.float64) - nmf.transform(X).astype(np.float64) @ nmf.components_.astype(np.float64)
            squared_residual += np.sum(residual ** 2)
            squared_norm += np.sum(X.data.astype(np.float64) ** 2)
        tqdm.write(
            f"NMF residuals on {len(sample)} spectra: relative Frobenius error "
            f"{np.sqrt(squared_residual / squared_norm):.3e}, "
            f"rms per pixel {np.sqrt(squared_residual / (len(sample) * self.S_log.shape[1])):.3e}"
        )

    def _train_rows(self):
        return np.arange(self._n_F_spectra()) if self.train_rows is None else self.train_rows

    def get_F(
            self,
            n_epochs=3,
            batch_size=256,
            seed=0,
            sparse_tol=None,
            sparse_floor=1e-10,
            residual_samples=20_000,
            batch=False,
            max_iter=10_000,
            tol=1e-4,
            init="nndsvda",
        ):
        """Builds F from BOSZ spectra using coordinate descent by default.

        Set ``batch=True`` to use MiniBatchNMF instead. In batch mode, ``n_epochs`` is the
        number of passes over the spectra, each in a fresh random batch order, and ``batch_size``
        sets the maximum spectra per batch.

        ``sparse_tol`` defaults to 1e-2 for coordinate descent and 1e-6 for batch mode. After
        fitting, residuals are reported on ``residual_samples`` random spectra (or all spectra if
        there are fewer).
        """
        if sparse_tol is None:
            sparse_tol = 1e-6 if batch else 1e-2

        if batch:
            cache_name = "nmf_F"
            params = {
                "K": int(self.K), "n_epochs": int(n_epochs), "batch_size": int(batch_size),
                "seed": int(seed), "solver": "minibatch",
                "n_spectra": self._n_F_spectra(),
                "bosz": self.bosz_spectra._processing_params(),
                "sparse_tol": float(sparse_tol), "sparse_floor": float(sparse_floor),
                "sparse_version": SPARSE_VERSION,
                "nmf_version": NMF_VERSION,
            }
        else:
            cache_name = "nmf_F_full"
            params = {
                "K": int(self.K), "max_iter": int(max_iter), "seed": int(seed),
                "tol": float(tol), "init": init, "solver": "cd",
                "n_spectra": self._n_F_spectra(),
                "bosz": self.bosz_spectra._processing_params(),
                "sparse_tol": float(sparse_tol), "sparse_floor": float(sparse_floor),
                "sparse_version": SPARSE_VERSION,
                "nmf_version": NMF_VERSION,
            }
        if self.train_filter is not None:
            params["filter"] = self.train_filter

        cache_dir = self.bosz_spectra.cache_dir
        cached = self._load_nmf(cache_dir, cache_name, params)
        if cached is not None:
            tqdm.write(f"Loaded cached F from {cache_dir}")
            self.F = cached
            return

        self._ensure_S_log(sparse_tol, sparse_floor)
        rows = self._train_rows()

        if batch:
            nmf_F = MiniBatchNMF(
                n_components=self.K,
                forget_factor=1,
                random_state=seed,
                verbose=False,
            )
            rng = np.random.default_rng(seed)
            for epoch in trange(n_epochs, unit="epoch", desc="Performing NMF on BOSZ spectra"):
                order = rng.permutation(rows)
                batches = [np.sort(order[a:a + batch_size]) for a in range(0, len(rows), batch_size)]
                for row_batch in tqdm(batches, desc=f"NMF epoch {epoch + 1}/{n_epochs}", leave=False):
                    nmf_F.partial_fit(self.S_log[row_batch])
            self._report_residuals(nmf_F, rows, residual_samples, seed)
        else:
            X = self.S_log if self.train_rows is None else self.S_log[self.train_rows]
            tqdm.write(f"Fitting NMF to {X.shape[0]} x {X.shape[1]} matrix, density {X.nnz / np.prod(X.shape):.2%}")
            nmf_F = NMF(
                n_components=self.K,
                init=init,
                solver="cd",
                max_iter=max_iter,
                tol=tol,
                random_state=seed,
                shuffle=True,
                verbose=1,
            )
            nmf_F.fit(X)
            tqdm.write(f"NMF finished after {nmf_F.n_iter_} iterations")
            self._report_residuals(nmf_F, rows, residual_samples, seed)

        self.F = nmf_F.components_
        self._save_nmf(cache_dir, cache_name, params, self.F)

    def fit_full(self, max_iter=10_000, seed=0, tol=1e-4, init="nndsvda", sparse_tol=1e-2, sparse_floor=1e-10, residual_samples=20_000):
        """Compatibility wrapper for ``get_F(batch=False, ...)``."""
        return self.get_F(
            batch=False,
            max_iter=max_iter,
            seed=seed,
            tol=tol,
            init=init,
            sparse_tol=sparse_tol,
            sparse_floor=sparse_floor,
            residual_samples=residual_samples,
        )

    def get_G(self, max_iter=10_000, seed=0):
        """
        Performs NMF on the dense -log(T) matrix self.telluric_spectra.M. This calculates the
        matrix G of telluric absorbtion basis vectors.
        """
        params = {
            "L": int(self.L), "max_iter": int(max_iter), "seed": int(seed), "solver": "cd",
            "telluric": self.telluric_spectra._params(),
            "nmf_version": NMF_VERSION,
        }
        cache_dir = self.telluric_spectra.cache_dir
        cached = self._load_nmf(cache_dir, "nmf_G", params)
        if cached is not None:
            tqdm.write(f"Loaded cached G from {cache_dir}")
            self.G = cached
            return

        tqdm.write("Performing NMF on telluric data")
        nmf_G = NMF(
            n_components=self.L,
            solver="cd",
            max_iter=max_iter,
            random_state=seed,
        )
        nmf_G.fit(self.telluric_spectra.M)

        tqdm.write("NMF on telluric data complete")
        self.G = nmf_G.components_
        self._save_nmf(cache_dir, "nmf_G", params, self.G)

    @staticmethod
    def _fourier_basis(order_wavelengths, n_modes):
        """
        Fourier design matrix for ONE echelle order (Casey et al. 2026, Eq. 9), shape (N, n_modes).

        With j = 1..n_modes, column j is cos((j-1)*pi*wave/P) for odd j (so j = 1 is all ones) and
        sin(j*pi*wave/P) for even j, where P is twice the wavelength range of the order.
        """
        wave = np.asarray(order_wavelengths, dtype=np.float64)
        P = 2 * (wave.max() - wave.min())

        basis = np.empty((len(wave), n_modes))
        for j in range(1, n_modes + 1): # j counts from 1, as in the paper
            if j % 2 == 1:
                basis[:, j - 1] = np.cos(np.pi * (j - 1) / P * wave) # j = 1 is the constant column
            else:
                basis[:, j - 1] = np.sin(np.pi * j / P * wave)
        return basis

    def get_H(self, n_modes=9):
        """
        Builds the continuum design matrix self.H, so that log(continuum) = H @ gamma.

        self.order_wavelengths has shape (n_orders, n_pixels). Each order has its own Fourier
        basis (with P set by that order's own wavelength range), and the bases are stacked
        block-diagonally: self.H is a sparse (n_orders * n_pixels, n_orders * n_modes) matrix
        whose rows follow order_wavelengths.ravel().
        """
        wave = np.atleast_2d(np.asarray(self.order_wavelengths, dtype=np.float64))
        self.n_modes = n_modes
        self.H = sparse.block_diag([self._fourier_basis(w, n_modes) for w in wave], format="csr")

    def fit(self, flux, rv=0.0):
        """
        Weighted least squares for the stellar (alpha), telluric (beta) and continuum (gamma)
        coefficients of one observed spectrum (Casey et al. 2026, Eqs. 10-13), with the
        constraints alpha >= 0 and beta >= 0.

        The model for Y = log(flux) is  Y = -alpha @ F - beta @ G + H @ gamma,  with the
        covariance of Y taken as diagonal, sigma_Y = flux_errors / flux.

        flux: array with the same shape as self.order_wavelengths (n_orders, n_pixels). The
            wavelengths (self.order_wavelengths) and errors (self.flux_errors) given to __init__
            are in the observer's frame.
        rv: stellar radial velocity in km/s (observer's frame). F is shifted to the observed
            frame; the data and G are not, since telluric lines don't move with the star.

        Returns alpha (K,), beta (L,) and gamma (n_orders, n_modes).
        """
        wave = np.asarray(self.order_wavelengths, dtype=np.float64).ravel()
        flux = np.asarray(flux, dtype=np.float64).ravel()
        flux_err = np.asarray(self.flux_errors, dtype=np.float64).ravel()

        # Drop pixels where the log or its variance is undefined
        good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0) & (flux_err > 0)
        wave, flux, flux_err = wave[good], flux[good], flux_err[good]

        Y = np.log(flux)
        weight = flux / flux_err # 1 / sigma_Y

        # Basis vectors on the observed wavelengths
        F = np.stack([np.interp(wave / (1 + rv / C_KMS), self.bosz_wave_out, f) for f in self.F], axis=1)
        G = np.stack([np.interp(wave, self.telluric_wave_out, g) for g in self.G], axis=1)
        H = self.H[good]

        # A weighted by 1/sigma turns the weighted problem into an ordinary bounded one
        A = sparse.hstack([sparse.csr_matrix(-F), sparse.csr_matrix(-G), H], format="csr")
        A = sparse.diags(weight) @ A

        K, L = F.shape[1], G.shape[1]
        lower = np.concatenate([np.zeros(K + L), np.full(H.shape[1], -np.inf)])
        upper = np.full(K + L + H.shape[1], np.inf)

        result = lsq_linear(A, weight * Y, bounds=(lower, upper), method="trf", lsq_solver="lsmr")
        X = result.x
        return X[:K], X[K:K + L], X[K + L:].reshape(-1, self.n_modes)

    # ---- Caching and data management ----

    @classmethod
    def from_cache(cls, F_path, G_path, flux_errors, order_wavelengths, n_modes=9):
        """
        Builds a normalizer from already-cached NMF results (the nmf_F_<hash>.npy and
        nmf_G_<hash>.npy files), without constructing BOSZSpectra or TelluricSpectra. The
        wavelength grids F and G live on are rebuilt from the parameters in the .json beside
        each file.
        """
        F_path, G_path = Path(F_path), Path(G_path)
        self = cls.__new__(cls) # skip __init__, which would compute or look up F and G
        self.S_log, self._S_log_key, self.train_rows, self.train_filter = None, None, None, None

        self.F = np.load(F_path)
        self.G = np.load(G_path)
        self.K, self.L = len(self.F), len(self.G)

        bosz = json.loads(F_path.with_suffix(".json").read_text())["bosz"]
        telluric = json.loads(G_path.with_suffix(".json").read_text())["telluric"]
        self.bosz_wave_out = np.geomspace(bosz["wave_min"], bosz["wave_max"], bosz["D"])
        self.telluric_wave_out = np.geomspace(telluric["wave_min"], telluric["wave_max"], telluric["D"])

        self.flux_errors = flux_errors
        self.order_wavelengths = order_wavelengths
        self.get_H(n_modes)
        return self

    # ---- Sparse matrix and NMF cache helpers ----

    @staticmethod
    def _nmf_cache_paths(cache_dir, name, params, suffix=".npy"):
        # Parameters are hashed into the file name so different settings can coexist in one
        # cache_dir; the json beside the data file records what they were.
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
        return cache_dir / f"{name}_{key}{suffix}", cache_dir / f"{name}_{key}.json"

    @classmethod
    def _load_nmf(cls, cache_dir, name, params):
        if cache_dir is None:
            return None
        npy_path, json_path = cls._nmf_cache_paths(cache_dir, name, params)
        if json_path.is_file() and npy_path.is_file(): # json is written last
            return np.load(npy_path)
        return None

    @classmethod
    def _save_nmf(cls, cache_dir, name, params, array):
        if cache_dir is None:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        npy_path, json_path = cls._nmf_cache_paths(cache_dir, name, params)

        # Same scheme as the loaders: write to a temp name, rename, json last
        tmp_npy = npy_path.with_name(npy_path.name + f".{os.getpid()}.tmp")
        with open(tmp_npy, "wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_npy, npy_path)

        tmp_json = json_path.with_name(json_path.name + f".{os.getpid()}.tmp")
        tmp_json.write_text(json.dumps(params, indent=2, sort_keys=True))
        os.replace(tmp_json, json_path)

    def get_sparse_S_log(self, tol=1e-2, floor=1e-10, chunk_size=1000, n_workers=None):
        """
        Builds the sparse -log(flux) matrix of the selected BOSZ spectra (float32 CSR, one row
        per spectrum), so NMF epochs can index rows instead of rereading and reprocessing every
        spectrum. It is cached as an .npz in the BOSZ cache_dir and held in memory in self.S_log.

        tol and floor are passed to BOSZSpectra.sparse. Multiprocess chunk building requires the
        selected spectra to be cached already (see BOSZSpectra.process_all_spectra).
        """
        n = self._n_F_spectra()
        params = {
            "tol": float(tol), "floor": float(floor),
            "n_spectra": n,
            "bosz": self.bosz_spectra._processing_params(),
            "sparse_version": SPARSE_VERSION,
        }
        cache_dir = self.bosz_spectra.cache_dir
        if cache_dir is not None:
            npz_path, json_path = self._nmf_cache_paths(cache_dir, "sparse_S_log", params, ".npz")
            if json_path.is_file() and npz_path.is_file(): # json is written last
                tqdm.write(f"Loading cached sparse matrix from {npz_path}")
                self.S_log = sparse.load_npz(npz_path).tocsr()
                self._S_log_key = (float(tol), float(floor))
                return

        bounds = [(a, min(a + chunk_size, n)) for a in range(0, n, chunk_size)]
        if n_workers is None:
            n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))

        global _BUILD_STATE
        _BUILD_STATE = (self.bosz_spectra, tol, floor)
        if n_workers <= 1:
            chunks = [_build_sparse_chunk(b) for b in tqdm(bounds, unit="chunk", desc="Building sparse -log(flux) matrix")]
        else:
            # Workers must not compute or touch the sqlite connection they inherit through fork, so
            # check that everything is cached and load the location index before forking
            needed_ids = set(self.bosz_spectra.ids[:n])
            if not needed_ids.issubset(self.bosz_spectra._cached_ids()):
                raise RuntimeError("Not every selected BOSZ spectrum is cached; run BOSZSpectra.process_all_spectra first")
            self.bosz_spectra._location_index()
            with multiprocessing.get_context("fork").Pool(n_workers) as pool:
                chunks = list(tqdm(pool.imap(_build_sparse_chunk, bounds), total=len(bounds),
                                   unit="chunk", desc=f"Building sparse -log(flux) matrix ({n_workers} workers)"))
        self.S_log = sparse.vstack(chunks, format="csr")
        self._S_log_key = (float(tol), float(floor))
        tqdm.write(f"Sparse matrix: shape {self.S_log.shape}, density {self.S_log.nnz / np.prod(self.S_log.shape):.2%}, "
                   f"{(self.S_log.data.nbytes + self.S_log.indices.nbytes + self.S_log.indptr.nbytes) / 1e9:.2f} GB")

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_npz = npz_path.with_name(npz_path.name + f".{os.getpid()}.tmp")
            with open(tmp_npz, "wb") as f:
                sparse.save_npz(f, self.S_log, compressed=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_npz, npz_path)

            tmp_json = json_path.with_name(json_path.name + f".{os.getpid()}.tmp")
            tmp_json.write_text(json.dumps(params, indent=2, sort_keys=True))
            os.replace(tmp_json, json_path)

    def _ensure_S_log(self, tol, floor):
        """Builds (or loads) S_log unless the one in memory was made with the same tol and floor."""
        if self.S_log is None or self._S_log_key != (float(tol), float(floor)):
            self.S_log = None # let the old matrix go before building another
            self.get_sparse_S_log(tol=tol, floor=floor)

    def _n_F_spectra(self):
        """Number of BOSZ rows selected for building and training F."""
        n = len(self.bosz_spectra)
        return n if self.debug_max_F_spectra is None else min(n, self.debug_max_F_spectra)

def read_harps_e2ds(path):
    """
    Read an ESO/HARPS e2ds file. Returns (wave, flux, flux_err, rv), where wave, flux and flux_err
    have shape (n_orders, n_pixels) = (72, 4096) and rv is the star's radial velocity [km/s] in
    the observer's frame.

    UNTESTED against a real file: the header keywords are from memory, so check them first.
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        header = hdul[0].header
        flux = np.asarray(hdul[0].data, dtype=np.float64)

    n_orders, n_pixels = flux.shape
    pixel = np.arange(n_pixels)

    # Wavelength solution: a cubic polynomial in pixel number per order, with the 4 coefficients
    # of order o in HIERARCH ESO DRS CAL TH COEFF LL{4o} .. LL{4o+3}
    wave = np.empty_like(flux)
    for o in range(n_orders):
        coeffs = [header[f"ESO DRS CAL TH COEFF LL{4 * o + k}"] for k in range(4)]
        wave[o] = np.polynomial.polynomial.polyval(pixel, coeffs)

    # e2ds has no error array; assume photon noise plus read noise, with flux in electrons
    read_noise = header.get("ESO DRS CCD SIGDET", 0.0)
    flux_err = np.sqrt(np.clip(flux, 0, None) + read_noise ** 2)

    # CCF RV is barycentric; the observed-frame velocity subtracts the barycentric Earth RV
    rv = header["ESO DRS CCF RVC"] - header["ESO DRS BERV"]
    return wave, flux, flux_err, rv


def read_harps_s1d(path, wave_min, wave_max, barycentric=False):
    """
    Read an ESO/HARPS s1d file (the pipeline's merged 1D spectrum on a uniform wavelength grid),
    keeping only wave_min <= wavelength <= wave_max [Angstrom]. Returns (wave, flux, flux_err, rv)
    with wave, flux and flux_err of shape (1, n_pixels), i.e. the whole range is treated as a
    single "order". rv is the star's radial velocity [km/s] in the observer's frame.

    UNTESTED against a real file: the header keywords are from memory, so check them first.
    I'm also unsure whether the s1d wavelength grid is in the barycentric frame. If it is, pass
    barycentric=True: the wavelengths are then moved back to the observer's frame, so that the
    telluric lines line up with G, using the same (unchecked) sign convention as read_harps_e2ds.
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        header = hdul[0].header
        if hdul[0].data is not None: # DRS image: wavelength grid is in the WCS keywords
            flux = np.asarray(hdul[0].data, dtype=np.float64)
            wave = header["CRVAL1"] + header["CDELT1"] * np.arange(len(flux))
            flux_err = None
        else: # table with WAVE, FLUX and ERR columns
            table = hdul[1].data
            wave = np.asarray(table["WAVE"], dtype=np.float64).ravel()
            flux = np.asarray(table["FLUX"], dtype=np.float64).ravel()
            flux_err = np.asarray(table["ERR"], dtype=np.float64).ravel() if "ERR" in table.names else None

    # No error array in the DRS image; assume photon noise with flux in electrons (approximate,
    # since s1d is resampled and its pixels are correlated)
    if flux_err is None:
        flux_err = np.sqrt(np.clip(flux, 0, None))

    berv = header["ESO DRS BERV"]
    rv = header["ESO DRS CCF RVC"] - berv
    if barycentric:
        wave = wave * (1 - berv / C_KMS)

    keep = (wave >= wave_min) & (wave <= wave_max)
    return wave[keep][None], flux[keep][None], flux_err[keep][None], rv
