"""Continuum-response bases and constrained fitting of individual observations.

Training and artifact persistence live in nmf.py and nmf_cache.py. Public classes
are re-exported here so callers can import the complete API from src.model.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from src.nmf import SpectralNMF, TelluricNMF
from src.nmf_cache import NMFCache, CacheMiss, CacheValidationError

C_KMS = 299792.458


def _orders(values):
    """Accept one order, rectangular arrays, or a sequence of unequal-length orders."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except ValueError:
        array = None
    if array is not None and array.ndim == 1:
        orders = [array]
    elif array is not None and array.ndim == 2:
        orders = list(array)
    elif array is None and isinstance(values, (list, tuple)):
        orders = [np.asarray(order, dtype=np.float64) for order in values]
    else:
        raise ValueError('Expected a 1-D spectrum or a sequence of 1-D echelle orders')
    if not orders or any(order.ndim != 1 or len(order) == 0 for order in orders):
        raise ValueError('Orders must be nonempty one-dimensional arrays')
    return orders


def _restore(flat, template, orders):
    try:
        shape = np.asarray(template).shape
    except ValueError:
        shape = None
    if shape is not None:
        return flat.reshape(shape)
    offsets = np.cumsum([0] + [len(order) for order in orders])
    return tuple(flat[a:b] for a, b in zip(offsets[:-1], offsets[1:]))


class ContinuumResponseBasis:
    """Per-order response design matrix with an injectable basis function.

    basis_function(wavelengths, n_modes) returns (pixels, n_modes), evaluated
    on each full order before any bad-pixel masking. The default is the paper's
    Fourier prescription, including its original absolute-wavelength phase.
    """

    def __init__(self, n_modes=9, *, basis_function=None):
        if not isinstance(n_modes, (int, np.integer)) or n_modes < 1:
            raise ValueError('n_modes must be a positive integer')
        self.n_modes = int(n_modes)
        self.basis_function = basis_function or self.fourier_basis

    @staticmethod
    def fourier_basis(wavelengths, n_modes):
        wave = np.asarray(wavelengths, dtype=np.float64)
        finite = wave[np.isfinite(wave)]
        if len(finite) < 2 or np.ptp(finite) <= 0:
            raise ValueError('Each order requires at least two distinct finite wavelengths')
        period = 2 * np.ptp(finite)
        basis = np.empty((len(wave), n_modes))
        # Preserve the original arithmetic order: tiny phase changes can affect ill-conditioned fits.
        for j in range(1, n_modes + 1):
            basis[:, j - 1] = np.cos(np.pi * (j - 1) / period * wave) if j % 2 else np.sin(np.pi * j / period * wave)
        return basis

    def design_matrix(self, order_wavelengths):
        blocks = []
        for wave in _orders(order_wavelengths):
            block = np.asarray(self.basis_function(wave, self.n_modes), dtype=np.float64)
            if block.shape != (len(wave), self.n_modes) or not np.isfinite(block[np.isfinite(wave)]).all():
                raise ValueError('Continuum basis returned an invalid matrix')
            blocks.append(sparse.csr_matrix(block))
        return sparse.block_diag(blocks, format='csr')


@dataclass
class ContinuumFitResult:
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    wavelengths: object
    continuum: object
    stellar_flux: object
    telluric_transmission: object
    model_flux: object
    rectified_flux: object
    mask: object
    solver_result: object

    @property
    def success(self):
        return self.solver_result.success

    @property
    def status(self):
        return self.solver_result.status

    @property
    def message(self):
        return self.solver_result.message

    def __iter__(self):
        # Retain tuple-unpacking convenience for callers interested only in coefficients.
        yield self.alpha
        yield self.beta
        yield self.gamma


class ContinuumNormalizer:
    """Assemble fitted absorption bases and solve bounded weighted least squares.

    No data loading, NMF fitting, or cache lookup occurs in this constructor.
    variance_order=1 preserves the previous weighting; 2 selects the paper's
    second-order approximation explicitly. No instrumental broadening is applied.
    """

    def __init__(self, stellar, telluric=None, response_basis=None, *, variance_order=1):
        if not isinstance(stellar, SpectralNMF):
            raise TypeError('stellar must be a fitted SpectralNMF; call fit() or load() first')
        stellar._require_fitted()
        if telluric is not None:
            if not isinstance(telluric, TelluricNMF):
                raise TypeError('telluric must be a fitted TelluricNMF or None')
            telluric._require_fitted()
            for key in ('wavelength_unit', 'wavelength_convention'):
                first, second = stellar.preprocessing.get(key), telluric.preprocessing.get(key)
                if first not in (None, 'unspecified') and second not in (None, 'unspecified') and first != second:
                    raise ValueError(f'Stellar and telluric {key} differ; convert explicitly before fitting')
        if variance_order not in (1, 2):
            raise ValueError('variance_order must be 1 or 2')
        self.stellar, self.telluric = stellar, telluric
        self.response_basis = response_basis if response_basis is not None else ContinuumResponseBasis()
        if not hasattr(self.response_basis, 'design_matrix') or not hasattr(self.response_basis, 'n_modes'):
            raise TypeError('response_basis must provide design_matrix() and n_modes')
        self.variance_order = variance_order

    @property
    def F(self):
        return self.stellar.components_

    @property
    def G(self):
        return None if self.telluric is None else self.telluric.components_

    @property
    def bosz_wave_out(self):
        return self.stellar.wavelengths

    @property
    def telluric_wave_out(self):
        return None if self.telluric is None else self.telluric.wavelengths

    def fit(
        self,
        wavelengths,
        flux=None,
        flux_errors=None,
        *,
        rv=0.,
        mask=None,
        tol=1e-10,
        lsq_solver='lsmr',
        max_iter=None
    ):
        """Fit one observation. True in mask means retain; outputs preserve input order shapes.

        Nonpositive/nonfinite flux/errors and wavelengths outside basis coverage are
        excluded. Output predictions are NaN at excluded pixels. RV is a fixed
        observed-frame stellar velocity in km/s; tellurics remain in the observer frame.
        Use lsq_solver="exact" for a dense solve on modest subsets; the default
        iterative solver keeps large observation design matrices sparse.
        """
        if flux is None and hasattr(self, '_legacy_observation'):
            flux = wavelengths
            wavelengths, flux_errors = self._legacy_observation
        if flux is None or flux_errors is None:
            raise ValueError('Provide wavelengths, flux, and flux_errors for this observation')
        if lsq_solver not in ('lsmr', 'exact'):
            raise ValueError("lsq_solver must be 'lsmr' or 'exact'")
        if not np.isfinite(rv) or 1 + rv / C_KMS <= 0:
            raise ValueError('Invalid observed-frame radial velocity')
        # Flatten orders for the solve; pack() restores their original shapes.
        wave_orders, flux_orders, error_orders = map(_orders, (wavelengths, flux, flux_errors))
        lengths = [len(order) for order in wave_orders]
        if [len(order) for order in flux_orders] != lengths or [len(order) for order in error_orders] != lengths:
            raise ValueError('Wavelengths, flux and errors must have matching order shapes')
        wave, observed, errors = map(np.concatenate, (wave_orders, flux_orders, error_orders))

        # Stellar absorption uses rest wavelengths; tellurics stay in the observer frame.
        rest_wave = wave / (1 + rv / C_KMS)
        good = np.isfinite(wave) & np.isfinite(observed) & np.isfinite(errors) & (observed > 0) & (errors > 0)
        good &= (rest_wave >= self.stellar.wavelengths[0]) & (rest_wave <= self.stellar.wavelengths[-1])
        if self.telluric is not None:
            good &= (wave >= self.telluric.wavelengths[0]) & (wave <= self.telluric.wavelengths[-1])
        if mask is not None:
            mask_orders = _orders(mask)
            if [len(order) for order in mask_orders] != lengths:
                raise ValueError('Mask must match the observed order shapes')
            flat_mask = np.concatenate(mask_orders)
            if not np.isin(flat_mask, [0, 1]).all():
                raise ValueError('Mask entries must be boolean')
            good &= flat_mask.astype(bool)
        if not good.any():
            raise ValueError('No usable pixels remain after masking and coverage checks')

        # Evaluate continuum modes on full orders before applying the pixel mask.
        H = self.response_basis.design_matrix(wavelengths)[good]
        F = self.stellar.evaluate(rest_wave[good]).T
        G = self.telluric.evaluate(wave[good]).T if self.telluric is not None else np.empty((good.sum(), 0))
        ratio = (errors[good] / observed[good])**2
        weight = 1 / np.sqrt(ratio + .5 * ratio**2) if self.variance_order == 2 else observed[good] / errors[good]
        A = sparse.hstack([sparse.csr_matrix(-F), sparse.csr_matrix(-G), H], format='csr')
        weighted_A = sparse.diags(weight) @ A
        K, L = F.shape[1], G.shape[1]

        # Absorption weights are nonnegative; continuum coefficients are unbounded.
        lower = np.r_[np.zeros(K + L), np.full(H.shape[1], -np.inf)]
        solve_matrix = weighted_A.toarray() if lsq_solver == 'exact' else weighted_A
        solved = lsq_linear(
            solve_matrix,
            weight * np.log(observed[good]),
            bounds=(lower, np.inf),
            method='trf',
            lsq_solver=lsq_solver,
            tol=tol,
            max_iter=max_iter
        )
        alpha, beta, gamma = solved.x[:K], solved.x[K:K + L], solved.x[K + L:]
        stellar_flux = np.exp(-F @ alpha)
        telluric_transmission = np.exp(-G @ beta)
        continuum = np.exp(H @ gamma)

        def pack(values):
            full = np.full(len(wave), np.nan)
            full[good] = values
            return _restore(full, wavelengths, wave_orders)

        return ContinuumFitResult(
            alpha=alpha,
            beta=beta,
            gamma=gamma.reshape(len(wave_orders), self.response_basis.n_modes),
            wavelengths=wavelengths,
            continuum=pack(continuum),
            stellar_flux=pack(stellar_flux),
            telluric_transmission=pack(telluric_transmission),
            model_flux=pack(np.exp(A @ solved.x)),
            rectified_flux=pack(observed[good] / continuum),
            mask=_restore(good, wavelengths, wave_orders),
            solver_result=solved
        )

    @classmethod
    def from_cache(cls, F_path, G_path=None, flux_errors=None, order_wavelengths=None, n_modes=9):
        """Compatibility adapter for explicit legacy or versioned F/G artifact paths."""
        model = cls(SpectralNMF.load(F_path), None if G_path is None else TelluricNMF.load(G_path),
                    ContinuumResponseBasis(n_modes))
        if flux_errors is not None or order_wavelengths is not None:
            if flux_errors is None or order_wavelengths is None:
                raise ValueError('Legacy observation defaults require wavelengths and errors together')
            warnings.warn(
                'Pass wavelengths and flux_errors to fit() instead of from_cache()',
                DeprecationWarning,
                stacklevel=2
            )
            model._legacy_observation = (order_wavelengths, flux_errors)
        return model


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
        if hdul[0].data is not None:  # DRS image: wavelength grid is in the WCS keywords
            flux = np.asarray(hdul[0].data, dtype=np.float64)
            wave = header["CRVAL1"] + header["CDELT1"] * np.arange(len(flux))
            flux_err = None
        else:  # table with WAVE, FLUX and ERR columns
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
