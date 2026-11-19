"""Resolution-based sampling for constant-log-wavelength grids."""
import math

import numpy as np


def log_wavelength_grid(wave_min, wave_max, *, D=None, resolving_power=None,
                        samples_per_fwhm=3.0):
    """Use explicit D, or sample the target resolution at three points per FWHM.

    At high resolving power R, an instrumental FWHM spans about 1/R in log
    wavelength. This chooses sampling density, rather than changing resolution.
    Explicit D takes precedence over resolving_power and samples_per_fwhm.
    """
    if wave_min is None or wave_max is None:
        raise ValueError('wave_min and wave_max must be provided')
    wave_min, wave_max = float(wave_min), float(wave_max)
    if not np.isfinite([wave_min, wave_max]).all() or not 0 < wave_min < wave_max:
        raise ValueError('Wavelength bounds must be finite, positive, and increasing')
    if D is None:
        if resolving_power is None or not np.isfinite(resolving_power) or resolving_power <= 0:
            raise ValueError('Automatic D requires a finite, positive resolving power')
        if not np.isfinite(samples_per_fwhm) or samples_per_fwhm <= 0:
            raise ValueError('samples_per_fwhm must be finite and positive')
        intervals = resolving_power * samples_per_fwhm * math.log(wave_max / wave_min)
        D = max(2, math.ceil(intervals) + 1)
    elif not isinstance(D, (int, np.integer)) or isinstance(D, (bool, np.bool_)) or D < 2:
        raise ValueError('D must be an integer >= 2, or None for automatic sampling')
    return np.geomspace(wave_min, wave_max, int(D))
