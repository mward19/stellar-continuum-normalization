from pathlib import Path
import json
import os
import re

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from load_spectra import BOSZSpectra


class TelluricSpectra:
    """
    Telluric spectra reconstructed from the PCA-compressed PypeIt grid
    (LBLRTM; Prochaska et al. 2025), as in Casey et al. (2026, ApJ 998:192).

    The PypeIt file stores the PCA coefficients of the training grid (5000 spectra) and
    the PCA components. Each spectrum is rebuilt as
        z   = mean + coefs[:L] @ components[:L]
        tau = max(sinh(z), 0)     # the file stores arcsinh(optical depth)
        T   = exp(-tau)
    Then T is optionally convolved to `R_out` and interpolated onto `wave_out`,
    which is the same grid BOSZSpectra uses. `self.M` is -log(T), the nonnegative
    matrix that gets factorized by NMF.
    """

    # Bump when the processing changes so caches from older code are rejected
    PROCESSING_VERSION = 1
    CACHE_NAME = "telluric_neglogT"

    def __init__(
            self,
            pca_file: Path,
            D, # Output resolution; must match BOSZSpectra
            wave_min,
            wave_max,
            R_out=None, # None: keep the file's native resolution
            n_pca=4, # number of PCA components used for the reconstruction
            t_floor=1e-10, # transmission floor so that -log(T) stays finite
            cache_dir=None,
        ):
        self.pca_file = Path(pca_file)
        self.D = D
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.R_out = R_out
        self.n_pca = n_pca
        self.t_floor = t_floor
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        match = re.search(r"_R(\d+)\.fits$", self.pca_file.name)
        if match is None:
            raise ValueError(f"Can't read the resolution from the file name {self.pca_file.name}")
        self.R_in = int(match.group(1))
        if self.R_out is not None and self.R_out >= self.R_in:
            raise ValueError(f"R_out ({self.R_out}) must be below the file's resolution ({self.R_in})")

        # Output wavelengths, identical to BOSZSpectra.wave_out
        self.wave_out = np.geomspace(self.wave_min, self.wave_max, self.D)

        self.M = self._load_cache() if self.cache_dir is not None else None
        if self.M is None:
            self.M = self._compute()
            if self.cache_dir is not None:
                self._save_cache()

    # ---- Cache: one .npy plus a json of the parameters ----

    def _params(self):
        return {
            "pca_file": self.pca_file.name,
            "R_in": self.R_in,
            "R_out": None if self.R_out is None else float(self.R_out),
            "n_pca": int(self.n_pca),
            "D": int(self.D),
            "wave_min": float(self.wave_min),
            "wave_max": float(self.wave_max),
            "t_floor": float(self.t_floor),
            "processing_version": self.PROCESSING_VERSION,
        }

    def _cache_paths(self):
        return (self.cache_dir / f"{self.CACHE_NAME}.npy",
                self.cache_dir / f"{self.CACHE_NAME}.json")

    def _load_cache(self):
        npy_path, json_path = self._cache_paths()
        if not json_path.is_file():
            return None

        stored = json.loads(json_path.read_text())
        current = self._params()
        diff = {k: (stored.get(k), v) for k, v in current.items() if stored.get(k) != v}
        if diff:
            raise ValueError(
                f"Processing parameters differ from those in the existing cache {json_path} "
                f"(cached, requested): {diff}. Use a different cache_dir or delete the cache."
            )
        if not npy_path.is_file():
            return None # json is only written after the npy, so this means it was deleted
        return np.load(npy_path)

    def _save_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        npy_path, json_path = self._cache_paths()

        # Write the data to a temp name and rename it; the json goes last
        # so that an existing json always means a complete npy
        tmp_npy = npy_path.with_name(npy_path.name + f".{os.getpid()}.tmp")
        with open(tmp_npy, "wb") as f:
            np.save(f, self.M)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_npy, npy_path)

        tmp_json = json_path.with_name(json_path.name + f".{os.getpid()}.tmp")
        tmp_json.write_text(json.dumps(self._params(), indent=2))
        os.replace(tmp_json, json_path)

    # ---- Reconstruction ----

    def _read_pca_file(self):
        with fits.open(self.pca_file) as hdul:
            ncomp = hdul[0].header.get("NCOMP")
            if ncomp is None:
                raise ValueError(f"{self.pca_file} has no NCOMP keyword, so it isn't a PCA file")
            comp = np.asarray(hdul[0].data, dtype=np.float64) # (ncomp + 1, n_wave); row 0 is the mean
            wave = np.asarray(hdul[1].data, dtype=np.float64)
            coefs = np.asarray(hdul[3].data, dtype=np.float64) # (n_spectra, ncomp)
        if self.n_pca > ncomp:
            raise ValueError(f"n_pca ({self.n_pca}) is more than the {ncomp} components in the file")
        return comp, wave, coefs

    def _compute(self):
        comp, wave, coefs = self._read_pca_file()

        # Crop with a pad so the convolution doesn't see the edge
        pad = 0.01
        sel = (wave >= self.wave_min * (1 - pad)) & (wave <= self.wave_max * (1 + pad))
        if wave[sel].min() > self.wave_min or wave[sel].max() < self.wave_max:
            raise ValueError(
                f"Requested range {self.wave_min}-{self.wave_max} is outside the file's "
                f"{wave.min():.0f}-{wave.max():.0f} Angstrom"
            )
        wave = wave[sel]
        mean = comp[0, sel]
        basis = comp[1:self.n_pca + 1, sel]

        M = np.empty((len(coefs), self.D), dtype=np.float32)
        for i in tqdm(range(len(coefs)), unit="spectrum", desc="Reconstructing telluric spectra", mininterval=10):
            tau = np.clip(np.sinh(mean + coefs[i, :self.n_pca] @ basis), 0, None)
            T = np.exp(-tau)
            if self.R_out is not None:
                T = BOSZSpectra.convolve_to_R(np.column_stack((wave, T)), self.R_in, self.R_out)
            T = np.interp(self.wave_out, wave, T)
            M[i] = -np.log(np.clip(T, self.t_floor, 1.0))
        return M

    # ---- Access ----

    def transmission(self, i):
        return np.exp(-self.M[i])

    def __getitem__(self, i):
        return self.M[i] # -log(T); corresponding wavelengths are in self.wave_out

    def __len__(self):
        return len(self.M)


if __name__ == "__main__":
    import argparse

    # Defaults match the stellar settings in load_spectra.py
    ap = argparse.ArgumentParser(description="Reconstruct and cache the telluric -log(T) matrix")
    ap.add_argument("--pca-file", type=Path, default=Path("/home/mward19/nobackup/autodelete/telluric/TellPCA_3000_10500_R120000.fits"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/home/mward19/nobackup/autodelete/telluric/processed"))
    ap.add_argument("--R-out", type=float, default=115_000)
    ap.add_argument("--D", type=int, default=3000)
    ap.add_argument("--wave-min", type=float, default=3000)
    ap.add_argument("--wave-max", type=float, default=7000)
    ap.add_argument("--n-pca", type=int, default=4)
    ap.add_argument("--t-floor", type=float, default=1e-10)
    args = ap.parse_args()

    tell = TelluricSpectra(
        args.pca_file, args.D, args.wave_min, args.wave_max,
        R_out=args.R_out, n_pca=args.n_pca, t_floor=args.t_floor, cache_dir=args.cache_dir,
    )
    print(f"{len(tell)} telluric spectra, -log(T) in [{tell.M.min():.3g}, {tell.M.max():.3g}], "
          f"cached to {args.cache_dir}", flush=True)
