from pathlib import Path
import gzip
import json
import multiprocessing
import os
import re
import sqlite3
import time
import weakref
import atexit
from collections import OrderedDict
from dataclasses import dataclass, fields

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm, trange


@dataclass(frozen=True) # frozen so the ID can be hashed
class SpectrumID:
    atmos: str # model atmosphere
    teff: str # effective temperature
    logg: str # surface gravity
    metal: str # solar metallicity
    alpha: str # alpha element enrichment
    carbon: str # carbon abundance
    micro: str # microturbulence velocity
    insbroad: str # instrumental broadening
    prod: str # "type of product", whatever that means

_WORKER_SPECTRA = None # set before forking so workers can reach the instance

def _compute_item(id):
    return id, _WORKER_SPECTRA._compute_spectrum(id)

class BOSZSpectra:
    # Bump when the processing code changes (clip, NaN handling, ...) so that
    # caches made by older code are rejected instead of silently reused.
    PROCESSING_VERSION = 1
    MAX_OPEN_MAPS = 64 # LRU of memory-mapped group files

    def __init__(
            self, 
            root: Path, 
            R_in, 
            R_out, 
            D, # Output resolution
            wave_min=None, 
            wave_max=None,
            cache_dir=None,
            spectra_per_file=1000 # Number of spectra stored in each cached .npy file
        ):
        self.root = root
        self.use_cache = cache_dir is not None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.spectra_per_file = spectra_per_file

        self.ids = []
        self.file_to_id = dict()
        self.id_to_file = dict()

        # For convolution
        self.R_in = R_in
        self.R_out = R_out

        # For cropping
        self.wave_min = wave_min
        self.wave_max = wave_max

        self.scan_root()

        # If min or max not provided, use first spectrum to infer
        if self.wave_min is None or self.wave_max is None:
            spec_0 = self.raw_spectrum(self.ids[0])
            wave = spec_0[:, 0]
            min, max = (np.min(wave), np.max(wave))

            if self.wave_min is None:
                self.wave_min = min
            if self.wave_max is None:
                self.wave_max = max

        # Output downsampling wavelengths
        self.D = D
        self.wave_out = np.geomspace(self.wave_min, self.wave_max, self.D)

        # Cache: spectra waiting to be written to the next group file
        self._pending = {} # SpectrumID -> processed spectrum
        self._maps = OrderedDict() # cache_file -> np.memmap
        if self.use_cache:
            self._open_cache()
            # Don't lose spectra computed ad hoc (e.g. via spectra[i]) at exit
            atexit.register(BOSZSpectra._flush_at_exit, weakref.ref(self))

    # ---- Cache (grouped .npy files + sqlite index) ----

    _ID_COLS = [f.name for f in fields(SpectrumID)]

    def _processing_params(self):
        return {
            "R_in": float(self.R_in),
            "R_out": float(self.R_out),
            "D": int(self.D),
            "wave_min": float(self.wave_min),
            "wave_max": float(self.wave_max),
            "processing_version": self.PROCESSING_VERSION,
        }

    def _open_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Remove temp files left by a crash (they count against the file quota).
        # Only old ones, in case another process is mid-write.
        for tmp in self.cache_dir.glob("processed_*.tmp"):
            try:
                if time.time() - tmp.stat().st_mtime > 3600:
                    tmp.unlink()
            except FileNotFoundError:
                pass

        self.index_path = self.cache_dir / "spectra_index.sqlite"
        self.db = sqlite3.connect(self.index_path, timeout=60)
        id_defs = ", ".join(f"{c} TEXT NOT NULL" for c in self._ID_COLS)
        id_list = ", ".join(self._ID_COLS)
        with self.db:
            self.db.execute(
                f"CREATE TABLE IF NOT EXISTS spectra ("
                f"source_file TEXT NOT NULL, {id_defs}, "
                f"cache_file TEXT NOT NULL, row_index INTEGER NOT NULL, "
                f"UNIQUE ({id_list}))"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS params "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

        # Check processing parameters against an existing index
        current = self._processing_params()
        stored = {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM params")}
        if stored:
            diff = {k: (stored.get(k), v) for k, v in current.items() if stored.get(k) != v}
            if diff:
                raise ValueError(
                    "Processing parameters differ from those in the existing cache "
                    f"{self.index_path} (cached, requested): {diff}. "
                    "Use a different cache_dir or delete the cache."
                )
        else:
            with self.db:
                self.db.executemany(
                    "INSERT INTO params (key, value) VALUES (?, ?)",
                    [(k, json.dumps(v)) for k, v in current.items()],
                )

        self._next_index = self._find_next_index()

    def _find_next_index(self):
        """One past the highest group file index in the DB or on disk."""
        pattern = re.compile(r"processed_(\d+)\.npy$")
        used = [int(m.group(1)) for (name,) in self.db.execute("SELECT DISTINCT cache_file FROM spectra")
                if (m := pattern.search(name))]
        used += [int(m.group(1)) for f in self.cache_dir.glob("processed_*.npy")
                 if (m := pattern.search(f.name))]
        return max(used, default=-1) + 1

    @staticmethod
    def _flush_at_exit(ref):
        self = ref()
        if self is not None:
            self.flush_cache()

    def _lookup(self, id):
        """Returns (cache_file, row_index) from the index, or None."""
        where = " AND ".join(f"{c} = ?" for c in self._ID_COLS)
        return self.db.execute(
            f"SELECT cache_file, row_index FROM spectra WHERE {where}",
            [getattr(id, c) for c in self._ID_COLS],
        ).fetchone()

    def flush_cache(self):
        """Writes pending spectra to the next group file and records them in the index."""
        if not self.use_cache or not self._pending:
            return

        ids = list(self._pending)
        data = np.stack([self._pending[id] for id in ids])
        tmp = self.cache_dir / f"processed_{os.getpid()}_{self._next_index:09d}.tmp"

        # Write to a temp name, then claim the final name, then insert the DB rows
        with open(tmp, "wb") as f:
            np.save(f, data)
            f.flush()
            os.fsync(f.fileno())

        # os.link fails if the name exists, so we never overwrite a group file
        # (e.g. one written by another process using the same cache_dir)
        self._next_index = max(self._next_index, self._find_next_index())
        while True:
            name = f"processed_{self._next_index:09d}.npy"
            try:
                os.link(tmp, self.cache_dir / name)
                break
            except FileExistsError:
                self._next_index += 1
        os.unlink(tmp)

        id_list = ", ".join(self._ID_COLS)
        placeholders = ", ".join("?" * (len(self._ID_COLS) + 3))
        rows = [
            (self.id_to_file[id].name, *(getattr(id, c) for c in self._ID_COLS), name, i)
            for i, id in enumerate(ids)
        ]
        with self.db:
            self.db.executemany(
                f"INSERT OR REPLACE INTO spectra (source_file, {id_list}, cache_file, row_index) "
                f"VALUES ({placeholders})",
                rows,
            )

        self._next_index += 1
        self._pending.clear()

    def _read_cached(self, id):
        if id in self._pending:
            return self._pending[id].copy()
        cache_file, row = self._lookup(id)
        data = self._maps.get(cache_file)
        if data is None:
            try:
                data = np.load(self.cache_dir / cache_file, mmap_mode="r")
            except FileNotFoundError:
                return None # index points at a missing file; caller recomputes
            self._maps[cache_file] = data
            if len(self._maps) > self.MAX_OPEN_MAPS:
                self._maps.popitem(last=False)
        else:
            self._maps.move_to_end(cache_file)
        return np.array(data[row])

    def scan_root(self):
        pattern = "**/bosz2024_*.txt.gz"
        files = list(self.root.glob(pattern))
        for file in tqdm(files, unit="file", desc="Parsing root directory"):
            id = self.id_from_path(file)
            if id in self.id_to_file: # dict lookup; `in self.ids` (a list) is O(n^2)
                raise ValueError(f"ID {id} was found twice in the dataset")
            
            self.file_to_id[file] = id
            self.id_to_file[id] = file
            self.ids.append(id)

    def id_from_path(self, file: Path):
        name = file.name.removesuffix(".txt.gz")
        # bosz2024_<atmos>_<teff>_<logg>_<metal>_<alpha>_<carbon>_<micro>_<insbroad>_<prod>
        _, atmos, teff, logg, metal, alpha, carbon, micro, insbroad, prod = name.split("_")
        return SpectrumID(
            atmos=atmos,
            teff=teff,
            logg=logg,
            metal=metal,
            alpha=alpha,
            carbon=carbon,
            micro=micro,
            insbroad=insbroad,
            prod=prod,
        )

    def raw_spectrum(self, id, is_path=False):
        """
        Loads a spectrum from its ID (or path). 
        Columns:
            - Wavelength (angstroms)
            - Eddington first moment (H, surface brightness)
            - Continuum
        """
        if is_path:
            id = self.path_to_id[id] # Convert to SpectrumID

        array = np.loadtxt(self.id_to_file[id], dtype=np.float32)
        return array

    def _cached_ids(self):
        """All cached ids (indexed or pending) from a single query."""
        id_list = ", ".join(self._ID_COLS)
        cached = {SpectrumID(*row) for row in self.db.execute(f"SELECT {id_list} FROM spectra")}
        return cached | set(self._pending)

    def is_cached(self, id):
        if not self.use_cache:
            return False

        return id in self._pending or self._lookup(id) is not None

    def process_all_spectra(self, force_recompute=False, n_workers=None):
        """
        Computes and caches every spectrum. Workers only compute; this process
        alone writes the group files and the sqlite index.
        `n_workers` defaults to SLURM_CPUS_PER_TASK, else the usable core count.
        """
        if not self.use_cache:
            raise Exception("Caching is not enabled, so this function is a no-op")

        if n_workers is None:
            n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))

        if force_recompute:
            todo = list(self.ids)
        else:
            cached = self._cached_ids() # one query; per-id is_cached is very slow at this scale
            todo = [id for id in self.ids if id not in cached]

        # Counts finished spectra out of all of them, including ones already cached
        pbar = tqdm(
            total=len(self.ids), initial=len(self.ids) - len(todo),
            unit='spectra', desc="Finished spectra", mininterval=10,
        )

        if n_workers <= 1:
            for id in todo:
                self.process_spectrum(id, force_recompute)
                pbar.update(1)
        else:
            # fork: workers inherit self via a global (self holds the sqlite
            # connection and can't be pickled); they never touch the db
            global _WORKER_SPECTRA
            _WORKER_SPECTRA = self
            with multiprocessing.get_context("fork").Pool(n_workers) as pool:
                for id, spectrum in pool.imap_unordered(_compute_item, todo, chunksize=4):
                    self._add_pending(id, spectrum)
                    pbar.update(1)

        pbar.close()
        self.flush_cache()

    @staticmethod
    def convolve_to_R(spectrum: np.ndarray, R_in=300_000, R_out=115_000):
        """
        Convolves a spectrum (shape (N, 2)) to have a new resolving power
        Generated by GPT-6 Luna with small modifications
        """
        wave, flux = spectrum.T

        # Sort by wavelength
        order = np.argsort(wave)
        wave, flux = wave[order], flux[order]
        log_wave = np.log(wave)

        # Resample onto a uniform log-wavelength grid
        dlog = np.median(np.diff(log_wave))
        log_uniform = np.arange(log_wave[0], log_wave[-1], dlog)
        flux_uniform = np.interp(log_uniform, log_wave, flux)

        # Gaussian sigma in log-wavelength units.
        # Resolving power R corresponds approximately to FWHM(log wavelength) = 1/R.
        if R_out >= R_in:
            raise ValueError("This only smooths to a lower resolving power.")
        sigma_log = np.sqrt(R_out**-2 - R_in**-2) / np.sqrt(8 * np.log(2))
        sigma_pixels = sigma_log / dlog

        smoothed_uniform = gaussian_filter1d(
            flux_uniform, sigma_pixels, mode="nearest"
        )

        # Return the smoothed flux on the original wavelength samples
        smoothed = np.interp(log_wave, log_uniform, smoothed_uniform)
        return smoothed

    def process_spectrum(self, id, force_recompute=False):
        """
        Loads and convolves a spectrum. Downsamples so that spectrum has `D` values.
        """
        if not force_recompute and self.is_cached(id):
            spectrum = self._read_cached(id)
            if spectrum is not None:
                return spectrum

        spectrum = self._compute_spectrum(id)

        if self.use_cache:
            self._add_pending(id, spectrum)

        return spectrum # corresponding wavelengths are in self.wave_out

    def _add_pending(self, id, spectrum):
        # Written to disk once a full group has accumulated (or on flush_cache)
        self._pending[id] = spectrum
        if len(self._pending) >= self.spectra_per_file:
            self.flush_cache()

    def _compute_spectrum(self, id):
        """Pure computation with no cache access, so it is safe in worker processes."""
        raw_spectrum = self.raw_spectrum(id)
        wave = raw_spectrum[:, 0]

        # normalized flux; NaNs allowed
        with np.errstate(divide="ignore", invalid="ignore"):
            spectrum = raw_spectrum[:, 1] / raw_spectrum[:, 2] 

        # Convolve spectrum to lower resolving power (R)
        spectrum = BOSZSpectra.convolve_to_R(
            np.column_stack((wave, spectrum)), 
            self.R_in, 
            self.R_out
        )
        # Mask to only desired wavelength range
        mask = (wave >= self.wave_min) & (wave <= self.wave_max)
        spectrum = spectrum[mask]

        # Downsample to desired resolution
        spectrum = np.interp(self.wave_out, wave[mask], spectrum)

        # Clip to [0, 1]
        spectrum = np.clip(spectrum, 0, 1)

        return spectrum

    def __getitem__(self, i):
        return self.process_spectrum(self.ids[i])

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    import argparse

    # Defaults match the settings used in test_processing.py
    ap = argparse.ArgumentParser(description="Process and cache all BOSZ spectra")
    ap.add_argument("--root", type=Path, default=Path("/home/mward19/nobackup/autodelete/bosz-spectra/data"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/home/mward19/nobackup/autodelete/bosz-spectra/processed"))
    ap.add_argument("--R-in", type=float, default=300_000)
    ap.add_argument("--R-out", type=float, default=115_000)
    ap.add_argument("--D", type=int, default=3000)
    ap.add_argument("--wave-min", type=float, default=3000)
    ap.add_argument("--wave-max", type=float, default=7000)
    ap.add_argument("--spectra-per-file", type=int, default=1000)
    ap.add_argument("--n-workers", type=int, default=None, help="default: SLURM_CPUS_PER_TASK")
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    spectra = BOSZSpectra(
        args.root,
        args.R_in,
        args.R_out,
        D=args.D,
        wave_min=args.wave_min,
        wave_max=args.wave_max,
        cache_dir=args.cache_dir,
        spectra_per_file=args.spectra_per_file,
    )
    print(f"{len(spectra)} spectra found; caching to {args.cache_dir}", flush=True)
    spectra.process_all_spectra(force_recompute=args.force_recompute, n_workers=args.n_workers)
    print("Done", flush=True)
