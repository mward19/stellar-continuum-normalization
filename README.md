# stellar-continuum-normalization

Continuum normalization of echelle spectra with a linear model, after
[Casey et al. (2026, ApJ 998:192)](https://iopscience.iop.org/article/10.3847/1538-4357/ae3bd6/pdf).
For the log of an observed spectrum, `Y = log(flux)`, the model is

```
Y = -alpha @ F - beta @ G + H @ gamma
```

- `F` (K stellar basis vectors) is an NMF of `-log(flux)` for a grid of BOSZ synthetic spectra.
- `G` (L telluric basis vectors) is an NMF of `-log(T)` for the PypeIt telluric grid.
- `H` is a per-echelle-order Fourier basis for the continuum.
- `alpha >= 0` and `beta >= 0` are found by weighted least squares, with `gamma` unconstrained.

## Layout

```
src/
  load_spectra.py    BOSZSpectra: reads, convolves and resamples BOSZ spectra; caches them
  load_telluric.py   TelluricSpectra: reconstructs -log(T) from the PypeIt PCA file; caches it
  model.py           ContinuumNormalizer (NMF for F and G, design matrix H, fit);
                     read_harps_e2ds; a __main__ that fits one HARPS exposure
scripts/
  get_scripts_bosz.sh, download_all_bosz.py   download the BOSZ spectra
  download_telluric.sh                         download the telluric PCA grid
  load_spectra.sh, load_spectra_grace.sh       Slurm jobs that process and cache the BOSZ spectra
test_load_telluric.py       pytest tests for TelluricSpectra
test_processing_*.py        marimo notebooks for exploring the BOSZ and telluric NMFs
```

Everything is run from the repository root. Environments are managed with
[uv](https://docs.astral.sh/uv/) (`uv run ...`, Python >= 3.13).

The paths below (`~/nobackup/autodelete/...`) are the defaults baked into the scripts. Change them
in the scripts or with the command-line arguments if your data lives elsewhere.

## Data

### 1. Telluric grid (36 MiB)

```bash
bash scripts/download_telluric.sh      # saves to $TELLURIC_DIR, default ~/nobackup/autodelete/telluric
```

This is the PypeIt `TellPCA_3000_10500_R120000.fits` file. Then reconstruct and cache the
matrix (about a minute):

```bash
uv run src/load_telluric.py
```

### 2. BOSZ synthetic spectra (large)

The spectra come from the [BOSZ 2024 grid](https://archive.stsci.edu/hlsp/bosz) at STScI. The
download is hundreds of gigabytes and hundreds of thousands of files, so use scratch space.

`get_scripts_bosz.sh` and `download_all_bosz.py` use `./scripts` and `./data` relative to the
directory you run them from. Run them from your BOSZ directory, not the repo, so they don't write
into this repo's `scripts/`:

```bash
mkdir -p ~/nobackup/autodelete/bosz-spectra/scripts && cd ~/nobackup/autodelete/bosz-spectra
bash /path/to/repo/scripts/get_scripts_bosz.sh        # fetches STScI's per-metallicity bulk-download scripts
python /path/to/repo/scripts/download_all_bosz.py     # runs them in parallel into ./data
```

`download_all_bosz.py` imports `show_progress_curl_script`, which is not in this repo. It has to be
on the Python path (for example, next to the script).

Then process and cache the spectra. This convolves from R=300,000 to R=115,000 and resamples to
3000 pixels over 3000-7000 Angstrom, which takes hours. Run it as a Slurm job:

```bash
mkdir -p logs                        # sbatch needs logs/ to exist
sbatch scripts/load_spectra.sh       # or load_spectra_grace.sh
```

It is safe to resubmit if the job times out. Cached spectra are skipped. Spectra are stored in
`.npy` files of `--spectra-per-file` spectra each, indexed by `spectra_index.sqlite`, in the
cache directory. That limits the file count.

### 3. HARPS spectrum (to test the fit)

`src/model.py` has readers for ESO/HARPS exposures, either the `e2ds` per-order product or the merged `s1d` spectrum over a sub-range. The paper's example stars (Table 2: HD 45418,
HD 186543, HD 222595, HD 20794, HD 203050, HD 85512, HD 95735) and alpha Cen A are good choices.
Download the `*_e2ds_A.fits` (per-order, closest to the paper) or `*_s1d_A.fits` (merged 1D) pipeline products from either:

- the [ESO Science Archive](https://archive.eso.org) (HARPS instrument query; a free ESO account
  is usually needed), or
- [DACE](https://dace.unige.ch) at the University of Geneva.

`read_harps_e2ds` has not been tested against a real file, so check its header keywords first:
the wavelength polynomial `ESO DRS CAL TH COEFF LL*`, `ESO DRS CCD SIGDET`, `ESO DRS CCF RVC` and
`ESO DRS BERV`. Two things are unconfirmed: the sign convention of the radial velocity, and
whether HARPS wavelengths are in air (BOSZ is in vacuum). Check both before trusting a fit.

## Running

```bash
uv run src/model.py
```

By default `__main__` fits a synthetic spectrum: the mean of 10 random BOSZ spectra times a cubic
"continuum", with a flux uncertainty of 1e-8. `read_harps_e2ds` and `read_harps_s1d` (both untested)
are there for real HARPS data but `__main__` no longer calls them.

This loads the cached BOSZ and telluric spectra, runs the NMFs to get F and G, and fits the
exposure. The NMF results are cached as `nmf_F_<hash>.npy/.json` in the BOSZ cache directory and
`nmf_G_<hash>.npy/.json` in the telluric one. The first run does the full NMF over all the BOSZ
spectra, which is slow. The JSON beside each file records the parameters it was made with.

Tests and notebooks:

```bash
uv run --with pytest pytest test_load_telluric.py
uv run marimo edit test_processing_bosz.py
uv run marimo edit test_processing_telluric.py
```
