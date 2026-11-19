# Stellar continuum normalization

Fit stellar absorption, telluric transmission, and the continuum of an echelle spectrum,
following [Casey et al. (2026)](https://doi.org/10.3847/1538-4357/ae3bd6).
Train the absorption bases once, save them, and reuse them for observations.

## Setup

The project uses Python 3.13+ and [uv](https://docs.astral.sh/uv/).
From the repository root:

```bash
uv sync
```

This installs the library in editable mode in `.venv`. Use that Python environment for
scripts and notebook kernels; imports work from any directory, and source edits are
available immediately.

The examples below use ordinary, persistent `cache/` folders. Replace the example data and
model filenames with your own. If you need to train a model first, start with
[Train a stellar basis](#train-a-stellar-basis).

## Load a model and fit a spectrum

Load your saved stellar and telluric bases, read a HARPS exposure, and fit it:

```python
from src.model import (
    ContinuumNormalizer,
    SpectralNMF,
    TelluricNMF,
    read_harps_e2ds,
)

stellar = SpectralNMF.load("cache/stellar/nmf_F_full_15262808.npy")
telluric = TelluricNMF.load("cache/telluric/nmf_G_15262808.npy")
wavelengths, flux, flux_errors, rv = read_harps_e2ds("data/star_e2ds_A.fits")

normalizer = ContinuumNormalizer(stellar, telluric)
result = normalizer.fit(wavelengths, flux, flux_errors, rv=rv)
if not result.success:
    raise RuntimeError(result.message)

continuum = result.continuum
normalized_flux = result.rectified_flux
```

`normalized_flux` is the observed flux divided by the continuum; it still contains telluric
absorption. To omit telluric modeling, use `ContinuumNormalizer(stellar)`.
The normalizer uses nine Fourier modes per order by default.

If you already have wavelength, flux, and error arrays, pass them directly to `.fit()`;
reading a FITS file is a separate step. The HARPS readers remain experimental: verify their
header assumptions, wavelength convention, RV convention, and error estimates for your data.

### Inputs and outputs

Wavelengths, flux, and errors must have matching shapes. They can be a 1-D spectrum, a 2-D
array of orders, or a list of unequal-length orders. Errors are standard deviations in the
same units as flux. Wavelength units and conventions must match the trained bases; the BOSZ
workflow uses vacuum Angstroms.

`rv` is a fixed observed-frame stellar velocity in km/s. The fit shifts the stellar basis to
the observed frame; telluric wavelengths remain in the observer frame. It does not estimate RV.

An optional `mask` has the same shapes as the data, with `True` meaning retain the pixel.
The fit also excludes nonfinite values, nonpositive flux/errors, and wavelengths outside
basis coverage. Returned predictions contain NaN at excluded pixels.

| Result | Contents |
|---|---|
| `continuum` | Fitted continuum response |
| `rectified_flux` | Observed flux divided by continuum |
| `stellar_flux`, `telluric_transmission` | Fitted absorption components |
| `model_flux` | Combined model of the observed flux |
| `alpha`, `beta`, `gamma` | Stellar, telluric, and per-order continuum coefficients |
| `mask` | Pixels actually used |
| `success`, `status`, `message` | Solver outcome |
| `solver_result` | Full SciPy solver diagnostics |

## Train a stellar basis

Use a **small folder of raw BOSZ spectra** for your first training run. The example below
assumes `data/bosz-subset/` contains at least 16 MARCS spectra. For a representative selection
from the full grid, use [sandbox/nmf_subset.ipynb](sandbox/nmf_subset.ipynb).

```python
from pathlib import Path

from src.load_spectra import BOSZSpectra
from src.model import SpectralNMF
from src.preparation import select_bosz

spectra = BOSZSpectra(
    root=Path("data/bosz-subset"),
    R_in=300_000,
    R_out=115_000,
    D=None,  # Automatic sampling at three points per instrumental FWHM.
    wave_min=3800,
    wave_max=4200,
    cache_dir=Path("cache/processed-bosz"),
)
selection = select_bosz(spectra)  # MARCS atmospheres, Teff <= 8000 K.

stellar = SpectralNMF(
    n_components=16, max_iter=500, tol=1e-4, sparse_tol=0.0, matrix_format="dense",
)
stellar.fit(spectra, selection=selection, cache="cache/stellar")
print(stellar.cache_path)
print(stellar.diagnostics)
```

The loader normalizes and convolves the flux. The trainer prepares its selected optical
depths as a **dense float32 matrix** by default, filling one preallocated array in batches.
This avoids intermediate CSR storage and a second full-matrix concatenation. Set
`matrix_format="sparse"` on either trainer to use CSR computation instead, including for
explicit matrix inputs; both formats work with coordinate descent and minibatch training.
`sparse_tol` is a separate absorption cutoff: `sparse_tol=0.0` preserves weak absorption
in either format. Optional `matrix_cache` files keep dense `.npy` and sparse `.npz`
representations separate; dense caches load through a memory map. Existing processed-flux
caches remain reusable. `select_bosz()` returns row indices without changing the loader.
Without a selection, the trainer uses all supplied spectra, subject to the loader's
`debug_max_F_spectra` cap. Setting `debug_max_F_spectra=1000` on `BOSZSpectra` limits NMF
preparation and training to at most 1,000 rows. With `select_bosz()`, this cap applies after
filtering, so excluded ATLAS spectra do not consume the MARCS limit. Explicit selections
are capped in their supplied order as well. This is an ordered debugging limit; use the
subset notebook for representative sampling.

Preparation reports completed spectra as they are read, including within a batch of
uncached inputs. It runs serially by default. Pass `n_workers=8` to `.fit()` to process
missing selected spectra in parallel, then build the optical-depth matrix. Parallel
preparation requires a processed-spectrum `cache_dir`; the parent process alone writes
that cache, and subsequent runs reuse it. The worker count controls preparation, while
NMF numerical libraries use their own thread settings.

### Full HARPS training jobs

The production HARPS configuration is in [runs/harps.py](runs/harps.py). It selects all
396,108 BOSZ MARCS spectra with `mp` or `ms` atmospheres and `Teff <= 8000 K` (row stride 1).
The BOSZ grid is normalized by its theoretical continuum and convolved from `R_in=300000` to
`R_out=115000`. It covers 3000–7000 Å on a logarithmic grid with three samples per resolution
element (`D=292319`). Training uses 16 NMF components, coordinate descent, `max_iter=10000`,
`tol=1e-4`, and `sparse_tol=0.0` so weak absorption is retained.

Run preprocessing and NMF as separate Slurm jobs, in this order. Submit from the repository
root; create the log directory first:

```bash
mkdir -p logs
sbatch runs/preprocess_harps.sh
sbatch runs/nmf_harps.sh
```

Both jobs request the `m12` partition and 512G of RAM. Preprocessing uses 128 CPUs and has a
48-hour limit; it caches the processed spectra and builds the dense float32 optical-depth
matrix. NMF uses 16 CPUs (16 numerical threads by default) and has a 24-hour limit; it loads
the prepared matrix and saves the fitted basis. `--matrix-format` and `--row-stride` must
match between jobs; their defaults are `dense` and `1`. For example, use
`--row-stride 3` on both only if you intend to train on every third selected spectrum.
The NMF job checks for the matching prepared matrix and exits if preprocessing has not
produced it. To change its thread count, use `NMF_THREADS=8 sbatch runs/nmf_harps.sh`.

At full selection, the prepared matrix is about **431 GiB** before overhead. The processed
spectra cache is also about **431 GiB**, so allow roughly **862 GiB** across the two cache
locations, plus temporary and raw data. The processed flux cache is stored at
`/home/mward19/nobackup/autodelete/bosz-spectra/processed-harps/cache-high-res`; the prepared
matrix and fitted basis are stored in the repository under
`cache-high-res/harps_prepared` and `cache-high-res/stellar`, respectively. The paths are
defined in `runs/harps.py`.

For coordinate-descent NMF, reconstruction diagnostics reuse the weights learned during
training (`residual_weight_source="training_fit"`). They do not launch another iterative
weight solve. Check `diagnostics["converged"]` and `diagnostics["iteration_limit_reached"]`
for the basis fit's stopping status.

`.fit()` always trains. Passing `cache="cache/stellar"` saves the fitted basis and its
wavelengths and settings there. The files remain available after the script finishes.
The printed path is the one to pass to `.load()` in another script:

```python
stellar = SpectralNMF.load("cache/stellar/nmf_F_full_15262808.npy")
```

The filename above is an example. Slurm runs use their job ID; local runs use a parameter
hash. Use the actual path printed by your training run.

### Train a telluric basis

Construct a telluric loader for the wavelength range you want to model, then fit and save it:

```python
from pathlib import Path

from src.load_telluric import TelluricSpectra
from src.model import TelluricNMF

transmission = TelluricSpectra(
    pca_file=Path("data/telluric/TellPCA_3000_10500_R120000.fits"),
    D=None,  # Automatic sampling at three points per instrumental FWHM.
    wave_min=3800,
    wave_max=4200,
    R_out=115_000,
    cache_dir=Path("cache/processed-telluric"),
)
telluric = TelluricNMF(n_components=4)
telluric.fit(transmission, cache="cache/telluric")
print(telluric.cache_path)
```

Both training examples cover 3800–4200 Å. To fit a wider interval, prepare both models
with wider coverage. Their wavelength grids can differ, but a joint fit uses their overlap.

Constructing `TelluricSpectra` loads or reconstructs its processed transmission grid.
Constructing either NMF trainer only sets its configuration; training starts at `.fit()`.
Loading a fitted basis does not require the original dataset.

### Other training options

For minibatch training, configure the same trainer with:

```python
stellar = SpectralNMF(
    n_components=16,
    solver="minibatch",
    batch_size=256,
    max_epochs=1000,
    tol=1e-4,
)
stellar.fit(spectra, selection=selection, cache="cache/stellar")
```

Full NMF uses `max_iter` and a default stellar tolerance of zero, which runs the iteration
budget. Minibatch NMF uses `max_epochs` and defaults to a `1e-4` relative basis-change
threshold. It still requires an indexable prepared matrix in memory; it is not a streaming
reader. `TelluricNMF` defaults to full coordinate descent with `tol=1e-4`.

Both trainers also accept a prepared matrix and its wavelength array:
`stellar.fit(tau, wavelengths, cache="cache/stellar")`.
Here `tau` must already be **negative-log rectified flux**, with shape `(spectra, pixels)`.
Matrix inputs are used as supplied; apply normalization, convolution, clipping, and any
thresholding beforehand. Optional `preprocessing` and `source_ids` arguments record provenance.

## Find an existing fit by its settings

An explicit filepath is the simplest way to load a model. If a script should reuse a fit
when available and train otherwise, ask for matching parameters first:

```python
from src.model import CacheMiss

params = stellar.parameters_for(spectra, selection=selection)
try:
    stellar = SpectralNMF.load(cache="cache/stellar", params=params)
except CacheMiss:
    stellar.fit(spectra, selection=selection, cache="cache/stellar")
```

This continues the stellar training example. Parameters include training settings,
preprocessing, wavelengths, and selected-data identity. `.load()` never starts training.
`CacheValidationError` indicates an invalid explicit artifact; `CacheMiss` means automatic
lookup found no compatible artifact. `NMFCache` manages this storage internally; you can also
use that class directly for lower-level cache operations.

## Sampling and fit settings

`R_out` is resolving power. `D` is the number of wavelength samples. For both loaders,
omitting `D` (or setting `D=None`) chooses a logarithmic grid with at least approximately
three samples per instrumental FWHM:

```text
D = 1 + ceil(samples_per_fwhm * R_out * log(wave_max / wave_min))
```

`samples_per_fwhm=3.0` is the default; set it to `4.0` for finer sampling. There is no unique
maximum useful pixel count: this default samples the requested resolution without trying
to preserve arbitrary oversampling in the source files. An explicit integer `D` overrides
automatic sampling. Telluric spectra with `R_out=None` use the native resolving power read
from the PCA filename instead. Telluric wavelength bounds must still be supplied; BOSZ
can infer omitted bounds from the first raw spectrum.

The stellar example automatically chooses 34,530 samples across 3800–4200 Å. Across
3000–7000 Å at the same resolution, it chooses 292,319. Caches record the resolved integer
`D`, so automatic and explicit settings reuse the same processed cache when their grids
match. Reducing the pixel count alone
does not convolve a spectrum to lower resolution. Inference interpolates the bases at their
trained resolution; it does not perform instrumental convolution or resolution calibration.

To change the continuum model or numerical options:

```python
from src.model import ContinuumResponseBasis

normalizer = ContinuumNormalizer(
    stellar, response_basis=ContinuumResponseBasis(n_modes=5), variance_order=2,
)
result = normalizer.fit(
    wavelengths, flux, flux_errors, lsq_solver="exact", max_iter=500,
)
```

The default sparse solver is `lsq_solver="lsmr"`. The dense `"exact"` solver is useful for
small diagnostics when the design matrix fits in memory. Always check `result.success`.
`variance_order=1` preserves the original log-variance approximation; `2` uses the paper's
second-order approximation. A custom continuum function can be passed as `basis_function`
to `ContinuumResponseBasis`; it receives a full order and a mode count and returns its matrix.

## Saved files

Keep each model's three files together:

```text
nmf_F_full_15262808.npy
nmf_F_full_15262808.wavelengths.npy
nmf_F_full_15262808.json
```

They contain the basis, its explicit wavelengths, and metadata with settings, provenance,
diagnostics, and checksums. Job-ID filenames get numbered suffixes when one job produces
multiple configurations. Compatible files from previous jobs remain reusable.

Legacy `.npy/.json` F/G files still load by explicit path. Their logarithmic wavelengths are
inferred from their metadata. Automatic legacy lookup requires the original complete
parameter dictionary, since new fingerprints cannot be verified against old metadata.
The old training constructor and `get_F*` / `get_G` methods have been replaced by the trainers.

Minibatch checkpoints have `_last_epoch` in their names. To inspect one, use
`SpectralNMF.load(checkpoint_path, allow_checkpoint=True)`. This loads the current basis;
it does not resume optimizer state.

## Data, notebooks, and larger jobs

- [sandbox/nmf_subset.ipynb](sandbox/nmf_subset.ipynb) selects 108 training and 36 validation spectra
  across temperature, gravity, and metallicity strata. It compares sampling, thresholding,
  initialization, and continuum recovery. Set its data paths before running.
- [sandbox/nmf_basis_vectors.ipynb](sandbox/nmf_basis_vectors.ipynb) loads existing basis files and plots them.
- [construct_big_nmf.py](construct_big_nmf.py) loads compatible fits or trains on a cache miss.
  Review its paths, selection, sampling, and memory requirements before using it.

Open notebooks with:

```bash
uv run --with jupyterlab --with ipykernel jupyter lab sandbox/
```

For a training job, edit the script's configuration and submit:

```bash
mkdir -p logs
sbatch compute_F.sh
```

`construct_big_nmf.py` still uses `D=3000` across 3000–7000 Å, which undersamples HARPS
lines. Use the subset notebook to validate finer sampling before creating a new production cache.
The provisional `R_in=300000` assumption also needs checking against native BOSZ resolution.
The saved subset results still show substantial continuum errors for cool stars.

BOSZ data are available from [STScI](https://archive.stsci.edu/hlsp/bosz), and HARPS exposures
from [ESO](https://archive.eso.org) or [DACE](https://dace.unige.ch).
For bulk BOSZ downloads, run `scripts/get_scripts_bosz.sh` and `scripts/download_all_bosz.py`
from a separate data directory containing an empty `scripts/` subdirectory. The helpers use
`./scripts` and `./data` relative to that directory.
For the telluric PCA file:

```bash
TELLURIC_DIR="data/telluric" bash scripts/download_telluric.sh
```

Use the Python loaders for preprocessing. The older `scripts/load_spectra*.sh` files reference
a loader CLI absent from the current module. Running either loader module alone does not
process data. Processed-cache settings must match; use a new cache folder when changing them.

## Project layout and tests

```text
src/          Library code
scripts/      Download and batch-processing helpers
tests/        Automated tests
sandbox/      Jupyter and marimo notebooks
reports/      Diagnostic figures and experiment outputs
```

Use the project environment after `uv sync`; notebooks import the installed library directly.
Jupyter starts each notebook in `sandbox/`. The subset notebook saves to
`../reports/bosz_subset_validation/`; set that output path explicitly if running with another
working directory.
Open the older marimo notebooks with:

```bash
uv run marimo edit sandbox/processing_bosz.py
uv run marimo edit sandbox/processing_telluric.py
```


| Module | Responsibility |
|---|---|
| `src/model.py` | Observation fitting, continuum response, results, HARPS readers; public API imports |
| `src/nmf.py` | Stellar/telluric training, loading, and fixed-basis weight solving |
| `src/nmf_cache.py` | Shared artifact storage, lookup, and validation |
| `src/preparation.py` | Selection and optical-depth input preparation |
| `src/load_spectra.py` | Raw BOSZ I/O and processed-spectrum cache |
| `src/load_telluric.py` | PypeIt reconstruction and processed-transmission cache |

```bash
# Core checks require no external spectra.
uv run python -m unittest tests.test_nmf_cache tests.test_model -v

# Run all tests; data-dependent checks skip if their PCA input is absent.
uv run --with pytest pytest

# Run only the telluric integration checks.
uv run --with pytest pytest tests/test_load_telluric.py
```

Generated subset experiments are saved under `reports/bosz_subset_validation/` and ignored
by Git. The `sandbox/processing_*.py` files are exploratory marimo notebooks.
