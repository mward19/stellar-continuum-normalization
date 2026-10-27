In load_spectra.py, I have written code to process a bunch of spectra. The script download_scripts/download_all_bosz.py downloads the BOSZ spectra. Now I want to download and process the data for telluric lines, as in pages 4 and 5 of https://iopscience.iop.org/article/10.3847/1538-4357/ae3bd6/pdf. I want to replicate their work as closely as possible. Make a plan for the telluric download scripts and the load_telluric.py file that will make it possible to process the telluric data.

# Plan

## 1. What the paper does (Casey et al. 2026, ApJ 998:192, pp. 4-5)

- Telluric transmission comes from the grid computed for the **PypeIt** project (Prochaska et al. 2025): LBLRTM (Clough et al. 2005) over observatory locations, temperatures, pressures, humidities and air masses.
- The original grid wasn't available to the authors. They used the **PCA-compressed representation shipped with PypeIt** (PCA of an inverse-hyperbolic-sine transform), and **reconstructed the original grid from it**.
- They factorize the negative log of the reconstructed transmission, `M = -log(T) ≈ W G` (their Eq. 6), into two nonnegative matrices using **L = 4** basis vectors ("the dimensionality of the PCA-compressed grid"), with **10,000 coordinate-descent iterations**.
- At inference time telluric transmission is `exp(-G^T beta)` with `beta >= 0` (Eqs. 7, 15). That inference step is out of scope here. This plan only produces `M` on the right wavelength grid.
- I only read pp. 4-5 (plus the title page). The text there does not say how the grid was resampled, convolved or masked. Anything about that below is my choice, not something I can claim they did.

## 2. Source data (verified by downloading and inspecting it)

- Use the PypeIt PCA file `TellPCA_3000_10500_R120000.fits` (36 MiB, 37,748,160 bytes).
- URL: `https://s3-west.nrp-nautilus.io/pypeit/telluric/atm_grids/TellPCA_3000_10500_R120000.fits`.
  - The host comes from `pypeit/data/s3_url.txt` in the PypeIt repo. A HEAD request returned 200.
  - The PypeIt docs say it's one of five `TellPCA_*` files, and that the S3 location can change. The download script should keep the URL in one variable.
  - We don't need to install PypeIt.
- File layout:

  | HDU | Contents | Shape |
  |---|---|---|
  | 0 (primary) | PCA components. Row 0 is the mean. Header `NCOMP = 10` | (11, 388844) |
  | 1 | wavelength grid, 2900-10600 Å, log-uniform with dlnλ = 1/300000 | (388844,) |
  | 2 | coefficient bounds. Row 0 is the lower bound and row 1 the upper bound. Column 0 is a placeholder 1 for the mean | (2, 11) |
  | 3 | **model coefficients: 5000 rows x 10 PCA coefficients** | (5000, 10) |

- I checked that HDU 2 equals the min and max of every column of HDU 3 exactly. So **HDU 3 is the PCA-space version of the training grid: 5000 spectra**. That is what "reconstruct the original grid" means.
- Reconstruction, from PypeIt's `eval_telluric`:
  ```
  z   = comp[0] + coefs[:, :L] @ comp[1:L+1]     # L = number of PCA components used
  tau = sinh(z); tau[tau < 0] = 0                # the stored transform is arcsinh(tau)
  T   = exp(-tau)
  ```
  - `-log(T)` is then just `tau`.
  - I tried this on 200 random rows over 4000-7000 Å. T stays in [0, 1], about 0.9% of `tau` values are negative before clipping, and the mean T is 0.974. Using 4 or 10 components gives nearly the same statistics.
- Provenance (PypeIt docs): LBLRTM v12.17, HITRAN 2016 line parameters, six observatories at different altitudes.
- Scratch copies and inspection scripts live in `~/nobackup/autodelete/tmp/codex/tell` and `~/nobackup/autodelete/tmp/codex/*.py`. They are not part of the deliverable.

## 3. Download script: `download_scripts/download_telluric.sh`

- Same style as `get_scripts_bosz.sh`: `set -euo pipefail`, `curl --fail --location --retry 3`, download to a `mktemp` name, then `mv` into place.
- Keep the base URL and filename in variables at the top.
- Output goes to `~/nobackup/autodelete/telluric/` (next to `bosz-spectra`), not into the repo.
- Skip the download if the file already exists with the expected size (37,748,160 bytes). Otherwise fail loudly.
- It's one 36 MiB file, so there is no need for the parallel machinery in `download_all_bosz.py`.
- No Slurm job is needed. It can run on a login node.

## 4. `load_telluric.py`

A small class like `BOSZSpectra`, with the same conventions: `wave_out`, `D`, `wave_min`, `wave_max`, `__len__`, `__getitem__`.

**Constructor** `TelluricSpectra(pca_file, D, wave_min, wave_max, R_out=None, n_pca=4, cache_dir=None)`

- `D`, `wave_min` and `wave_max` must be identical to the stellar side. `wave_out = np.geomspace(wave_min, wave_max, D)`, so the stellar and telluric bases share one wavelength grid. Defaults to match: `D=3000`, 3000-7000 Å.

**Steps** (all in memory):
1. Read HDUs 0-3 with `astropy.io.fits`.
2. Crop the wavelength axis to the output range plus a small pad, so convolution edges don't leak in. The crop is about 254k of the 388,844 points.
3. In chunks of rows (the whole matrix at once would be about 10 GB), build `tau` as in section 2, for all 5000 coefficient rows, using the first `n_pca` components.
4. Optionally convolve to `R_out`. Reuse `BOSZSpectra.convolve_to_R`, with `R_in=120000` (the file's resolution).
5. Interpolate onto `wave_out` with `np.interp`.
6. Result: `M`, shape (5000, D), float32 or float64, with `M >= 0`. That is the `-log T` matrix the paper factorizes. Provide `transmission(i) = exp(-M[i])` as a convenience.

**Cache**
- 5000 x 3000 x 8 bytes is about 120 MB, so a single `telluric_<...>.npy` is fine. The file quota is not a concern here, unlike for the stellar spectra.
- Write a JSON sidecar with the parameters: `D`, `wave_min`, `wave_max`, `R_out`, `n_pca`, the PCA file name, and a `PROCESSING_VERSION`.
- On init, if a cache exists and its parameters differ, **raise an error**, the same behavior as `BOSZSpectra`. If there's no cache, compute it and save it, using a temp file plus rename.

**Dependencies**
- `astropy` is not in `pyproject.toml`. Add it with `uv add astropy`.

## 5. Following steps (not part of this task)

- Add a telluric NMF in `spectral_nmf.py`:
  - input `M` from `TelluricSpectra`
  - `NMF(n_components=4, solver="cd", max_iter=10000)`
  - compare to the stellar NMF (K = 16 in the paper)
- The stellar side would use the same recipe on `-log` of the rectified BOSZ flux.

## 6. Tests

- Check the reconstruction against PypeIt's own `eval_telluric` output for a few coefficient rows. I'd prefer to do this in a scratch environment and not add PypeIt as a dependency.
- Check that `M >= 0`, that `exp(-M)` lies in [0, 1], and that the output has shape (5000, D).
- Check that a changed `D`, `wave_min`, `wave_max`, `R_out` or `n_pca` raises an error against an existing cache.
- Check that a second run loads from the cache and matches the first.

## 7. Open questions (recommended defaults)

1. **Which PCA file.** Default: `R120000`, 3000-10600 Å. It matches HARPS (R about 115,000), the paper's instrument, and your stellar settings (`R_out=115000`, 3000-7000 Å). The other four files have lower resolution or a different wavelength range.
2. **How many components to reconstruct with.** Default: `n_pca=4`. It matches PypeIt's default `tell_npca=4` and the paper's L = 4. Using all 10 gives nearly the same statistics, but I can't tell which the authors used.
3. **Convolve 120,000 to 115,000?** Default: yes, for consistency with the stellar spectra. The paper text I read doesn't say. If you'd rather keep the file's native resolution, set `R_out=None`.
4. **Convolve transmission or `tau`?** Default: convolve `T`, then take `-log`. That is physically correct but needs a floor on `T` to avoid `log(0)`. Convolving `tau` is simpler. Say if you prefer it.
5. **Add `astropy` to `pyproject.toml`?** Default: yes.
