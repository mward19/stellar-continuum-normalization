# Why the stellar NMF differs from Figure 1

Analysis date: 2026-11-13. Reviewed `test_nmfs.ipynb` **including its saved plots**, `src/load_spectra.py`, `src/model.py`, `construct_big_nmf.py`, `compute_F.sh`, the requested job log and its `.err` file, cached basis arrays and JSON metadata, and three representative raw BOSZ spectra. No training code or existing caches were changed, and no full training run was repeated.

## Main findings

**There is a substantial preprocessing problem: `D=3000` is much too small for a HARPS-resolution model spanning 3000–7000 Å.** Your code smooths to high spectral resolution and then samples that spectrum at intervals about 32.5 instrumental line widths apart. NMF cannot recover the discarded information. This is more consequential than the random seed.

Two other effects make the figure comparison misleading: the notebook displays a short cached fit, and the scales and ordering of individual NMF components are arbitrary. In particular, `exp(-F[k])` implicitly assigns a weight of one to that component, which need not be a representative physical weight.

**The repeated “Converged” messages are normal.** There is one main basis fit followed by 20 separate coefficient solves during residual evaluation. They do not mean that training repeatedly restarts or loses convergence.

**The main fit took 3985.9 seconds, or 66.4 minutes of wall time.** The log does not establish CPU time. The matrix is enormous and 79% occupied despite its sparse storage; the exact reason the authors' reported timing is faster cannot be established from these files alone.

## What is being compared

The publisher PDF could not be fetched through the browser, so I inspected the authors' accepted arXiv manuscript and its actual Figure 1. Its methods specify MARCS/BOSZ rectified flux, convolution to approximately R=115,000, a 300–700 nm interval, negative-log NMF, 16 components, NNDSVD initialization with average-filled zeros, and coordinate descent. They report 10,000 training steps and approximately 30 minutes on one CPU core. Figure 1 covers roughly **3800–5000 Å**, rather than the notebook's 3800–4200 Å. [Casey et al., accepted manuscript, Section II and Figure 1](https://arxiv.org/html/2601.19973v1#S2)

The saved notebook output identifies its stellar cache as:

```text
nmf_F_full_fbcf2defd9.npy
```

Its adjacent JSON specifies `max_iter=100`, `tol=1e-4`, `init="nndsvda"`, `sparse_tol=0.01`, and MARCS atmospheres with `teff_max=8000`. The notebook selected 396,108 of 628,620 spectra. **This last plot is a full coordinate-descent NMF result, not a MiniBatchNMF result.** The separate `test_nmfs.py` file contains older experiments and does not describe this saved notebook execution.

The requested log, [spectral_NMF_marcs_15262808.out](logs/spectral_NMF_marcs_15262808.out), records `max_iter=500`, the same number of training spectra, and `teff_max=None`. The matching available cache is `nmf_F_full_b011815400.npy`. In this dataset the temperature-filter difference does not change the selected count: BOSZ 2024 MARCS models stop at 8000 K. [STScI BOSZ grid documentation](https://archive.stsci.edu/hlsp/bosz)

The notebook and log therefore concern different cached fits. Increasing the iteration budget can change the short-run result, but it does not repair its sampling.

## 1. What D means, and why 3000 is wrong here

In [load_spectra.py](src/load_spectra.py), the constructor creates:

```python
self.wave_out = np.geomspace(self.wave_min, self.wave_max, self.D)
```

`D` is the **number of wavelength samples**, not resolving power and not the number of NMF components. `K=16` controls the number of components. The constructor comment `D, # Output resolution` obscures this distinction.

For your logarithmic wavelength grid,

```text
Δln λ = ln(7000/3000) / (3000−1) = 0.000282527
Δv    ≈ c Δln λ                  = 84.70 km/s
```

At 4000 Å the sample spacing is about 1.130 Å, whereas an R=115,000 instrumental FWHM is only `4000/115000 = 0.0348 Å`. There are **0.0308 samples per FWHM**, or **32.5 FWHM between samples**. Only 354 samples fall in the notebook's 3800–4200 Å window.

The processing sequence in `_compute_spectrum()` is:

1. Divide emergent flux by theoretical continuum.
2. Convolve to the requested resolving power.
3. Crop the wavelength range.
4. Evaluate `np.interp()` on the 3000-point grid.

Step 4 does not average a bin's flux or apply an anti-aliasing filter appropriate to the coarse output grid. It picks values at widely separated locations. Narrow lines can be missed, or a deep line can be sampled at one point and appear as a broad triangular dip when connected to its neighbors. Across a dense line forest this produces irregular wavelength structure. **This is undersampling/aliasing, not a properly smoothed low-resolution spectrum.** Smoothing to R=115,000 beforehand is nowhere near sufficient for this degree of downsampling.

I checked a solar-composition 6000 K raw spectrum using your convolution function and overlaid the 3000-point sampling. The upper panel shows the loss directly. The lower panel illustrates the independent basis-scale issue discussed below.

![Sampling and basis-scale diagnostics](reports/nmf_diagnostics.png)

The upper panel uses a padded 3980–4030 Å extraction before convolution and displays its interior, 4000–4010 Å; it is a local diagnostic, not a rerun of the full cache pipeline. The lower panel uses the notebook's actual cached first basis vector.

### A defensible choice of D

For `s` samples per instrumental FWHM on this logarithmic grid, require approximately:

```text
D ≥ 1 + ceil[s × R × ln(λmax/λmin)]
```

For R=115,000 and 3000–7000 Å:

| Samples per FWHM | Minimum D from this criterion |
|---|---:|
| 2 | 194,880 |
| 3 | 292,319 |
| 4 | 389,759 |

I would start a resolution-preserving experiment around **D=300,000**, then check whether reconstruction and interpolation accuracy change with finer sampling. These are sampling calculations, not a claim about the authors' exact grid length. Two samples per FWHM is a rough lower-bound heuristic; Gaussian profiles are not strictly band limited.

If memory makes that impractical, reduce the training-set size or investigate a short wavelength interval first. For a genuinely lower-resolution model, choose a lower target R, convolve appropriately, and then sample densely enough for that R. Setting `D=3000` while retaining `R_out=115000` does not accomplish that.

The telluric setup also uses `D=3000` over the same interval, so its narrow-line representation has the same sampling limitation. That does not cause the stellar-basis plot, since F is trained separately, but it matters for later fits.

## 2. Why the first basis vector need not look like the paper's first vector

### Component number is not a physical label

NMF does not order components by explained variance as PCA does. A permutation of the columns of W and corresponding rows of F leaves the reconstruction unchanged. `F[0]` has no universal status as a hydrogen-line component, or as the smoothest component.

The broad depressions in the paper's bottom curve are consistent with broad hydrogen absorption; that is a physical interpretation of its shape, not a guaranteed property of component zero. A different factorization can distribute those features among several components. The unregularized objective in your code imposes non-negativity, but **no wavelength-smoothness constraint**.

### Basis amplitudes are also arbitrary

With the code's conventional shapes, write:

```text
X ≈ W F,     X = −ln(S).
```

For any positive `c_k`, replacing `F[k]` by `F[k]/c_k` and `W[:,k]` by `W[:,k]*c_k` gives the same product. But it changes the appearance of `exp(-F[k])` dramatically.

I measured the notebook cache:

- All entries are finite and non-negative; shape is `(16, 3000)` and dtype is float32.
- Its largest entry is **118.42**.
- Row L2 norms range from approximately **41.4 to 1089.2**.
- For component 7, about **85.6%** of the samples in 3800–4200 Å satisfy `exp(-F[7]) < 0.01`.

Consequently, the many nearly saturated curves are partly a consequence of plotting unit-weight components at very different scales. They are not, by themselves, evidence of invalid reconstructed stellar flux.

For a shape comparison, display `F[k]/max(F[k])`, or a stated, consistent positive scaling before exponentiating. For a physical test, solve weights and inspect `exp(-(W @ F))`. If you normalize F for actual fitting, compensate the weights as above. Rescaling cannot recover missing lines or turn a mixed component into a pure broad-line component.

### The short run is another difference, but not the main repair

The notebook cache was created with a 100-iteration cap; its JSON does not retain the actual stopping iteration. The longer logged run stopped after 346 iterations at its tolerance. I also inspected that longer run's cached F: it remains finite and non-negative, with maximum entry **141.34** and comparably large scale differences. Thus additional iterations do not automatically fix the appearance of unscaled `exp(-F)` plots.

Compare reconstructions, component shapes after a stated scaling, and multiple initializations. Do not use agreement of row zero alone as a pass/fail test. Jointly optimizing W and F is non-convex; optimizing non-negative weights with F fixed is convex. Iterative NMF is not guaranteed to find a unique globally optimal basis.

## 3. Other preprocessing issues worth correcting

### Weak absorption is explicitly deleted

`BOSZSpectra.sparse()` sets every optical-depth value below `tol` to zero. For the full fits, `sparse_tol=0.01` means:

```text
−ln(S) < 0.01  ⇔  S > exp(−0.01) ≈ 0.99005.
```

Absorption shallower than approximately **0.995%** is replaced by exactly continuum. This changes the training target; it is not just a storage format. It can remove wings and weak blends and introduce threshold boundaries. In subsequent least squares those zeros are ordinary target values, not missing pixels.

Test zero threshold or a substantially smaller threshold on a manageable subset. Measure the reconstruction cost and memory tradeoff. A large threshold should not be justified solely by a desire to call the matrix sparse.

### The input resolution is not uniformly 300,000

The cache database confirms that the input products are `rorig/noresam`. STScI describes these original spectra as having parameter-dependent native resolution of roughly **200,000–600,000**. They are not a uniform R=300,000 product. [STScI BOSZ 2024 product definitions](https://archive.stsci.edu/hlsp/bosz)

Your Gaussian quadrature formula is appropriate if the input and output line-spread functions are Gaussian with known widths, but its fixed `R_in=300000` assumption needs verification against the native synthesis prescription. For illustration, if a native spectrum were equivalent to Gaussian R=200,000, your added kernel would give approximately R=105,700 rather than 115,000. At R=600,000 it would give approximately R=121,900. Those examples are conditional calculations, not measurements of these spectra's actual line-spread functions.

This is secondary to the factor-of-32.5 sampling problem. Check native line profiles and wavelength spacing before choosing a uniform convolution prescription.

### What the inspection did not find

The flux/continuum division is consistent with the three-column `rorig` products. Representative 3500, 6000, and 8000 K solar-composition spectra had finite, positive optical-range ratios no greater than one, and the cached samples had plausible flux ranges. These checks do not establish the quality of the entire grid, but they give no evidence for a missing global flux conversion factor.

The sampled raw wavelength arrays had duplicate values but no descending steps. The routine sorts internally and returns flux in sorted order while its caller retains the original wavelength array; that interface is fragile for truly unordered inputs, but these checks did not establish a consequential ordering bug in the inspected files. Return wavelengths with flux, and explicitly handle duplicates when improving the loader.

`flux_errors=1e-8`, Fourier modes, radial velocity, and the HARPS observation reader do not enter `get_F_full()`'s NMF objective. They cannot explain this particular basis plot. Separately, `fit()` uses a first-order log-variance approximation; that concerns inference, not basis training.

## 4. Why it “converges” repeatedly

The exact sequence is visible in the log and in [model.py, `_report_residuals()`](src/model.py):

| Log evidence | Meaning |
|---|---|
| Lines 12–358: violations and `Converged at iteration 347` | One joint fit of W and F |
| Lines 359–360: fit finished; `346 iterations`, `3985.9s` | Main training has ended |
| Lines 361–1374: 20 new violation sequences and convergence messages | Solve W on successive residual-check batches, with F fixed |
| Line 1375: residual summary | Evaluation complete |
| Line 1376: cached G loaded | Separate telluric basis reused |

The residual checker selects 20,000 training spectra and processes them in batches of 1024: `ceil(20000/1024)=20`. Each call to `nmf.transform(X_batch)` solves for fresh non-negative weights. It inherits `verbose=1`, so each solve prints its own progress. The installed scikit-learn implementation explicitly calls `_fit_transform(..., H=self.components_, update_H=False)` for `transform()`: **it does not update F**. The public API likewise documents `transform()` as obtaining weights from a fitted model. [scikit-learn NMF API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)

The installed solver also explains the apparent 347/346 mismatch: its loop starts at one but the convergence print uses `n_iter + 1`, whereas the returned iteration count is `n_iter`. This is an off-by-one display discrepancy in the installed implementation, not another iteration or another fit.

`violation` is a normalized projected-gradient stopping statistic, computed relative to the initial violation of **each solve**. It is not fractional reconstruction error. Each new solve starts at 1.0 by construction; fluctuations do not alone imply increasing reconstruction loss.

To make the log clearer, temporarily set `nmf.verbose=0` around residual evaluation and add a single “evaluating residuals with F fixed” message. This is a logging improvement, not a mathematical repair.

The reported relative Frobenius residual is **0.1063**, and RMS is **0.1796 in negative-log-flux units**. These are reconstruction diagnostics on sampled **training** rows and the thresholded target, not held-out accuracy, continuum error, or a statement of 10.63% flux error.

The `.err` file also contains `RuntimeWarning: invalid value encountered in sqrt` from scikit-learn's reconstruction-error calculation. Its sparse float32 path evaluates a difference of large norm terms, which can become negative through numerical error. The finite cached factors and explicitly recomputed float64 residual support a numerical diagnostic issue rather than proof that F is NaN. Your explicit residual routine already addresses this; check factor finiteness and compute direct residuals on batches when evaluating quality.

## 5. Why runtime differs so much

### What the log actually establishes

| Stage | Observed wall time |
|---|---:|
| Load/prepare cached sparse matrix | 31.6 s |
| Select training rows | 6.7 s |
| `NMF.fit()`, including initialization | 3985.9 s = 66.4 min |
| Residual evaluation | Approximately 17 s in `.err` progress output |

This was not a 10,000-iteration run: it stopped at 346. `max_iter=500` is an upper limit. The source comment in `construct_big_nmf.py` describing 10,000 steps is stale relative to the executable arguments.

The fit timing is obtained with `time.perf_counter()`, so it measures elapsed wall time. CPU time must come from process accounting or Slurm accounting, for example:

```bash
sacct -j 15262808 --format=JobID,Elapsed,TotalCPU,AllocCPUS,MaxRSS,State
```

### This matrix is expensive despite its sparse type

The selected fit matrix is `(396108, 3000)` with **938,948,151 nonzero entries**, or **79.01% density**. Assuming float32 data and 32-bit sparse indices, it occupies approximately **7.51 GB in CSR**, versus **4.75 GB as dense float32**. The full preselection sparse matrix is also retained. These are array-size calculations, not measured peak resident memory.

The installed coordinate-descent code repeatedly multiplies X and its transpose by dense factor matrices. With K=16, the two sparse products alone involve roughly **30 billion nonzero–component contributions per full iteration**, before the coordinate updates. Sparse indexing and memory access can cost more than dense BLAS at this occupancy. Benchmark both formats on the same representative subset before committing to either.

Your Slurm script also requests `--ntasks=4`, launches one Python process, and sets only `OMP_NUM_THREADS`. That does not create four distributed NMF workers. The inspected coordinate-update kernel contains serial loops, and BLAS threading depends on the linked backend and environment. Prefer `--ntasks=1` with an explicit `--cpus-per-task`, inspect `threadpoolctl.threadpool_info()`, and benchmark thread counts. More allocated CPUs do not necessarily accelerate this solver.

These facts explain why a lengthy fit is plausible, but **do not quantitatively account for the authors' much faster per-step timing**. Their effective sampling, matrix occupancy, implementation details, hardware, and timing boundaries are not established by this comparison. It would be unjustified to attribute the entire gap to one factor, or promise their timing after a single setting change. The repeats after training contribute only seconds here.

Simply raising D to 300,000 for all 396,108 rows would require about **475 GB for dense float32 X alone**. A corrected sampling experiment therefore needs a deliberate memory strategy; fix the grid on a small, representative training set before scaling up.

## 6. Recommended next experiment

1. **Build a fresh small cache with adequate wavelength sampling.** Choose a parameter-stratified subset covering temperature, gravity, composition, and microturbulence, rather than taking the first files returned by directory traversal. Use R≈115,000 with approximately three samples per FWHM, initially on a short padded wavelength interval. Verify raw, convolved, and resampled line profiles before NMF.
2. **Check the native-resolution prescription and remove aggressive thresholding for the diagnostic.** On this manageable subset, compare `sparse_tol=0` with 0.01 and inspect their effects on weak lines and wings. Benchmark dense versus sparse storage with the resulting occupancy.
3. **Keep K=16 and NNDSVD-average initialization for the first comparison.** Use enough iterations to assess a stable reconstruction, then compare seeds and initialization choices. Increasing K or iteration count should follow evidence from reconstruction quality; neither restores discarded samples.
4. **Validate on held-out spectra at the original useful resolution.** Solve non-negative weights for fixed F, reconstruct `S_hat=exp(-(W @ F))`, and inspect flux residuals, especially broad wings and narrow line forests. Report errors by stellar type and wavelength as well as a global norm. Only then test continuum recovery from representative observed spectra.
5. **Make the figure comparison explicit.** Use 3800–5000 Å, document component rescaling, and inspect all components for broad-line structure. Compare the reconstructed absorption model before judging component ordering.
6. **Scale the training set gradually and record provenance.** Store actual iteration count, fit/evaluation timings, density, dtype, software versions, thread settings, and a manifest or hash of selected spectrum IDs. The existing cache keys include processing parameters but do not uniquely identify the input manifest. Use new cache directories when changing D or preprocessing, and bump processing/NMF version numbers for algorithm changes so old results cannot be silently reused.

The strongest conclusion is that the current output is an NMF of an aggressively undersampled and thresholded spectral grid. Correct that input representation first; then evaluate reconstruction accuracy and interpret the basis plots with their scale and permutation freedom in mind.
