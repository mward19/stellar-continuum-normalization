import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    from src.load_spectra import BOSZSpectra
    from src.load_telluric import TelluricSpectra
    from src.model import ContinuumNormalizer

    import numpy as np
    from matplotlib import pyplot as plt

    return BOSZSpectra, ContinuumNormalizer, Path, TelluricSpectra, np, plt


@app.cell
def _(np):
    G = np.load("/home/mward19/nobackup/autodelete/telluric/processed/nmf_G_a406df2c65.npy")
    G.shape
    return


@app.cell
def _(BOSZSpectra, Path, TelluricSpectra):
    # Same parameters as the caches the NMFs were built from, so nothing is recomputed
    bosz = BOSZSpectra(
        Path("/home/mward19/nobackup/autodelete/bosz-spectra/data"),
        300_000,
        115_000,
        D=3000,
        wave_min=3000,
        wave_max=7000,
        cache_dir=Path("/home/mward19/nobackup/autodelete/bosz-spectra/processed-harps"),
        spectra_per_file=1000,
    )
    telluric = TelluricSpectra(
        Path("/home/mward19/nobackup/autodelete/telluric/TellPCA_3000_10500_R120000.fits"),
        D=3000,
        wave_min=3000,
        wave_max=7000,
        R_out=115_000,
        n_pca=4,
        t_floor=1e-10,
        cache_dir=Path("/home/mward19/nobackup/autodelete/telluric/processed"),
    )
    return bosz, telluric


@app.cell
def _(ContinuumNormalizer, bosz, np, telluric):
    wave = bosz.wave_out
    # Finds nmf_F_3b9d9a3bfc.npy and nmf_G_a406df2c65.npy in the caches above instead of recomputing;
    # F_kwargs are the get_F settings that file was made with (they are part of its name)
    cn = ContinuumNormalizer(
        bosz,
        telluric,
        flux_errors=1e-8 * np.ones(3000),
        order_wavelengths=wave,
        F_kwargs=dict(n_epochs=4, batch_size=1024),
    )
    return cn, wave


@app.cell
def _():
    return


@app.cell
def _(cn):
    F_minibatch = cn.F.copy() # the cached MiniBatchNMF result, kept for comparison
    return (F_minibatch,)


@app.cell
def _(F_minibatch, np, plt, wave):
    _n_vecs = 16
    _fig, _axes = plt.subplots(_n_vecs, figsize=(15, 30))

    for _i, _ax in enumerate(_axes):
        _ax.plot(wave, np.exp(-F_minibatch[_i]))

    plt.show()
    return


@app.cell
def _(cn):
    # MARCS atmospheres only, Teff <= 8000 K, as in the paper; then one ordinary NMF by coordinate
    # descent. The paper runs 10_000 steps (~30 min); this is a short run to check the vectors.
    cn.filter_spectra()
    cn.fit_full(max_iter=100)
    F_full = cn.F.copy()
    return (F_full,)


@app.cell
def _(F_full, np, plt, wave):
    # Like Figure 1 of Casey et al. (2026): basis vectors exp(-F) with offsets, 3800-4200 A
    _window = (wave >= 3800) & (wave <= 4200)
    _fig, _ax = plt.subplots(figsize=(12, 8))
    for _i, _f in enumerate(F_full):
        _ax.plot(wave[_window], np.exp(-_f[_window]) + _i, lw=0.8)
    _ax.set_xlabel("Wavelength [A]")
    _ax.set_ylabel("exp(-F) + offset")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
