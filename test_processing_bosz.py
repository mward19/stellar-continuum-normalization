import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    from sklearn.decomposition import NMF
    import numpy as np
    from matplotlib import pyplot as plt
    from tqdm.auto import trange, tqdm

    from load_spectra import BOSZSpectra

    return BOSZSpectra, NMF, Path, np, plt, tqdm


@app.cell
def _(BOSZSpectra, Path):
    root = Path("/home/mward19/nobackup/autodelete/bosz-spectra/data")

    spectra = BOSZSpectra(
        root,
        300_000,
        115_000,
        D=3000,
        wave_min=3000,
        wave_max=7000,
        cache_dir=Path("/home/mward19/nobackup/autodelete/bosz-spectra/processed")
    )
    wave = spectra.wave_out
    return spectra, wave


@app.cell
def _(spectra):
    cached_ids = spectra._cached_ids()
    return (cached_ids,)


@app.cell
def _(cached_ids, plt, spectra, tqdm, wave):

    for _i, _id in tqdm(enumerate(list(cached_ids)[100:110])):
        plt.figure(figsize=(5, 3))
        plt.plot(wave, spectra.process_spectrum(_id), label=f"{_i}", alpha=0.5)

        plt.xlabel("Angstroms")
        plt.ylabel("Normalized Intensity")
        plt.title(f"processed spectrum {_i}")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try NMF
    """)
    return


@app.cell
def _(cached_ids, np, spectra, tqdm):
    spectral_matrix = np.vstack([spectra.process_spectrum(_id) for _id in tqdm(list(cached_ids))])
    return (spectral_matrix,)


@app.cell
def _(spectral_matrix):
    spectral_matrix
    return


@app.cell
def _(NMF):
    nmf = NMF(n_components=16, max_iter=1000)
    return (nmf,)


@app.cell
def _(spectral_matrix):
    spectral_matrix.shape
    return


@app.cell
def _(nmf, np, spectral_matrix):
    nmf.fit(-np.log(spectral_matrix));
    return


@app.cell
def _(nmf, np, plt, wave):
    mask = (wave >= 3800) & (wave <= 4200)
    fig, axes = plt.subplots(5)
    for _i, ax in enumerate(reversed(axes)):
        _spec = np.exp(-nmf.components_[_i])
        _n = 1
        _weights = np.repeat(1.0, _n) / _n
        _sma = np.convolve(_spec, _weights, 'same')
        ax.plot(wave[mask], _sma[mask])
    plt.show()
    return


@app.cell
def _(np, plt, spectral_matrix, wave):
    _spec = spectral_matrix[1]
    _n = 30
    _weights = np.repeat(1.0, _n) / _n
    _sma = np.convolve(_spec, _weights, 'same')
    plt.plot(wave, _sma)
    return


if __name__ == "__main__":
    app.run()
