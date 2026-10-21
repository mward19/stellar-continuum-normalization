import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    from sklearn.decomposition import NMF
    import numpy as np
    from matplotlib import pyplot as plt
    from tqdm.auto import trange, tqdm

    from load_spectra import BOSZSpectra

    return BOSZSpectra, NMF, Path, np, plt, tqdm, trange


@app.cell
def _(BOSZSpectra, Path):
    root = Path("/home/mward19/nobackup/autodelete/bosz-spectra/data-test")
    
    spectra = BOSZSpectra(
        root,
        300_000,
        115_000,
        D=3000,
        wave_min=3000,
        wave_max=7000
    )
    wave = spectra.wave_out
    return spectra, wave


@app.cell
def _(plt, spectra, tqdm, wave):
    indices = [1, 6, 7, 8]
    _n_spec = len(indices)
    for _i in tqdm(indices):
        plt.figure(figsize=(5, 3))
        plt.plot(wave, spectra[_i], label=f"{_i}", alpha=0.5)

        plt.xlabel("Angstroms")
        plt.ylabel("Normalized Intensity")
        plt.title(f"processed spectrum {_i}")
        plt.show()
    return (indices,)


@app.cell
def _(indices, spectra):
    print('IDs:')
    for _i in indices:
        print(f"{_i}: {spectra.ids[_i]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try NMF
    """)
    return


@app.cell
def _(np, spectra, trange):
    spectral_matrix = np.vstack([spectra[i] for i in trange(len(spectra))])
    return (spectral_matrix,)


@app.cell
def _(NMF):
    nmf = NMF(n_components=6, max_iter=10000)
    return (nmf,)


@app.cell
def _(spectral_matrix):
    spectral_matrix.shape
    return


@app.cell
def _(nmf, spectral_matrix):
    nmf.fit(spectral_matrix);
    return


@app.cell
def _(nmf, plt, wave):
    plt.plot(wave, nmf.components_[5])
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
