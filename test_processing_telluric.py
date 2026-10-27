import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from sklearn.decomposition import NMF
    import numpy as np
    from matplotlib import pyplot as plt

    from load_telluric import TelluricSpectra

    return NMF, Path, TelluricSpectra, mo, np, plt


@app.cell
def _(Path, TelluricSpectra):
    # These must match the parameters load_telluric.py was run with; otherwise this
    # raises instead of reusing the cache
    telluric = TelluricSpectra(
        Path("/home/mward19/nobackup/autodelete/telluric/TellPCA_3000_10500_R120000.fits"),
        D=3000,
        wave_min=3000,
        wave_max=7000,
        R_out=115_000,
        n_pca=4,
        cache_dir=Path("/home/mward19/nobackup/autodelete/telluric/processed"),
    )
    wave = telluric.wave_out
    telluric_matrix = telluric.M # -log(T), shape (5000, D)
    telluric_matrix.shape
    return telluric_matrix, wave


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Telluric NMF basis vectors

    Factorize $-\log T \approx W G$ with $L = 4$ basis vectors, as in Casey et al. (2026).
    """)
    return


@app.cell
def _(NMF, telluric_matrix):
    nmf = NMF(n_components=4, solver="cd", max_iter=10_000)
    nmf.fit(telluric_matrix);
    return (nmf,)


@app.cell
def _(nmf, np, plt, wave):
    # Basis vectors G_l as transmission, exp(-G_l)
    fig, axes = plt.subplots(len(nmf.components_), 1, figsize=(10, 7), sharex=True)
    for _i, _ax in enumerate(axes):
        _ax.plot(wave, np.exp(-nmf.components_[_i]), lw=0.8)
        _ax.set_ylabel(f"exp(-G[{_i}])")
    axes[-1].set_xlabel("Wavelength [Å]")
    fig.suptitle("Telluric basis vectors")
    plt.show()
    return


@app.cell
def _(nmf, np, plt, wave):
    # Zoom on the strongest features: O2 B-band, around 6870 Å
    _mask = (wave >= 6800) & (wave <= 7000)
    fig2, axes2 = plt.subplots(len(nmf.components_), 1, figsize=(10, 7), sharex=True)
    for _i, _ax in enumerate(axes2):
        _ax.plot(wave[_mask], np.exp(-nmf.components_[_i])[_mask], lw=0.8)
        _ax.set_ylabel(f"exp(-G[{_i}])")
    axes2[-1].set_xlabel("Wavelength [Å]")
    fig2.suptitle("Telluric basis vectors, 6800-7000 Å")
    plt.show()
    return


@app.cell
def _(nmf, telluric_matrix):
    # How well do 4 basis vectors reconstruct the telluric matrix?
    _recon = nmf.transform(telluric_matrix) @ nmf.components_
    nmf.reconstruction_err_, abs(_recon - telluric_matrix).mean()
    return


if __name__ == "__main__":
    app.run()
