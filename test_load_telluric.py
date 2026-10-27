"""Run with: uv run --with pytest pytest test_load_telluric.py --basetemp ~/nobackup/autodelete/tmp/codex/pytest"""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from load_telluric import TelluricSpectra

PCA_FILE = Path("~/nobackup/autodelete/telluric/TellPCA_3000_10500_R120000.fits").expanduser()
pytestmark = pytest.mark.skipif(not PCA_FILE.is_file(), reason="run download_scripts/download_telluric.sh first")

KW = dict(D=3000, wave_min=3000, wave_max=7000, R_out=115_000, n_pca=4)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("telluric_cache")


@pytest.fixture(scope="module")
def tell(cache_dir):
    return TelluricSpectra(PCA_FILE, cache_dir=cache_dir, **KW)


def test_shape_and_range(tell):
    assert tell.M.shape == (5000, 3000)
    assert np.all(np.isfinite(tell.M))
    assert tell.M.min() >= 0
    T = np.exp(-tell.M)
    assert T.min() >= 0 and T.max() <= 1


def test_wave_grid_matches_stellar(tell):
    stellar = np.geomspace(KW["wave_min"], KW["wave_max"], KW["D"])
    assert np.array_equal(tell.wave_out, stellar)


def test_matches_direct_reconstruction():
    """Unconvolved, un-interpolated spectrum equals the formula applied by hand."""
    with fits.open(PCA_FILE) as h:
        comp, wave, coefs = (np.asarray(h[i].data, dtype=float) for i in (0, 1, 3))
    t = TelluricSpectra(PCA_FILE, D=3000, wave_min=3000, wave_max=7000, R_out=None, n_pca=4)
    for i in (0, 123, 4999):
        z = comp[0] + coefs[i, :4] @ comp[1:5]
        tau = np.clip(np.sinh(z), 0, None)
        expected = -np.log(np.clip(np.interp(t.wave_out, wave, np.exp(-tau)), 1e-10, 1))
        np.testing.assert_allclose(t[i], expected, rtol=1e-5, atol=1e-6)


def test_convolution_smooths(tell):
    raw = TelluricSpectra(PCA_FILE, D=3000, wave_min=3000, wave_max=7000, R_out=None, n_pca=4)
    assert not np.array_equal(raw.M, tell.M)
    assert np.mean(np.abs(raw.M - tell.M)) < 0.05 # a small change: 120k -> 115k


def test_cache_reload_matches(tell, cache_dir):
    again = TelluricSpectra(PCA_FILE, cache_dir=cache_dir, **KW)
    assert np.array_equal(again.M, tell.M)
    assert sorted(p.name for p in cache_dir.iterdir()) == ["telluric_neglogT.json", "telluric_neglogT.npy"]


@pytest.mark.parametrize("change", [
    dict(D=2000), dict(wave_min=3100), dict(wave_max=6900), dict(R_out=100_000),
    dict(n_pca=5), dict(t_floor=1e-6),
])
def test_param_change_raises(tell, cache_dir, change):
    with pytest.raises(ValueError, match="differ"):
        TelluricSpectra(PCA_FILE, cache_dir=cache_dir, **{**KW, **change})


def test_bad_inputs():
    with pytest.raises(ValueError, match="R_out"):
        TelluricSpectra(PCA_FILE, D=10, wave_min=3000, wave_max=7000, R_out=130_000)
    with pytest.raises(ValueError, match="n_pca"):
        TelluricSpectra(PCA_FILE, D=10, wave_min=3000, wave_max=7000, n_pca=11)
    with pytest.raises(ValueError, match="outside"):
        TelluricSpectra(PCA_FILE, D=10, wave_min=2000, wave_max=7000)
