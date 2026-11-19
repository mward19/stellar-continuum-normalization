"""Automatic resolution sampling and compatibility with explicit-grid caches."""

import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from src.load_spectra import BOSZSpectra
from src.load_telluric import TelluricSpectra
from src.wavelength import log_wavelength_grid


class WavelengthGridTests(unittest.TestCase):
    def test_default_grid_samples_requested_resolution(self):
        wave = log_wavelength_grid(3800, 4200, resolving_power=115000)
        self.assertEqual(len(wave), 34530)
        self.assertEqual(wave[0], 3800)
        self.assertEqual(wave[-1], 4200)
        points_per_fwhm = 1 / (115000 * np.diff(np.log(wave)))
        self.assertGreaterEqual(points_per_fwhm.min(), 3 * (1 - 1e-8))
        self.assertLess(points_per_fwhm.max(), 3.001)
        self.assertEqual(len(log_wavelength_grid(3000, 7000, resolving_power=115000)), 292319)

    def test_explicit_D_preserves_original_grid(self):
        expected = np.geomspace(3800, 4200, 123)
        for resolution, sampling in [(115000, 3), (120000, 4)]:
            np.testing.assert_array_equal(
                log_wavelength_grid(
                    3800,
                    4200,
                    D=123,
                    resolving_power=resolution,
                    samples_per_fwhm=sampling
                ), expected)
        self.assertEqual(len(log_wavelength_grid(3800, 4200, D=np.int64(123))), 123)

    def test_finer_sampling_is_configurable(self):
        wave = log_wavelength_grid(3800, 4200, resolving_power=115000, samples_per_fwhm=4)
        self.assertGreater(len(wave), 34530)
        self.assertGreaterEqual(np.min(1 / (115000 * np.diff(np.log(wave)))), 4 * (1 - 1e-8))

    def test_invalid_sample_counts_and_bounds(self):
        for D in [0, 1, True, np.bool_(True), 3.5]:
            with self.subTest(D=D), self.assertRaises(ValueError):
                log_wavelength_grid(3800, 4200, D=D)
        for bounds in [
            (0, 4200),
            (-1, 4200),
            (4200, 3800),
            (3800, 3800),
            (np.nan, 4200),
            (3800, np.inf),
            (None, 4200)
        ]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                log_wavelength_grid(*bounds, D=100)

    def test_automatic_grid_requires_valid_resolution_and_sampling(self):
        for resolution in [None, 0, -1, np.nan, np.inf]:
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                log_wavelength_grid(3800, 4200, resolving_power=resolution)
        for sampling in [0, -1, np.nan, np.inf]:
            with self.subTest(sampling=sampling), self.assertRaises(ValueError):
                log_wavelength_grid(3800, 4200, resolving_power=115000, samples_per_fwhm=sampling)


class LoaderGridTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        wave = np.geomspace(3790., 4210., 2000)
        tau = .2 * np.exp(-.5 * ((wave - 4000) / 3)**2)
        filename = self.raw / 'bosz2024_mp_t6000_g+4.0_m+0.00_a+0.00_c+0.00_v1_rorig_noresam.txt.gz'
        with gzip.open(filename, 'wt') as stream:
            np.savetxt(stream, np.column_stack([wave, np.exp(-tau), np.ones_like(wave)]))
        self.pca = self.root / 'TellPCA_3000_10500_R120000.fits'
        components = fits.PrimaryHDU(np.stack([np.arcsinh(tau), np.zeros_like(tau)]))
        components.header['NCOMP'] = 1
        fits.HDUList([
            components,
            fits.ImageHDU(wave),
            fits.ImageHDU(),
            fits.ImageHDU(np.array([[0.], [.1]]))
        ]).writeto(self.pca)

    def test_bosz_omitted_D_reuses_explicit_grid_cache(self):
        settings = dict(
            root=self.raw,
            R_in=300000,
            R_out=115000,
            wave_min=3800.,
            wave_max=4200.,
            cache_dir=self.root / 'processed'
        )
        manual = BOSZSpectra(D=34530, **settings)
        self.addCleanup(manual.db.close)
        flux = manual[0]
        manual.flush_cache()
        automatic = BOSZSpectra(**settings)
        self.addCleanup(automatic.db.close)
        self.assertEqual(automatic.D, 34530)
        self.assertEqual(automatic._processing_params(), manual._processing_params())
        np.testing.assert_array_equal(automatic.wave_out, manual.wave_out)
        with patch.object(
            automatic,
            '_compute_spectrum',
            side_effect=AssertionError('Must reuse cache')
        ):
            np.testing.assert_array_equal(automatic[0], flux)

    def test_bosz_auto_D_also_handles_inferred_bounds(self):
        loader = BOSZSpectra(
            self.raw,
            R_in=300000,
            R_out=115000,
            samples_per_fwhm=4
        )
        self.assertAlmostEqual(loader.wave_min, 3790.)
        self.assertAlmostEqual(loader.wave_max, 4210.)
        self.assertGreaterEqual(np.min(1 / (loader.R_out * np.diff(np.log(loader.wave_out)))),
                                4 * (1 - 1e-8))

    def test_telluric_auto_grid_uses_target_or_native_resolution(self):
        for resolution, expected_D in [(115000, 34530), (None, 36032)]:
            with self.subTest(resolution=resolution):
                loader = TelluricSpectra(
                    self.pca,
                    wave_min=3800.,
                    wave_max=4200.,
                    R_out=resolution,
                    n_pca=1
                )
                self.assertEqual(loader.D, expected_D)
                self.assertEqual(loader.M.shape, (2, expected_D))
                self.assertTrue(np.isfinite(loader.M).all())

    def test_telluric_auto_grid_reuses_explicit_grid_cache(self):
        settings = dict(
            wave_min=3800.,
            wave_max=4200.,
            R_out=115000,
            n_pca=1,
            cache_dir=self.root / 'telluric'
        )
        manual = TelluricSpectra(self.pca, D=34530, **settings)
        with patch.object(
            TelluricSpectra,
            '_compute',
            side_effect=AssertionError('Must reuse cache')
        ):
            automatic = TelluricSpectra(self.pca, **settings)
        self.assertEqual(automatic._params(), manual._params())
        np.testing.assert_array_equal(automatic.wave_out, manual.wave_out)
        np.testing.assert_array_equal(automatic.M, manual.M)

    def test_telluric_explicit_positional_D_still_works(self):
        loader = TelluricSpectra(self.pca, 50, 3800., 4200., R_out=115000, n_pca=1)
        self.assertEqual(loader.M.shape, (2, 50))
        with self.assertRaisesRegex(ValueError, 'wave_min and wave_max'):
            TelluricSpectra(self.pca, n_pca=1)


if __name__ == '__main__':
    unittest.main()
