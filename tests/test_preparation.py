"""Parallel BOSZ preparation with real temporary spectra and cache files."""
import gzip
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scipy import sparse
from threadpoolctl import threadpool_info

from src.load_spectra import BOSZSpectra
from src.preparation import prepare_bosz


class _TrackedBOSZ(BOSZSpectra):
    def _compute_spectrum(self, identifier):
        if os.getpid() != self.owner_pid:
            for pool in threadpool_info():
                if pool['num_threads'] != 1:
                    raise AssertionError('Preprocessing workers must use one numerical thread')
        (self.computation_log / identifier.teff).write_text(str(os.getpid()))
        return super()._compute_spectrum(identifier)

    def _add_pending(self, identifier, spectrum):
        if os.getpid() != self.owner_pid:
            raise AssertionError('Only the parent may write the cache')
        return super()._add_pending(identifier, spectrum)


class ParallelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.root = self.directory / 'raw'
        self.root.mkdir()
        wave = np.geomspace(3790., 4210., 2000)
        for i in range(7):
            name = f'bosz2024_mp_t{5000+250*i}_g+4.0_m+0.00_a+0.00_c+0.00_v1_rorig_noresam.txt.gz'
            flux = 1 - (.1 + .05*i)*np.exp(-.5*((wave - 4000.)/2.)**2)
            with gzip.open(self.root / name, 'wt') as handle:
                np.savetxt(handle, np.column_stack([wave, flux, np.ones_like(wave)]))

    def loader(self, cache, limit=3):
        loader = _TrackedBOSZ(self.root, R_in=300000, R_out=115000, D=120,
                              wave_min=3800., wave_max=4200.,
                              cache_dir=self.directory / cache, spectra_per_file=2,
                              debug_max_F_spectra=limit)
        # File discovery order is unspecified; fix it for the selection assertions.
        loader.ids.sort(key=lambda identifier: identifier.teff)
        loader.owner_pid = os.getpid()
        loader.computation_log = self.directory / f'{cache}-computations'
        loader.computation_log.mkdir(exist_ok=True)
        self.addCleanup(loader.db.close)
        self.addCleanup(loader.flush_cache)
        return loader

    def test_parallel_matches_serial_and_reuses_selected_cache(self):
        selection = np.array([6, 2, 4, 1])
        serial = self.loader('serial')
        expected = prepare_bosz(serial, selection, threshold=0.)
        serial.flush_cache()

        parallel = self.loader('parallel')
        # Already processed rows should be reused; the other selected rows run in children.
        parallel.process_spectrum(parallel.ids[6])
        actual = prepare_bosz(parallel, selection, threshold=0., n_workers=2)
        self.assertFalse(sparse.issparse(actual.optical_depth))
        self.assertEqual(actual.optical_depth.dtype, np.float32)
        np.testing.assert_array_equal(actual.optical_depth, expected.optical_depth)
        self.assertEqual(actual.source_manifest, expected.source_manifest)
        self.assertEqual(actual.preprocessing, expected.preprocessing)
        selected_ids = {parallel.ids[i] for i in [6, 2, 4]}
        self.assertEqual(parallel._cached_ids(), selected_ids)
        logs = {path.name: path.read_text() for path in parallel.computation_log.iterdir()}
        self.assertEqual(set(logs), {identifier.teff for identifier in selected_ids})
        self.assertEqual(int(logs[parallel.ids[6].teff]), os.getpid())
        for row in [2, 4]:
            self.assertNotEqual(int(logs[parallel.ids[row].teff]), os.getpid())

        # Verify persistent cache reads with a new SQLite connection and empty pending data.
        reloaded = self.loader('parallel')
        cached = prepare_bosz(reloaded, selection, threshold=0., n_workers=2)
        np.testing.assert_array_equal(cached.optical_depth, expected.optical_depth)
        self.assertEqual(logs, {path.name: path.read_text() for path in reloaded.computation_log.iterdir()})

    def test_bulk_processing_still_obeys_debug_limit(self):
        loader = self.loader('bulk', limit=2)
        loader.process_all_spectra(n_workers=2)
        self.assertEqual(loader._cached_ids(), set(loader.ids[:2]))
        self.assertEqual(loader._pending, {})


if __name__ == '__main__':
    unittest.main()
