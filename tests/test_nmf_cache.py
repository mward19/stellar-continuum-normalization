"""NMF artifact compatibility, integrity, and job naming without spectral data."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src.nmf_cache import NMFCache, CacheMiss, CacheValidationError
from src.model import SpectralNMF, TelluricNMF


class NMFCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = NMFCache(self.root)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('SLURM_JOB_ID', None)
        self.params = {'K': 2, 'bosz': {'D': 3, 'wave_min': 3800., 'wave_max': 4200.}}
        self.array = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.wave = np.array([3800., 3901., 4200.])  # Deliberately not geomspace.

    def save(self, name='nmf_F_full', params=None, array=None, **kwargs):
        return self.cache.save(name, self.array if array is None else array, self.wave,
                params=self.params if params is None else params, role='stellar', **kwargs)

    def test_job_filename_explicit_wave_and_roundtrip(self):
        os.environ['SLURM_JOB_ID'] = '15262808'
        path = self.save()
        self.assertEqual(path.name, 'nmf_F_full_15262808.npy')
        artifact = self.cache.load(path)
        np.testing.assert_array_equal(artifact.basis, self.array)
        np.testing.assert_array_equal(artifact.wavelengths, self.wave)
        self.assertTrue(path.with_suffix('.wavelengths.npy').is_file())
        self.assertEqual(artifact.metadata['provenance']['slurm_job_id'], '15262808')
        self.assertEqual(artifact.metadata['schema_version'], 2)

    def test_hash_fallback_and_cross_job_lookup(self):
        path = self.save()
        key = hashlib.sha1(json.dumps(self.params, sort_keys=True).encode()).hexdigest()[:10]
        self.assertEqual(path.name, f'nmf_F_full_{key}.npy')
        for job in ['123', '456', '']:
            os.environ['SLURM_JOB_ID'] = job
            artifact = self.cache.load(name='nmf_F_full', params=self.params, role='stellar')
            self.assertEqual(artifact.path, path)
        self.assertEqual(len(list(self.root.glob('nmf_F_full_*.json'))), 1)

    def test_different_settings_same_job_and_checkpoint_updates(self):
        os.environ['SLURM_JOB_ID'] = '123'
        first = self.save()
        second = self.save(params={**self.params, 'K': 9})
        self.assertEqual(first.name, 'nmf_F_full_123.npy')
        self.assertEqual(second.name, 'nmf_F_full_123_2.npy')
        checkpoint = self.save(name='nmf_F_last_epoch', artifact_kind='checkpoint')
        again = self.save(name='nmf_F_last_epoch', artifact_kind='checkpoint', array=self.array+1)
        self.assertEqual(checkpoint, again)
        with self.assertRaises(CacheValidationError):
            self.cache.load(checkpoint)
        np.testing.assert_array_equal(self.cache.load(checkpoint, allow_checkpoint=True).basis, self.array+1)
        with self.assertRaises(CacheMiss):
            self.cache.load(name='nmf_F', params=self.params)

    def test_wrong_params_and_role_rejected(self):
        path = self.save()
        with self.assertRaises(CacheValidationError):
            self.cache.load(path, params={**self.params, 'K': 9})
        with self.assertRaises(CacheValidationError):
            TelluricNMF.load(path)
        with self.assertRaises(CacheMiss):
            self.cache.load(name='nmf_F_full', params={**self.params, 'K': 9})

    def test_corrupt_missing_and_replaced_arrays_rejected(self):
        path = self.save()
        np.save(path, self.array + 1)
        with self.assertRaisesRegex(CacheValidationError, 'Checksum'):
            self.cache.load(path)
        path = self.save()
        path.with_suffix('.wavelengths.npy').unlink()
        with self.assertRaises(CacheValidationError):
            self.cache.load(path)
        path = self.save()
        path.with_suffix('.json').write_text('{invalid')
        with self.assertRaises(CacheValidationError):
            self.cache.load(path)

    def test_unknown_schema_and_missing_params_rejected(self):
        path = self.save()
        meta_path = path.with_suffix('.json')
        metadata = json.loads(meta_path.read_text())
        metadata['schema_version'] = 999
        meta_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(CacheValidationError, 'schema'):
            self.cache.load(path)
        del metadata['params']
        meta_path.write_text(json.dumps(metadata))
        with self.assertRaises(CacheValidationError):
            self.cache.load(path)

    def test_invalid_basis_or_grid_cannot_be_saved(self):
        for array, wave in [(self.array-1, self.wave), (self.array, self.wave[::-1]),
                             (self.array, self.wave[:2]), (self.array*np.nan, self.wave)]:
            with self.assertRaises(CacheValidationError):
                self.cache.save('bad', array, wave, params={}, role='stellar')
        self.assertFalse((self.root/'bad.npy').exists())

    def test_legacy_grid_and_role_adapter(self):
        key = hashlib.sha1(json.dumps(self.params, sort_keys=True).encode()).hexdigest()[:10]
        path = self.root / f'nmf_F_full_{key}.npy'
        np.save(path, self.array)
        path.with_suffix('.json').write_text(json.dumps(self.params))
        loaded = SpectralNMF.load(cache=self.cache, params=self.params)
        np.testing.assert_array_equal(loaded.wavelengths, np.geomspace(3800, 4200, 3))
        self.assertTrue(loaded.metadata['provenance']['wavelengths_inferred'])
        self.assertEqual(loaded.n_components, 2)

    def test_sparse_matrix_cache_ignores_job_and_checks_selection(self):
        matrix = sparse.csr_matrix(self.array)
        params = {'selection': [1, 4], 'threshold': .01}
        path = self.cache.save_matrix(matrix, params)
        os.environ['SLURM_JOB_ID'] = '123'
        np.testing.assert_array_equal(self.cache.load_matrix(params).toarray(), self.array)
        with self.assertRaises(CacheMiss):
            self.cache.load_matrix({**params, 'selection': [2, 4]})
        self.assertNotIn('123', path.name)

    def test_dense_matrix_cache_is_mapped_and_reused_across_jobs(self):
        matrix = self.array.astype(np.float32)
        params = {'selection': [1, 4], 'threshold': 0.}
        os.environ['SLURM_JOB_ID'] = '123'
        path = self.cache.save_matrix(matrix, params)
        self.assertEqual(path.suffix, '.npy')
        os.environ['SLURM_JOB_ID'] = '456'
        loaded = self.cache.load_matrix(params, storage='dense')
        self.assertIsInstance(loaded, np.memmap)
        self.assertFalse(loaded.flags.writeable)
        self.assertEqual(loaded.dtype, np.float32)
        np.testing.assert_array_equal(loaded, matrix)
        self.assertEqual(self.cache.save_matrix(matrix, params), path)
        with self.assertRaises(CacheMiss):
            self.cache.load_matrix({**params, 'selection': [2, 4]}, storage='dense')

    def test_dense_and_sparse_matrix_caches_are_separate(self):
        params = {'selection': [1, 4]}
        self.cache.save_matrix(sparse.csr_matrix(self.array), params)
        with self.assertRaises(CacheMiss):
            self.cache.load_matrix(params, storage='dense')
        self.cache.save_matrix(self.array, params)
        self.assertIsInstance(self.cache.load_matrix(params), np.ndarray)
        self.assertTrue(sparse.issparse(self.cache.load_matrix(params, storage='sparse')))

    def test_dense_matrix_cache_rejects_corruption(self):
        params = {'selection': [1, 4]}
        path = self.cache.save_matrix(self.array, params)
        np.save(path, self.array + 1)
        with self.assertRaises(CacheValidationError):
            self.cache.load_matrix(params, storage='dense')

    def test_auto_load_never_trains_or_creates_cache(self):
        cache = NMFCache(self.root/'missing')
        with self.assertRaises(CacheMiss):
            SpectralNMF.load(cache=cache, params={'training': {'solver': 'cd'}})
        self.assertFalse(cache.directory.exists())
        with self.assertRaises(ValueError):
            SpectralNMF.load()


if __name__ == '__main__':
    unittest.main()
