"""Numerical contracts for NMF training and observation fitting; no external files."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.exceptions import ConvergenceWarning

from src.model import (SpectralNMF, TelluricNMF, NMFCache, ContinuumResponseBasis,
                       ContinuumNormalizer, CacheValidationError)
from src.load_spectra import BOSZSpectra, SpectrumID
from src.preparation import select_bosz, PreparedSpectra


class NMFTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.geomspace(3800, 4200, 50)
        rng = np.random.default_rng(42)
        self.X = (rng.uniform(0, 1, (24, 3)) @ rng.uniform(0, 1, (3, 50))).astype(np.float32)

    def test_full_fit_matches_previous_sklearn_configuration(self):
        expected = NMF(n_components=3, init='nndsvda', solver='cd', max_iter=30,
                       tol=0., random_state=4, shuffle=True).fit(self.X)
        actual = SpectralNMF(3, max_iter=30, tol=0., seed=4).fit(self.X, self.wave)
        np.testing.assert_array_equal(actual.components_, expected.components_)
        self.assertEqual(actual.diagnostics['n_iter'], expected.n_iter_)
        self.assertTrue(actual.diagnostics['iteration_limit_reached'])
        self.assertIsNone(actual.cache_path)
        self.assertTrue(np.isfinite(actual.diagnostics['rms_optical_depth']))

    def test_cd_residuals_use_training_weights_without_another_solve(self):
        for matrix in [self.X, sparse.csr_matrix(self.X)]:
            with self.subTest(sparse=sparse.issparse(matrix)):
                reference = NMF(n_components=3, init='nndsvda', solver='cd', max_iter=30,
                                tol=0., random_state=4, shuffle=True)
                weights = reference.fit_transform(matrix)
                model = SpectralNMF(3, max_iter=30, tol=0., seed=4,
                                    matrix_format='sparse' if sparse.issparse(matrix) else 'dense')
                # A diagnostic must not start an independent iterative weight solve.
                with patch.object(model, 'transform', side_effect=AssertionError('Unexpected weight solve')):
                    model.fit(matrix, self.wave, residual_samples=7)
                rows = np.sort(np.random.default_rng(4).choice(len(self.X), size=7, replace=False))
                residual = self.X[rows].astype(np.float64) - weights[rows].astype(np.float64) @ reference.components_.astype(np.float64)
                self.assertAlmostEqual(model.diagnostics['rms_optical_depth'], np.sqrt(np.mean(residual**2)))
                self.assertAlmostEqual(model.diagnostics['relative_frobenius_error'],
                                       np.linalg.norm(residual)/np.linalg.norm(self.X[rows].astype(np.float64)))
                self.assertEqual(model.diagnostics['residual_weight_source'], 'training_fit')
                self.assertTrue(model.diagnostics['iteration_limit_reached'])

    def test_minibatch_fit_matches_partial_fit_epochs(self):
        expected = MiniBatchNMF(n_components=3, init='nndsvda', forget_factor=1, random_state=4)
        rng = np.random.default_rng(4)
        for _ in range(2):
            order = rng.permutation(len(self.X))
            for start in range(0, len(order), 7):
                expected.partial_fit(self.X[np.sort(order[start:start+7])])
        actual = SpectralNMF(3, solver='minibatch', seed=4, max_epochs=2,
                             batch_size=7, check_interval=100, tol=0.).fit(self.X, self.wave)
        np.testing.assert_array_equal(actual.components_, expected.components_)
        self.assertEqual(actual.diagnostics['epochs'], 2)
        self.assertEqual(actual.diagnostics['batches_seen'], 8)

    def test_minibatch_stopping_and_checkpoint(self):
        with TemporaryDirectory() as directory:
            model = SpectralNMF(3, solver='minibatch', max_epochs=5, batch_size=6,
                                tol=1e6, tol_batches=2, check_interval=1).fit(self.X, self.wave, cache=directory)
            self.assertTrue(model.diagnostics['converged'])
            self.assertEqual(model.diagnostics['batches_seen'], 2)
            checkpoints = list(Path(directory).glob('nmf_F_last_epoch_*.json'))
            self.assertTrue(checkpoints)
            checkpoint = checkpoints[0].with_suffix('.npy')
            loaded_checkpoint = SpectralNMF.load(checkpoint, allow_checkpoint=True)
            self.assertEqual(loaded_checkpoint.metadata['artifact_kind'], 'checkpoint')
            with self.assertRaises(CacheValidationError): SpectralNMF.load(checkpoint)
            loaded = SpectralNMF.load(cache=directory, params=model.params)
            np.testing.assert_array_equal(loaded.components_, model.components_)

    def test_roundtrip_arbitrary_wave_and_shared_telluric_solver(self):
        with TemporaryDirectory() as directory:
            model = TelluricNMF(3, max_iter=20, tol=0.).fit(self.X, self.wave, cache=directory)
            loaded = TelluricNMF.load(model.cache_path)
            np.testing.assert_array_equal(loaded.wavelengths, self.wave)
            np.testing.assert_array_equal(loaded.transform(self.X), model.transform(self.X))
            with self.assertRaises(CacheValidationError):
                SpectralNMF.load(model.cache_path)

    def test_cache_identity_includes_values_selection_and_grid(self):
        model = SpectralNMF(3)
        first = model.parameters_for(self.X, self.wave)
        self.assertNotEqual(first, model.parameters_for(self.X+1, self.wave))
        self.assertNotEqual(first, model.parameters_for(self.X, self.wave+1))
        self.assertNotEqual(first, model.parameters_for(self.X, self.wave, selection=np.arange(12)))
        self.assertNotEqual(first, model.parameters_for(self.X, self.wave, source_ids=list(range(24))))

    def test_preprocessing_overrides_and_prepared_input_roundtrip(self):
        settings = dict(wavelength_unit='Angstrom', wavelength_convention='vacuum',
                        reference_frame='stellar_rest', R_out=115000)
        model = SpectralNMF(3, max_iter=10).fit(self.X, self.wave, preprocessing=settings)
        self.assertEqual(model.preprocessing, settings)
        with TemporaryDirectory() as directory:
            data = PreparedSpectra(self.X, self.wave, settings, {'count': len(self.X), 'ids': ['example']})
            fitted = SpectralNMF(3, max_iter=10).fit(data, cache=directory)
            loaded = SpectralNMF.load(cache=directory, params=fitted.params)
            self.assertEqual(loaded.preprocessing, settings)
            np.testing.assert_array_equal(loaded.components_, fitted.components_)

    def test_fixed_basis_weight_solves_do_not_retrain(self):
        model = SpectralNMF(3, max_iter=10).fit(self.X, self.wave)
        before = model.components_.copy()
        weights = model.transform(self.X)
        np.testing.assert_array_equal(before, model.components_)
        self.assertEqual(weights.shape, (24, 3))
        with self.assertRaises(ValueError):
            model.evaluate(self.wave-10000)

    def test_selection_is_external_and_does_not_mutate_loader(self):
        ids = [SpectrumID('mp','t6000','g+4.0','m+0.00','a+0.00','c+0.00','v1','rorig','noresam'),
               SpectrumID('ap','t9000','g+4.0','m+0.00','a+0.00','c+0.00','v1','rorig','noresam')]
        class Loader:
            def __len__(self): return len(self.ids)
        loader = Loader(); loader.ids = ids
        np.testing.assert_array_equal(select_bosz(loader), [0])
        self.assertEqual(loader.ids, ids)
        with self.assertRaises(ValueError):
            select_bosz(loader, teff_max=1000)

    def test_bosz_preparation_builds_only_selected_rows_and_reuses_matrix(self):
        from src.preparation import prepare_bosz
        X = self.X[:3]
        ids = [SpectrumID('mp',f't{5000+250*i}','g+4.0','m+0.00','a+0.00','c+0.00','v1','rorig','noresam') for i in range(3)]
        class Loader:
            def __len__(self): return len(self.ids)
            def _processing_params(self):
                return dict(D=len(self.wave_out), R_in=300000, R_out=115000,
                            wave_min=float(self.wave_out[0]), wave_max=float(self.wave_out[-1]))
            def optical_depth(self, rows, tol, floor, progress=None):
                self.calls.append(list(rows))
                if progress is not None:
                    progress(len(rows))
                return self.X[rows]
        loader = Loader(); loader.ids = ids; loader.X = X
        loader.wave_out = self.wave; loader.root = Path('/example/bosz'); loader.calls = []
        selection = np.array([2,0])
        with TemporaryDirectory() as directory:
            model = SpectralNMF(2,max_iter=10)
            expected_params = model.parameters_for(loader,selection=selection)
            self.assertEqual(loader.calls, [])
            fitted = model.fit(loader,selection=selection,matrix_cache=directory)
            self.assertEqual(fitted.params,expected_params)
            self.assertEqual(loader.calls, [[2,0]])
            prepared = prepare_bosz(loader,selection,matrix_cache=directory)
            self.assertEqual(loader.calls, [[2,0]])
            self.assertFalse(sparse.issparse(prepared.optical_depth))
            self.assertEqual(prepared.optical_depth.dtype, np.float32)
            np.testing.assert_array_equal(prepared.optical_depth,X[selection])
            self.assertEqual(prepared.source_manifest['count'],2)

    def test_matrix_format_controls_solver_input_and_survives_loading(self):
        for trainer in [SpectralNMF, TelluricNMF]:
            for solver in ['cd', 'minibatch']:
                for matrix_format in ['dense', 'sparse']:
                    with self.subTest(trainer=trainer.__name__, solver=solver, matrix_format=matrix_format), TemporaryDirectory() as directory:
                        # Supply the opposite format to exercise conversion for prepared input too.
                        source = sparse.csr_matrix(self.X) if matrix_format == 'dense' else self.X
                        cls, method = (NMF, 'fit_transform') if solver == 'cd' else (MiniBatchNMF, 'partial_fit')
                        original = getattr(cls, method)
                        observed = []

                        def checked_fit(estimator, matrix, *args, **kwargs):
                            observed.append(sparse.issparse(matrix))
                            return original(estimator, matrix, *args, **kwargs)

                        model = trainer(3, solver=solver, matrix_format=matrix_format,
                                        max_iter=10, tol=0., max_epochs=1, batch_size=6,
                                        check_interval=1000)
                        with patch.object(cls, method, new=checked_fit):
                            model.fit(source, self.wave, cache=directory)
                        self.assertTrue(observed)
                        self.assertEqual(set(observed), {matrix_format == 'sparse'})
                        loaded = trainer.load(cache=directory, params=model.params)
                        self.assertEqual(loaded.config['matrix_format'], matrix_format)
                        np.testing.assert_array_equal(loaded.components_, model.components_)
                        np.testing.assert_array_equal(loaded.transform(source), model.transform(source))
        with self.assertRaises(ValueError):
            SpectralNMF(matrix_format='csr')

    def test_invalid_and_unfitted_inputs(self):
        with self.assertRaises(ValueError): SpectralNMF(1.5)
        self.assertEqual(SpectralNMF().config['tol'], 0.)
        self.assertEqual(SpectralNMF(solver='minibatch').config['tol'], 1e-4)
        model = SpectralNMF(3)
        self.assertIsNone(model.components_)
        with self.assertRaises(ValueError): model.transform(self.X)
        with self.assertRaises(ValueError): model.fit(-self.X, self.wave)
        with self.assertRaises(ValueError): model.fit(self.X, self.wave[::-1])
        with self.assertRaises(ValueError): model.fit(self.X, self.wave, selection=np.array([1,1]))
        with self.assertRaises(ValueError): SpectralNMF(25).fit(self.X, self.wave)


class _DebugBOSZ:
    """Small input with excluded ATLAS rows before the eligible MARCS spectra."""
    def __init__(self, limit=2):
        self.root = Path('/example/bosz')
        self.debug_max_F_spectra = limit
        self.ids = [SpectrumID(atmos, f't{teff}', 'g+4.0', 'm+0.00', 'a+0.00',
                               'c+0.00', 'v1', 'rorig', 'noresam')
                    for atmos, teff in [('ap', 9000), ('ap', 8500), ('mp', 5000),
                                        ('ms', 5250), ('mp', 5500), ('ms', 5750)]]
        self.wave_out = np.linspace(3800., 4200., 4)
        self.tau = np.random.default_rng(47).uniform(.1, 1., (6, 4)).astype(np.float32)
        self.read_rows = []

    def __len__(self):
        return len(self.ids)

    def _processing_params(self):
        return dict(D=4, R_in=300000, R_out=115000, wave_min=3800., wave_max=4200.)

    def optical_depth(self, rows, tol, floor, progress=None):
        self.read_rows.extend(rows)
        if progress is not None:
            progress(len(rows))
        return self.tau[rows]


class BOSZPreparationTests(unittest.TestCase):
    def test_debug_cap_is_applied_after_filtering_without_mutating_ids(self):
        loader = _DebugBOSZ()
        original_ids = loader.ids.copy()
        np.testing.assert_array_equal(select_bosz(loader), [2, 3])
        np.testing.assert_array_equal(select_bosz(loader, indices=np.array([1, 5, 4, 3, 2])), [5, 4])
        self.assertEqual(loader.ids, original_ids)
        self.assertEqual(len(loader), 6)
        loader.debug_max_F_spectra = None
        np.testing.assert_array_equal(select_bosz(loader), [2, 3, 4, 5])

    def test_debug_cap_limits_fitting_and_parameter_identity_for_all_entry_paths(self):
        cases = [(None, [0, 1]), (np.array([5, 4, 3, 2]), [5, 4]),
                 (np.array([False, False, True, True, True, True]), [2, 3])]
        for selection, expected in cases:
            with self.subTest(selection=selection), TemporaryDirectory() as directory:
                loader = _DebugBOSZ()
                model = SpectralNMF(1, max_iter=5)
                params = model.parameters_for(loader, selection=selection)
                self.assertEqual(params['source_manifest']['count'], 2)
                self.assertEqual(loader.read_rows, [])
                model.fit(loader, selection=selection, cache=directory)
                self.assertEqual(loader.read_rows, expected)
                self.assertEqual(model.params, params)
                self.assertEqual(model.diagnostics['residual_samples'], 2)
                loaded = SpectralNMF.load(cache=directory, params=params)
                np.testing.assert_array_equal(loaded.components_, model.components_)

    def test_changed_debug_cap_does_not_reuse_an_uncapped_matrix(self):
        from src.preparation import prepare_bosz
        loader = _DebugBOSZ(limit=None)
        selection = np.array([2, 3, 4, 5])
        with TemporaryDirectory() as directory:
            uncapped = prepare_bosz(loader, selection, matrix_cache=directory)
            self.assertEqual(uncapped.optical_depth.shape[0], 4)
            loader.debug_max_F_spectra = 2
            loader.read_rows.clear()
            capped = prepare_bosz(loader, selection, matrix_cache=directory)
            self.assertEqual(loader.read_rows, [2, 3])
            self.assertEqual(capped.optical_depth.shape[0], 2)
            self.assertNotEqual(capped.source_manifest, uncapped.source_manifest)
            loader.read_rows.clear()
            cached = prepare_bosz(loader, selection, matrix_cache=directory)
            self.assertEqual(loader.read_rows, [])
            np.testing.assert_array_equal(cached.optical_depth, loader.tau[[2, 3]])
            loader.debug_max_F_spectra = None
            cached = prepare_bosz(loader, selection, matrix_cache=directory)
            self.assertEqual(loader.read_rows, [])
            np.testing.assert_array_equal(cached.optical_depth, loader.tau[selection])

    def test_dense_batches_keep_selection_order_without_sparse_conversion(self):
        from src.preparation import prepare_bosz
        loader = _DebugBOSZ(limit=None)
        loader.sparse = Mock(side_effect=AssertionError('Dense preparation must not build CSR'))
        loader.optical_depth = Mock(wraps=loader.optical_depth)
        selection = np.array([5, 1, 4, 0, 3])
        prepared = prepare_bosz(loader, selection, chunk_size=2, threshold=0.)
        self.assertIsInstance(prepared.optical_depth, np.ndarray)
        self.assertEqual(prepared.optical_depth.dtype, np.float32)
        self.assertTrue(prepared.optical_depth.flags.c_contiguous)
        self.assertEqual(loader.read_rows, selection.tolist())
        self.assertEqual([len(call.args[0]) for call in loader.optical_depth.call_args_list], [2, 2, 1])
        np.testing.assert_array_equal(prepared.optical_depth, loader.tau[selection])
        loader.sparse.assert_not_called()

    def test_sparse_dense_toggle_preserves_values_and_keeps_caches_separate(self):
        from src.preparation import prepare_bosz
        loader = _DebugBOSZ(limit=None)
        loader.tau[2, 0] = 0.
        selection = np.array([5, 2, 4])
        with TemporaryDirectory() as directory:
            results = {}
            for matrix_format in ['dense', 'sparse']:
                loader.read_rows.clear()
                prepared = prepare_bosz(loader, selection, threshold=0., chunk_size=2,
                                        matrix_cache=directory, matrix_format=matrix_format)
                self.assertEqual(loader.read_rows, selection.tolist())
                matrix = prepared.optical_depth
                self.assertEqual(sparse.issparse(matrix), matrix_format == 'sparse')
                self.assertEqual(matrix.dtype, np.float32)
                np.testing.assert_array_equal(matrix.toarray() if sparse.issparse(matrix) else matrix,
                                              loader.tau[selection])
                model = SpectralNMF(2, max_iter=5, tol=0., sparse_tol=0., matrix_format=matrix_format)
                params = model.parameters_for(loader, selection=selection)
                self.assertEqual(params['preprocessing'], prepared.preprocessing)
                results[matrix_format] = params
                loader.read_rows.clear()
                cached = prepare_bosz(loader, selection, threshold=0., matrix_cache=directory,
                                      matrix_format=matrix_format)
                self.assertEqual(loader.read_rows, [])
                self.assertEqual(sparse.issparse(cached.optical_depth), matrix_format == 'sparse')
                # Dense cache hits are read-only memory maps: fitting must accept them.
                model.fit(loader, selection=selection, matrix_cache=directory)
                self.assertEqual(loader.read_rows, [])
            self.assertNotEqual(results['dense'], results['sparse'])
        with self.assertRaises(ValueError):
            prepare_bosz(loader, selection, matrix_format='csr')

    def test_progress_advances_after_each_uncached_spectrum(self):
        loader = BOSZSpectra.__new__(BOSZSpectra)
        loader.ids = ['a', 'b', 'c']
        loader.wave_out = np.arange(4, dtype=float)
        fluxes = {'a': np.array([1., .8, .6, .4]), 'b': np.array([.9, .7, .5, .3]),
                  'c': np.array([.8, .6, .4, .2])}
        loader._read_cached_batch = Mock(return_value=None)
        loader.process_spectrum = Mock(side_effect=lambda identifier: fluxes[identifier])
        events = []
        tau = loader.sparse([2, 0, 1], progress=lambda n: events.append((n, loader.process_spectrum.call_count)))
        self.assertEqual(events, [(1, 1), (1, 2), (1, 3)])
        np.testing.assert_array_equal(tau.toarray(), -np.log(np.stack([fluxes['c'], fluxes['a'], fluxes['b']])))
        # Ordinary indexing uses the same values without requiring a progress callback.
        np.testing.assert_array_equal(loader[[2, 0]], np.stack([fluxes['c'], fluxes['a']]))

    def test_progress_preserves_bulk_cached_reads(self):
        loader = BOSZSpectra.__new__(BOSZSpectra)
        loader.ids = ['a', 'b']
        loader.wave_out = np.arange(3, dtype=float)
        flux = np.array([[1., .5, .25], [.8, .4, .2]])
        loader._read_cached_batch = Mock(return_value=flux)
        loader.process_spectrum = Mock()
        updates = []
        tau = loader.sparse([0, 1], progress=updates.append)
        self.assertEqual(updates, [2])
        loader.process_spectrum.assert_not_called()
        loader._read_cached_batch.assert_called_once_with(['a', 'b'])
        np.testing.assert_array_equal(tau.toarray(), -np.log(flux))


class ContinuumTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(3800, 4200, 400)
        self.basis = np.stack([np.exp(-.5*((self.wave-3920)/6)**2),
                               .7*np.exp(-.5*((self.wave-4110)/15)**2)])
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.cache = NMFCache(self.temp.name)
        path = self.cache.save('nmf_F_full', self.basis, self.wave, params={}, role='stellar')
        self.stellar = SpectralNMF.load(path)

    def test_known_coefficients_and_reconstruction(self):
        response = ContinuumResponseBasis(3)
        H = response.design_matrix(self.wave).toarray()
        alpha = np.array([.4, .8]); gamma=np.array([.1,.02,-.03])
        continuum = np.exp(H@gamma)
        flux = np.exp(-alpha@self.basis)*continuum
        result = ContinuumNormalizer(self.stellar, response_basis=response).fit(self.wave, flux, flux*.01, lsq_solver='exact')
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.alpha, alpha, atol=1e-6)
        np.testing.assert_allclose(result.gamma.ravel(), gamma, atol=1e-6)
        np.testing.assert_allclose(result.continuum, continuum, rtol=1e-7)
        np.testing.assert_allclose(result.model_flux, flux, rtol=1e-7)
        self.assertEqual(result.beta.shape, (0,))

    def test_orders_mask_bad_pixels_and_custom_basis_keep_full_domains(self):
        wave=[self.wave[:150],self.wave[200:]]
        calls=[]
        def polynomial(w, n):
            calls.append(w.copy())
            x=(w-w.min())/np.ptp(w)
            return np.column_stack([np.ones(len(w)),x])
        response=ContinuumResponseBasis(2,basis_function=polynomial)
        true=[np.exp(-np.array([.2,.3])@self.stellar.evaluate(w)+.1+.02*(w-w.min())/np.ptp(w)) for w in wave]
        errors=[f*.01 for f in true]
        mask=[np.ones(len(w),bool) for w in wave]; mask[0][0]=False; mask[1][-1]=False
        observed=[f.copy() for f in true]; observed[0][4]=np.nan; errors[1][5]=0
        result=ContinuumNormalizer(self.stellar,response_basis=response).fit(wave,observed,errors,mask=mask)
        self.assertTrue(result.success)
        self.assertIsInstance(result.model_flux,tuple)
        self.assertEqual(result.gamma.shape,(2,2))
        self.assertFalse(result.mask[0][0]); self.assertFalse(result.mask[0][4]); self.assertFalse(result.mask[1][5])
        self.assertTrue(np.isnan(result.model_flux[0][0]))
        np.testing.assert_array_equal(calls[0],wave[0])
        np.testing.assert_array_equal(calls[1],wave[1])
        np.testing.assert_allclose(result.alpha,[.2,.3],atol=1e-6)

    def test_different_telluric_grid_and_fixed_radial_velocity(self):
        wave=np.linspace(3790,4210,500)
        tell_basis=np.exp(-.5*((wave-4020)/2)**2)[None,:]
        path=self.cache.save('nmf_G',tell_basis,wave,params={},role='telluric')
        telluric=TelluricNMF.load(path)
        observed_wave=self.wave[2:-2]
        rv=20.
        F=self.stellar.evaluate(observed_wave/(1+rv/299792.458))
        G=telluric.evaluate(observed_wave)
        flux=np.exp(-np.array([.2,.5])@F-.3*G[0]+.1)
        result=ContinuumNormalizer(self.stellar,telluric,ContinuumResponseBasis(1)).fit(observed_wave,flux,flux*.01,rv=rv)
        np.testing.assert_allclose(result.alpha,[.2,.5],atol=1e-6)
        np.testing.assert_allclose(result.beta,[.3],atol=1e-6)
        np.testing.assert_allclose(result.model_flux,flux,rtol=1e-6)

    def test_second_order_weighting_explicit_and_legacy_adapter(self):
        flux=np.exp(-self.basis[0]+.1)
        model=ContinuumNormalizer(self.stellar,response_basis=ContinuumResponseBasis(1),variance_order=2)
        result=model.fit(self.wave,flux,flux*.1)
        self.assertTrue(result.success)
        with self.assertRaises(ValueError): model.fit(self.wave,flux,np.zeros_like(flux))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            adapter=ContinuumNormalizer.from_cache(self.stellar.cache_path,flux_errors=flux*.01,order_wavelengths=self.wave,n_modes=1)
        adapted=adapter.fit(flux)
        np.testing.assert_allclose(adapted.alpha,[1,0],atol=1e-5)
        alpha,beta,gamma=adapted
        self.assertEqual(gamma.shape,(1,1))

    def test_nonconvergence_is_exposed_and_solver_options_are_checked(self):
        model = ContinuumNormalizer(self.stellar, response_basis=ContinuumResponseBasis(1))
        # An emission feature cannot be fit with non-negative absorption weights.
        flux = np.exp(.3*self.basis[0] + .1)
        result = model.fit(self.wave, flux, flux*.01, lsq_solver='exact', max_iter=1)
        self.assertFalse(result.success)
        self.assertEqual(result.status, 0)
        self.assertTrue(np.isfinite(result.model_flux).all())
        with self.assertRaises(ValueError):
            model.fit(self.wave, flux, flux*.01, lsq_solver='unknown')

    def test_masked_nonfinite_wavelength_and_rectangular_shapes(self):
        wave=np.stack([self.wave,self.wave]); wave[0,0]=np.nan
        flux=np.stack([np.exp(-self.basis[0]+.1)]*2)
        result=ContinuumNormalizer(self.stellar,response_basis=ContinuumResponseBasis(1)).fit(wave,flux,flux*.01)
        self.assertEqual(result.model_flux.shape,wave.shape)
        self.assertFalse(result.mask[0,0])
        self.assertTrue(result.success)


if __name__ == '__main__':
    unittest.main()
