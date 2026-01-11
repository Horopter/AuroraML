#!/usr/bin/env python3
"""
Test Suite for IngenuityML Manifold Learning Algorithms
Includes positive and negative test cases
All tests run in shuffled order with 5-minute timeout
"""

import sys
import os
import unittest
import random
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))


class TestManifoldAlgorithms(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(40, 5).astype(np.float64)

    def test_mds(self):
        import ingenuityml.manifold as ing_module

        model = ing_module.MDS(n_components=2)
        embedding = model.fit_transform(self.X, np.array([]))
        self.assertEqual(embedding.shape, (self.X.shape[0], 2))
        self.assertTrue(model.is_fitted())

        with self.assertRaises(RuntimeError):
            model.inverse_transform(embedding)

    def test_isomap(self):
        import ingenuityml.manifold as ing_module

        model = ing_module.Isomap(n_components=2, n_neighbors=5)
        embedding = model.fit_transform(self.X, np.array([]))
        self.assertEqual(embedding.shape, (self.X.shape[0], 2))
        self.assertTrue(model.is_fitted())

    def test_lle(self):
        import ingenuityml.manifold as ing_module

        model = ing_module.LocallyLinearEmbedding(n_components=2, n_neighbors=5)
        embedding = model.fit_transform(self.X, np.array([]))
        self.assertEqual(embedding.shape, (self.X.shape[0], 2))
        self.assertTrue(model.is_fitted())

    def test_spectral_embedding(self):
        import ingenuityml.manifold as ing_module

        model = ing_module.SpectralEmbedding(n_components=2, n_neighbors=5)
        embedding = model.fit_transform(self.X, np.array([]))
        self.assertEqual(embedding.shape, (self.X.shape[0], 2))
        self.assertTrue(model.is_fitted())


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)

    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)
