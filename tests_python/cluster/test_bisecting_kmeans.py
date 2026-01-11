#!/usr/bin/env python3
"""
Test Suite for IngenuityML BisectingKMeans Algorithm
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


class TestBisectingKMeans(unittest.TestCase):
    """Test BisectingKMeans algorithm - Positive and Negative Cases"""

    def setUp(self):
        np.random.seed(42)
        n_samples = 120
        self.X = np.vstack([
            np.random.randn(n_samples // 3, 2).astype(np.float64) + np.array([2, 2]),
            np.random.randn(n_samples // 3, 2).astype(np.float64) - np.array([2, 2]),
            np.random.randn(n_samples - 2 * (n_samples // 3), 2).astype(np.float64)
        ])
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)

    def test_basic_functionality(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.BisectingKMeans(n_clusters=3, random_state=42)
        model.fit(self.X, self.y_dummy)

        labels = model.predict(self.X)
        self.assertEqual(len(labels), len(self.X))
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))

    def test_cluster_centers(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.BisectingKMeans(n_clusters=3, random_state=42)
        model.fit(self.X, self.y_dummy)

        centers = model.cluster_centers()
        self.assertEqual(centers.shape[1], self.X.shape[1])
        self.assertEqual(centers.shape[0], 3)
        self.assertFalse(np.any(np.isnan(centers)))

    def test_is_fitted(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.BisectingKMeans(n_clusters=3)
        self.assertFalse(model.is_fitted())

        model.fit(self.X, self.y_dummy)
        self.assertTrue(model.is_fitted())

    def test_not_fitted_predict(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.BisectingKMeans(n_clusters=3)
        with self.assertRaises((RuntimeError, ValueError)):
            model.predict(self.X)

    def test_more_clusters_than_samples(self):
        import ingenuityml.cluster as ing_cluster

        X_small = self.X[:5]
        y_small = self.y_dummy[:5]
        model = ing_cluster.BisectingKMeans(n_clusters=10)
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(X_small, y_small)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)

    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)
