#!/usr/bin/env python3
"""
Test Suite for IngenuityML AffinityPropagation Algorithm
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


class TestAffinityPropagation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.vstack([
            np.random.randn(30, 2).astype(np.float64) + np.array([2, 2]),
            np.random.randn(30, 2).astype(np.float64) - np.array([2, 2]),
            np.random.randn(30, 2).astype(np.float64)
        ])

    def test_basic_functionality(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.AffinityPropagation(random_state=42)
        model.fit(self.X, None)

        labels = model.labels()
        self.assertEqual(len(labels), len(self.X))
        self.assertTrue(np.all(labels >= 0))

        centers = model.cluster_centers()
        n_clusters = len(np.unique(labels))
        self.assertEqual(centers.shape[0], n_clusters)
        self.assertEqual(centers.shape[1], self.X.shape[1])

    def test_predict(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.AffinityPropagation(random_state=42)
        model.fit(self.X, None)

        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_is_fitted(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.AffinityPropagation()
        self.assertFalse(model.is_fitted())

        model.fit(self.X, None)
        self.assertTrue(model.is_fitted())

    def test_empty_data(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.AffinityPropagation()
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(np.array([]).reshape(0, 2), None)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)

    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)
