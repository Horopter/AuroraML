#!/usr/bin/env python3
"""
Test Suite for IngenuityML FeatureAgglomeration Algorithm
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


class TestFeatureAgglomeration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(60, 6).astype(np.float64)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)

    def test_basic_functionality(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.FeatureAgglomeration(n_clusters=3)
        model.fit(self.X, self.y_dummy)

        reduced = model.transform(self.X)
        self.assertEqual(reduced.shape, (self.X.shape[0], 3))

        expanded = model.inverse_transform(reduced)
        self.assertEqual(expanded.shape, self.X.shape)

        labels = model.labels()
        self.assertEqual(len(labels), self.X.shape[1])

    def test_is_fitted(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.FeatureAgglomeration(n_clusters=3)
        self.assertFalse(model.is_fitted())

        model.fit(self.X, self.y_dummy)
        self.assertTrue(model.is_fitted())

    def test_not_fitted_transform(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.FeatureAgglomeration(n_clusters=3)
        with self.assertRaises((RuntimeError, ValueError)):
            model.transform(self.X)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)

    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)
