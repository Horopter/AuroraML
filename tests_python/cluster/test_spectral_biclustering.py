#!/usr/bin/env python3
"""
Test Suite for IngenuityML SpectralBiclustering Algorithm
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


class TestSpectralBiclustering(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        block_a = np.ones((10, 10), dtype=np.float64) * 5.0
        block_b = np.ones((10, 10), dtype=np.float64) * 0.5
        top = np.hstack([block_a, block_b])
        bottom = np.hstack([block_b, block_a])
        self.X = np.vstack([top, bottom])

    def test_basic_functionality(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.SpectralBiclustering(n_clusters=2, random_state=42)
        model.fit(self.X, None)

        row_labels = model.row_labels()
        col_labels = model.column_labels()
        self.assertEqual(len(row_labels), self.X.shape[0])
        self.assertEqual(len(col_labels), self.X.shape[1])
        self.assertTrue(np.all(row_labels >= 0))
        self.assertTrue(np.all(col_labels >= 0))

    def test_is_fitted(self):
        import ingenuityml.cluster as ing_cluster

        model = ing_cluster.SpectralBiclustering(n_clusters=2, random_state=42)
        self.assertFalse(model.is_fitted())

        model.fit(self.X, None)
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
