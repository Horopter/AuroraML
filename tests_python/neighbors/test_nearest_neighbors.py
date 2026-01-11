#!/usr/bin/env python3
"""
Tests for NearestNeighbors
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))


class TestNearestNeighbors(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(80, 4).astype(np.float64)
        self.X_test = np.random.randn(12, 4).astype(np.float64)
        self.y_dummy = np.zeros(self.X.shape[0]).astype(np.float64)

    def test_kneighbors(self):
        import ingenuityml.neighbors as ing_neighbors

        model = ing_neighbors.NearestNeighbors(n_neighbors=3, radius=1.0)
        model.fit(self.X, self.y_dummy)
        distances, indices = model.kneighbors(self.X_test)

        self.assertEqual(distances.shape, (self.X_test.shape[0], 3))
        self.assertEqual(indices.shape, (self.X_test.shape[0], 3))
        self.assertTrue(np.all(distances >= 0))
        self.assertTrue(np.all(indices < self.X.shape[0]))

    def test_radius_neighbors(self):
        import ingenuityml.neighbors as ing_neighbors

        model = ing_neighbors.NearestNeighbors(n_neighbors=3, radius=1.0)
        model.fit(self.X, self.y_dummy)
        distances, indices = model.radius_neighbors(self.X_test, radius=1.0)

        self.assertEqual(len(distances), self.X_test.shape[0])
        self.assertEqual(len(indices), self.X_test.shape[0])
        for d_row, i_row in zip(distances, indices):
            self.assertEqual(len(d_row), len(i_row))
            for d in d_row:
                self.assertGreaterEqual(d, 0.0)

    def test_not_fitted(self):
        import ingenuityml.neighbors as ing_neighbors

        model = ing_neighbors.NearestNeighbors(n_neighbors=3, radius=1.0)
        with self.assertRaises((RuntimeError, ValueError)):
            model.kneighbors(self.X_test)
        with self.assertRaises((RuntimeError, ValueError)):
            model.radius_neighbors(self.X_test, radius=1.0)

    def test_wrong_feature_count(self):
        import ingenuityml.neighbors as ing_neighbors

        model = ing_neighbors.NearestNeighbors(n_neighbors=3, radius=1.0)
        model.fit(self.X, self.y_dummy)
        X_wrong = np.random.randn(10, 6).astype(np.float64)
        with self.assertRaises((RuntimeError, ValueError)):
            model.kneighbors(X_wrong)
        with self.assertRaises((RuntimeError, ValueError)):
            model.radius_neighbors(X_wrong, radius=1.0)


if __name__ == '__main__':
    unittest.main()
