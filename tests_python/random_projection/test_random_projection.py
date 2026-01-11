#!/usr/bin/env python3
"""
Random projection tests.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))


class TestRandomProjection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(60, 10).astype(np.float64)

    def test_gaussian_random_projection(self):
        import ingenuityml.random_projection as ing_rp

        model = ing_rp.GaussianRandomProjection(n_components=4, random_state=0)
        model.fit(self.X, np.zeros(self.X.shape[0]))
        X_proj = model.transform(self.X)
        self.assertEqual(X_proj.shape, (self.X.shape[0], 4))
        X_inv = model.inverse_transform(X_proj)
        self.assertEqual(X_inv.shape, self.X.shape)

    def test_sparse_random_projection(self):
        import ingenuityml.random_projection as ing_rp

        model = ing_rp.SparseRandomProjection(n_components=4, density=0.5, random_state=0)
        model.fit(self.X, np.zeros(self.X.shape[0]))
        X_proj = model.transform(self.X)
        self.assertEqual(X_proj.shape, (self.X.shape[0], 4))
        X_inv = model.inverse_transform(X_proj)
        self.assertEqual(X_inv.shape, self.X.shape)


if __name__ == '__main__':
    unittest.main()
