#!/usr/bin/env python3
"""
Cross-decomposition tests (PLSCanonical, PLSRegression, CCA, PLSSVD).
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))


class TestCrossDecomposition(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(50, 6).astype(np.float64)
        self.Y = np.random.randn(50, 3).astype(np.float64)

    def test_pls_canonical(self):
        import ingenuityml.cross_decomposition as ing_cd

        model = ing_cd.PLSCanonical(n_components=2)
        model.fit(self.X, self.Y)
        X_scores = model.transform(self.X)
        Y_scores = model.transform_y(self.Y)
        self.assertEqual(X_scores.shape, (self.X.shape[0], 2))
        self.assertEqual(Y_scores.shape, (self.Y.shape[0], 2))
        self.assertTrue(model.is_fitted())

    def test_pls_regression(self):
        import ingenuityml.cross_decomposition as ing_cd

        model = ing_cd.PLSRegression(n_components=2)
        model.fit(self.X, self.Y)
        Y_pred = model.predict(self.X)
        self.assertEqual(Y_pred.shape, (self.X.shape[0], self.Y.shape[1]))
        X_scores = model.transform(self.X)
        self.assertEqual(X_scores.shape, (self.X.shape[0], 2))

    def test_cca(self):
        import ingenuityml.cross_decomposition as ing_cd

        model = ing_cd.CCA(n_components=2)
        model.fit(self.X, self.Y)
        X_scores = model.transform(self.X)
        Y_scores = model.transform_y(self.Y)
        self.assertEqual(X_scores.shape, (self.X.shape[0], 2))
        self.assertEqual(Y_scores.shape, (self.Y.shape[0], 2))

    def test_plssvd(self):
        import ingenuityml.cross_decomposition as ing_cd

        model = ing_cd.PLSSVD(n_components=2)
        model.fit(self.X, self.Y)
        X_scores = model.transform(self.X)
        Y_scores = model.transform_y(self.Y)
        self.assertEqual(X_scores.shape, (self.X.shape[0], 2))
        self.assertEqual(Y_scores.shape, (self.Y.shape[0], 2))


if __name__ == '__main__':
    unittest.main()
