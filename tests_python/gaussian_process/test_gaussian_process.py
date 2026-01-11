#!/usr/bin/env python3
"""
Gaussian process tests.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))


class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(40, 3).astype(np.float64)
        self.y_reg = np.random.randn(40).astype(np.float64)
        self.y_clf = (np.random.randn(40) > 0).astype(np.int32)

    def test_gaussian_process_regressor(self):
        import ingenuityml.gaussian_process as ing_gp

        model = ing_gp.GaussianProcessRegressor(length_scale=1.0, alpha=1e-6)
        model.fit(self.X, self.y_reg)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape[0], self.X.shape[0])
        self.assertFalse(np.any(np.isnan(preds)))

    def test_gaussian_process_classifier(self):
        import ingenuityml.gaussian_process as ing_gp

        model = ing_gp.GaussianProcessClassifier(length_scale=1.0, alpha=1e-6)
        model.fit(self.X, self.y_clf.astype(np.float64))
        preds = model.predict(self.X)
        self.assertEqual(preds.shape[0], self.X.shape[0])
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])
        self.assertEqual(proba.shape[1], len(model.classes()))


if __name__ == '__main__':
    unittest.main()
