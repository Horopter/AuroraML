#!/usr/bin/env python3
"""
Test Suite for AuroraML DummyRegressor
Tests DummyRegressor baseline strategies
"""

import sys
import os
import unittest
import numpy as np
import random

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestDummyRegressor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(50, 3).astype(np.float64)
        self.y = np.linspace(0.0, 1.0, 50).astype(np.float64)
        self.X_test = np.random.randn(10, 3).astype(np.float64)

    def test_mean_strategy(self):
        import auroraml.ensemble as aml_ensemble
        model = aml_ensemble.DummyRegressor(strategy="mean")
        model.fit(self.X, self.y)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(np.allclose(preds, np.mean(self.y)))

    def test_median_strategy(self):
        import auroraml.ensemble as aml_ensemble
        model = aml_ensemble.DummyRegressor(strategy="median")
        model.fit(self.X, self.y)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(np.allclose(preds, np.median(self.y)))

    def test_quantile_strategy(self):
        import auroraml.ensemble as aml_ensemble
        model = aml_ensemble.DummyRegressor(strategy="quantile", quantile=0.25)
        model.fit(self.X, self.y)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(np.allclose(preds, np.quantile(self.y, 0.25)))

    def test_constant_strategy(self):
        import auroraml.ensemble as aml_ensemble
        model = aml_ensemble.DummyRegressor(strategy="constant", constant=2.5)
        model.fit(self.X, self.y)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(np.allclose(preds, 2.5))

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)

    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)
