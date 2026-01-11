#!/usr/bin/env python3
"""
Test Suite for IngenuityML BernoulliRBM Algorithm
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

class TestBernoulliRBM(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.rand(50, 6).astype(np.float64)
        self.X_test = np.random.rand(10, 6).astype(np.float64)

    def test_basic_functionality(self):
        import ingenuityml.neural_network as ing_module
        y_dummy = np.zeros(self.X.shape[0], dtype=np.float64)

        model = ing_module.BernoulliRBM(n_components=4, n_iter=5, random_state=42)
        model.fit(self.X, y_dummy)
        transformed = model.transform(self.X_test)

        self.assertEqual(transformed.shape[0], self.X_test.shape[0])
        self.assertEqual(transformed.shape[1], 4)

    def test_inverse_transform(self):
        import ingenuityml.neural_network as ing_module
        y_dummy = np.zeros(self.X.shape[0], dtype=np.float64)

        model = ing_module.BernoulliRBM(n_components=3, n_iter=5, random_state=42)
        model.fit(self.X, y_dummy)
        hidden = model.transform(self.X_test)
        reconstructed = model.inverse_transform(hidden)

        self.assertEqual(reconstructed.shape, self.X_test.shape)

    def test_is_fitted(self):
        import ingenuityml.neural_network as ing_module
        y_dummy = np.zeros(self.X.shape[0], dtype=np.float64)

        model = ing_module.BernoulliRBM(n_components=3, n_iter=2, random_state=42)
        self.assertFalse(model.is_fitted())

        model.fit(self.X, y_dummy)
        self.assertTrue(model.is_fitted())

    def test_not_fitted_transform(self):
        import ingenuityml.neural_network as ing_module

        model = ing_module.BernoulliRBM(n_components=3, n_iter=2, random_state=42)
        with self.assertRaises((RuntimeError, ValueError)):
            model.transform(self.X_test)

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)

    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)
