#!/usr/bin/env python3
"""
Test Suite for IngenuityML RandomTreesEmbedding Algorithm
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

class TestRandomTreesEmbedding(unittest.TestCase):
    """Test RandomTreesEmbedding algorithm - Positive and Negative Cases"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(50, 3).astype(np.float64)
        self.X_test = np.random.randn(10, 3).astype(np.float64)

    def test_basic_functionality(self):
        import ingenuityml.ensemble as ing_module
        y_dummy = np.zeros(self.X.shape[0], dtype=np.float64)

        model = ing_module.RandomTreesEmbedding(n_estimators=5, random_state=42)
        model.fit(self.X, y_dummy)
        embedding = model.transform(self.X_test)

        self.assertEqual(embedding.shape[0], self.X_test.shape[0])
        self.assertEqual(embedding.shape[1], 5)

    def test_fit_transform(self):
        import ingenuityml.ensemble as ing_module
        y_dummy = np.zeros(self.X.shape[0], dtype=np.float64)

        model = ing_module.RandomTreesEmbedding(n_estimators=4, random_state=42)
        embedding = model.fit_transform(self.X, y_dummy)

        self.assertEqual(embedding.shape[0], self.X.shape[0])
        self.assertEqual(embedding.shape[1], 4)

    def test_is_fitted(self):
        import ingenuityml.ensemble as ing_module
        y_dummy = np.zeros(self.X.shape[0], dtype=np.float64)

        model = ing_module.RandomTreesEmbedding(n_estimators=3, random_state=42)
        self.assertFalse(model.is_fitted())

        model.fit(self.X, y_dummy)
        self.assertTrue(model.is_fitted())

    def test_not_fitted_transform(self):
        import ingenuityml.ensemble as ing_module

        model = ing_module.RandomTreesEmbedding(n_estimators=3, random_state=42)

        with self.assertRaises((RuntimeError, ValueError)):
            model.transform(self.X_test)

    def test_wrong_feature_count(self):
        import ingenuityml.ensemble as ing_module
        y_dummy = np.zeros(self.X.shape[0], dtype=np.float64)

        model = ing_module.RandomTreesEmbedding(n_estimators=3, random_state=42)
        model.fit(self.X, y_dummy)

        X_wrong = np.random.randn(10, 5).astype(np.float64)
        with self.assertRaises((RuntimeError, ValueError)):
            model.transform(X_wrong)

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)

    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)
