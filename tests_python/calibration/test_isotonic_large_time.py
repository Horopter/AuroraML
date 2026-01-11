"""Tests for Isotonic module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ingenuityml
import random

class TestIsotonic(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create 1D data for isotonic regression
        self.X = np.random.randn(100, 1).astype(np.float64)
        self.y = np.random.randn(100).astype(np.float64)

    def test_isotonic_regression(self):
        """Test IsotonicRegression"""
        iso = ingenuityml.isotonic.IsotonicRegression(increasing=True)
        iso.fit(self.X, self.y)
        
        predictions = iso.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        # Test transform
        transformed = iso.transform(self.y)
        self.assertEqual(len(transformed), len(self.y))

    def test_isotonic_regression_decreasing(self):
        """Test IsotonicRegression with decreasing=True"""
        iso = ingenuityml.isotonic.IsotonicRegression(increasing=False)
        iso.fit(self.X, self.y)
        
        predictions = iso.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

if __name__ == '__main__':
    # Shuffle tests within this file
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    test_methods = [test for test in suite]
    random.seed(42)  # Reproducible shuffle
    random.shuffle(test_methods)
    
    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)

