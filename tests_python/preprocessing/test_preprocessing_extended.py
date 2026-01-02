"""Tests for extended preprocessing utilities"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml
import random

class TestPreprocessingExtended(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)

    def test_max_abs_scaler(self):
        """Test MaxAbsScaler"""
        scaler = auroraml.preprocessing.MaxAbsScaler()
        scaler.fit(self.X, None)
        
        X_transformed = scaler.transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)
        
        # Values should be in [-1, 1]
        self.assertTrue(np.all(np.abs(X_transformed) <= 1.0 + 1e-10))
        
        max_abs = scaler.max_abs()
        self.assertEqual(len(max_abs), self.X.shape[1])
        
        # Test inverse transform
        X_inverse = scaler.inverse_transform(X_transformed)
        np.testing.assert_array_almost_equal(X_inverse, self.X, decimal=5)

    def test_binarizer(self):
        """Test Binarizer"""
        binarizer = auroraml.preprocessing.Binarizer(threshold=0.0)
        binarizer.fit(self.X, None)
        
        X_binary = binarizer.transform(self.X)
        self.assertEqual(X_binary.shape, self.X.shape)
        
        # Should be binary (0 or 1)
        self.assertTrue(np.all((X_binary == 0) | (X_binary == 1)))
        
        # Test fit_transform
        X_binary2 = binarizer.fit_transform(self.X, None)
        self.assertTrue(np.all((X_binary2 == 0) | (X_binary2 == 1)))

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

