"""Tests for Isotonic module - Fast tests only"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml

class TestIsotonic(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create small 1D data for isotonic regression (fast test)
        self.X = np.random.randn(20, 1).astype(np.float64)  # Reduced from 100 to 20
        self.y = np.random.randn(20).astype(np.float64)

    def test_isotonic_regression_small(self):
        """Test IsotonicRegression with small dataset"""
        iso = auroraml.isotonic.IsotonicRegression(increasing=True)
        iso.fit(self.X, self.y)
        
        predictions = iso.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        # Test transform
        transformed = iso.transform(self.y)
        self.assertEqual(len(transformed), len(self.y))

    def test_isotonic_regression_decreasing_small(self):
        """Test IsotonicRegression with decreasing=True and small dataset"""
        iso = auroraml.isotonic.IsotonicRegression(increasing=False)
        iso.fit(self.X, self.y)
        
        predictions = iso.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

if __name__ == '__main__':
    unittest.main()
