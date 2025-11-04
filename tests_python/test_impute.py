"""Tests for Impute module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml

class TestImpute(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        # Introduce some NaN values
        self.X[0, 0] = np.nan
        self.X[1, 1] = np.nan
        self.X[2, 2] = np.nan
        self.X[3, 3] = np.nan

    def test_knn_imputer(self):
        """Test KNNImputer"""
        imputer = auroraml.impute.KNNImputer(n_neighbors=5)
        imputer.fit(self.X, None)
        
        X_imputed = imputer.transform(self.X)
        
        # Should have no NaN values
        self.assertFalse(np.isnan(X_imputed).any())
        self.assertEqual(X_imputed.shape, self.X.shape)
        
        # Test fit_transform
        X_imputed2 = imputer.fit_transform(self.X, None)
        self.assertFalse(np.isnan(X_imputed2).any())

    def test_iterative_imputer(self):
        """Test IterativeImputer"""
        imputer = auroraml.impute.IterativeImputer(max_iter=10)
        imputer.fit(self.X, None)
        
        X_imputed = imputer.transform(self.X)
        
        # Should have no NaN values
        self.assertFalse(np.isnan(X_imputed).any())
        self.assertEqual(X_imputed.shape, self.X.shape)
        
        # Test fit_transform
        X_imputed2 = imputer.fit_transform(self.X, None)
        self.assertFalse(np.isnan(X_imputed2).any())

    def test_imputer_all_missing(self):
        """Test imputer with all missing values in a column"""
        X = self.X.copy()
        X[:, 0] = np.nan
        
        imputer = auroraml.impute.KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X, None)
        
        # Should still work (uses column mean)
        self.assertFalse(np.isnan(X_imputed).any())

if __name__ == '__main__':
    unittest.main()

