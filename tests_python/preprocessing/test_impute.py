"""Tests for Impute module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml
import random

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

    def test_missing_indicator(self):
        """Test MissingIndicator"""
        indicator = auroraml.impute.MissingIndicator(features="missing-only")
        indicator.fit(self.X, None)
        indicators = indicator.transform(self.X)
        self.assertEqual(indicators.shape[0], self.X.shape[0])

        indicator_all = auroraml.impute.MissingIndicator(features="all")
        indicator_all.fit(self.X, None)
        indicators_all = indicator_all.transform(self.X)
        self.assertEqual(indicators_all.shape, self.X.shape)

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
