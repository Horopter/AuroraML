#!/usr/bin/env python3
"""
Test Suite for AuroraML LinearRegression Algorithm
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

class TestLinearRegression(unittest.TestCase):
    """Test LinearRegression algorithm - Positive and Negative Cases"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5).astype(np.float64)
        self.y = self.X @ np.array([1.0, -2.0, 0.5, 3.0, -1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        self.X_test = np.random.randn(20, 5).astype(np.float64)
        
    # Positive test cases
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertFalse(np.any(np.isnan(predictions)))
        self.assertFalse(np.any(np.isinf(predictions)))
        
    def test_coefficients(self):
        """Test coefficient retrieval"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        model.fit(self.X, self.y)
        coef = model.coef()
        intercept = model.intercept()
        
        self.assertEqual(len(coef), self.X.shape[1])
        self.assertIsInstance(intercept, (int, float, np.number))
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        
        params = model.get_params()
        self.assertIn('fit_intercept', params)
        self.assertIn('copy_X', params)
        
        # Test parameter setting
        model.set_params({'fit_intercept': 'false'})
        self.assertEqual(model.get_params()['fit_intercept'], "false")
        
    def test_performance(self):
        """Test model performance on simple data"""
        import auroraml.linear_model as aml_lm
        import auroraml.metrics as aml_metrics
        
        model = aml_lm.LinearRegression()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        r2 = aml_metrics.r2_score(self.y, predictions)
        self.assertGreater(r2, 0.8)
        
    def test_single_feature(self):
        """Test with single feature"""
        import auroraml.linear_model as aml_lm
        
        X_single = self.X[:, [0]]
        model = aml_lm.LinearRegression()
        model.fit(X_single, self.y)
        predictions = model.predict(X_single)
        
        self.assertEqual(len(predictions), len(self.y))
        
    def test_is_fitted(self):
        """Test is_fitted method"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    # Negative test cases
    def test_empty_data(self):
        """Test with empty data - should raise error"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(np.array([]).reshape(0, 5), np.array([]))
            
    def test_dimension_mismatch(self):
        """Test with dimension mismatch - should raise error"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y[:-1])  # y has wrong length
            
    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        
        with self.assertRaises((RuntimeError, ValueError)):
            model.predict(self.X_test)
            
    def test_wrong_feature_count(self):
        """Test predict with wrong feature count - should raise error"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        model.fit(self.X, self.y)
        
        X_wrong = np.random.randn(20, 6).astype(np.float64)  # Wrong number of features
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.predict(X_wrong)
            
    def test_single_sample(self):
        """Test with single sample - edge case"""
        import auroraml.linear_model as aml_lm
        
        X_single = self.X[:1]
        y_single = self.y[:1]
        
        model = aml_lm.LinearRegression()
        # Should either work or raise appropriate error
        try:
            model.fit(X_single, y_single)
            pred = model.predict(X_single)
            self.assertEqual(len(pred), 1)
        except (ValueError, RuntimeError):
            # Acceptable if single sample is not supported
            pass
            
    def test_nan_values(self):
        """Test with NaN values - should handle gracefully"""
        import auroraml.linear_model as aml_lm
        
        X_with_nan = self.X.copy()
        X_with_nan[0, 0] = np.nan
        
        model = aml_lm.LinearRegression()
        # Should either handle NaN or raise error
        try:
            model.fit(X_with_nan, self.y)
        except (ValueError, RuntimeError):
            # Acceptable if NaN handling is not implemented
            pass

if __name__ == '__main__':
    # Shuffle tests within this file
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Get all test methods and shuffle them
    test_methods = [test for test in suite]
    random.seed(42)  # Reproducible shuffle
    random.shuffle(test_methods)
    
    # Create new suite with shuffled tests
    shuffled_suite = unittest.TestSuite(test_methods)
    
    # Run with timeout (handled by test runner)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)

