#!/usr/bin/env python3
"""
Test Suite for AuroraML ElasticNet Algorithm
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

class TestElasticNet(unittest.TestCase):
    """Test ElasticNet regression algorithm - Positive and Negative Cases"""
    
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
        
        model = aml_lm.ElasticNet(alpha=0.1, l1_ratio=0.5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertFalse(np.any(np.isnan(predictions)))
        
    def test_alpha_parameter(self):
        """Test alpha parameter effect"""
        import auroraml.linear_model as aml_lm
        
        alphas = [0.1, 1.0, 10.0]
        for alpha in alphas:
            model = aml_lm.ElasticNet(alpha=alpha, l1_ratio=0.5)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_l1_ratio_parameter(self):
        """Test l1_ratio parameter"""
        import auroraml.linear_model as aml_lm
        
        l1_ratios = [0.1, 0.5, 0.9]
        for l1_ratio in l1_ratios:
            model = aml_lm.ElasticNet(alpha=0.1, l1_ratio=l1_ratio)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        params = model.get_params()
        self.assertIn('alpha', params)
        self.assertIn('l1_ratio', params)
        self.assertIn('fit_intercept', params)
        
        model.set_params({'alpha': '2.0', 'l1_ratio': '0.7'})
        updated_params = model.get_params()
        self.assertEqual(updated_params['alpha'], "2.000000")
        self.assertEqual(updated_params['l1_ratio'], "0.700000")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.linear_model as aml_lm
        import auroraml.metrics as aml_metrics
        
        model = aml_lm.ElasticNet(alpha=0.1, l1_ratio=0.5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        r2 = aml_metrics.r2_score(self.y, predictions)
        self.assertGreater(r2, 0.7)
        
    def test_is_fitted(self):
        """Test is_fitted method"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.ElasticNet(alpha=0.1, l1_ratio=0.5)
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    # Negative test cases
    def test_empty_data(self):
        """Test with empty data - should raise error"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(np.array([]).reshape(0, 5), np.array([]))
            
    def test_dimension_mismatch(self):
        """Test with dimension mismatch - should raise error"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y[:-1])
            
    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        with self.assertRaises((RuntimeError, ValueError)):
            model.predict(self.X_test)
            
    def test_wrong_feature_count(self):
        """Test predict with wrong feature count - should raise error"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.ElasticNet(alpha=0.1, l1_ratio=0.5)
        model.fit(self.X, self.y)
        
        X_wrong = np.random.randn(20, 6).astype(np.float64)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.predict(X_wrong)
            
    def test_negative_alpha(self):
        """Test with negative alpha - should raise error"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.ElasticNet(alpha=-1.0, l1_ratio=0.5)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y)
            
    def test_invalid_l1_ratio(self):
        """Test with invalid l1_ratio - should raise error"""
        import auroraml.linear_model as aml_lm
        
        # Test l1_ratio < 0
        model = aml_lm.ElasticNet(alpha=0.1, l1_ratio=-0.1)
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y)
            
        # Test l1_ratio > 1
        model2 = aml_lm.ElasticNet(alpha=0.1, l1_ratio=1.5)
        with self.assertRaises((ValueError, RuntimeError)):
            model2.fit(self.X, self.y)

if __name__ == '__main__':
    # Shuffle tests within this file
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)
    
    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)

