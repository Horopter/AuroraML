#!/usr/bin/env python3
"""
Test Suite for IngenuityML DecisionTreeRegressor Algorithm
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

class TestDecisionTreeRegressor(unittest.TestCase):
    """Test DecisionTreeRegressor algorithm - Positive and Negative Cases"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = self.X @ np.array([1.0, -2.0, 0.5, 1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    # Positive test cases
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import ingenuityml.tree as ing_module
        
        model = ing_module.DecisionTreeRegressor(max_depth=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertFalse(np.any(np.isnan(predictions)))
        self.assertFalse(np.any(np.isinf(predictions)))
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import ingenuityml.tree as ing_module
        
        model = ing_module.DecisionTreeRegressor(max_depth=5)
        
        params = model.get_params()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
        
    def test_performance(self):
        """Test model performance"""
        import ingenuityml.tree as ing_module
        import ingenuityml.metrics as ing_metrics
        
        model = ing_module.DecisionTreeRegressor(max_depth=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        r2 = ing_metrics.r2_score(self.y, predictions)
        self.assertGreater(r2, 0.5)
        
    def test_is_fitted(self):
        """Test is_fitted method"""
        import ingenuityml.tree as ing_module
        
        model = ing_module.DecisionTreeRegressor(max_depth=5)
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    # Negative test cases
    def test_empty_data(self):
        """Test with empty data - should raise error"""
        import ingenuityml.tree as ing_module
        
        model = ing_module.DecisionTreeRegressor(max_depth=5)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(np.array([]).reshape(0, 4), np.array([]))
            
    def test_dimension_mismatch(self):
        """Test with dimension mismatch - should raise error"""
        import ingenuityml.tree as ing_module
        
        model = ing_module.DecisionTreeRegressor(max_depth=5)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y[:-1])
            
    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        import ingenuityml.tree as ing_module
        
        model = ing_module.DecisionTreeRegressor(max_depth=5)
        
        with self.assertRaises((RuntimeError, ValueError)):
            model.predict(self.X_test)
            
    def test_wrong_feature_count(self):
        """Test predict with wrong feature count - should raise error"""
        import ingenuityml.tree as ing_module
        
        model = ing_module.DecisionTreeRegressor(max_depth=5)
        model.fit(self.X, self.y)
        
        X_wrong = np.random.randn(20, 6).astype(np.float64)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.predict(X_wrong)

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
