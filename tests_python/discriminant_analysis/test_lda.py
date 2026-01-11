#!/usr/bin/env python3
"""
Test Suite for IngenuityML LDA Algorithm
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

class TestLDA(unittest.TestCase):
    """Test LDA algorithm - Positive and Negative Cases"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 10).astype(np.float64)
        # Create classification labels
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 10).astype(np.float64)
        
    # Positive test cases
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.LDA(n_components=1)
        model.fit(self.X, self.y)
        
        X_transformed = model.transform(self.X_test)
        self.assertEqual(X_transformed.shape, (len(self.X_test), 1))
        self.assertFalse(np.any(np.isnan(X_transformed)))
        
    def test_explained_variance_ratio(self):
        """Test explained variance ratio"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.LDA(n_components=1)
        model.fit(self.X, self.y)
        
        variance_ratio = model.explained_variance_ratio()
        self.assertGreater(len(variance_ratio), 0)
        self.assertTrue(np.all(variance_ratio >= 0))
        
    def test_components(self):
        """Test components"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.LDA(n_components=1)
        model.fit(self.X, self.y)
        
        components = model.components()
        self.assertGreater(components.shape[0], 0)
        self.assertFalse(np.any(np.isnan(components)))
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.LDA(n_components=1)
        
        params = model.get_params()
        self.assertIsInstance(params, dict)
        self.assertIn('n_components', params)
        
    def test_is_fitted(self):
        """Test is_fitted method"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.LDA(n_components=1)
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    # Negative test cases
    def test_empty_data(self):
        """Test with empty data - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.LDA(n_components=1)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(np.array([]).reshape(0, 10), np.array([]))
            
    def test_dimension_mismatch(self):
        """Test with dimension mismatch - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.LDA(n_components=1)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y[:-1])
            
    def test_single_class(self):
        """Test with single class - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.LDA(n_components=1)
        y_single = np.zeros(len(self.X), dtype=np.int32)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, y_single)
            
    def test_not_fitted_transform(self):
        """Test transform without fitting - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.LDA(n_components=1)
        
        with self.assertRaises((RuntimeError, ValueError)):
            model.transform(self.X_test)
            
    def test_wrong_feature_count(self):
        """Test transform with wrong feature count - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.LDA(n_components=1)
        model.fit(self.X, self.y)
        
        X_wrong = np.random.randn(20, 15).astype(np.float64)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.transform(X_wrong)

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

