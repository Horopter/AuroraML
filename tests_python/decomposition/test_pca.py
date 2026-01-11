#!/usr/bin/env python3
"""
Test Suite for IngenuityML PCA Algorithm
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

class TestPCA(unittest.TestCase):
    """Test PCA algorithm - Positive and Negative Cases"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 10).astype(np.float64)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        self.X_test = np.random.randn(20, 10).astype(np.float64)
        
    # Positive test cases
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=3)
        model.fit(self.X, self.y_dummy)
        
        X_transformed = model.transform(self.X_test)
        self.assertEqual(X_transformed.shape, (len(self.X_test), 3))
        self.assertFalse(np.any(np.isnan(X_transformed)))
        
    def test_explained_variance_ratio(self):
        """Test explained variance ratio"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=3)
        model.fit(self.X, self.y_dummy)
        
        variance_ratio = model.explained_variance_ratio()
        self.assertEqual(len(variance_ratio), 3)
        self.assertTrue(np.all(variance_ratio >= 0))
        self.assertTrue(np.all(variance_ratio <= 1))
        
    def test_components(self):
        """Test components"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=3)
        model.fit(self.X, self.y_dummy)
        
        components = model.components()
        self.assertEqual(components.shape, (3, 10))
        self.assertFalse(np.any(np.isnan(components)))
        
    def test_different_n_components(self):
        """Test with different numbers of components"""
        import ingenuityml.decomposition as ing_decomp
        
        for n_components in [1, 3, 5]:
            model = ing_decomp.PCA(n_components=n_components)
            model.fit(self.X, self.y_dummy)
            X_transformed = model.transform(self.X_test)
            self.assertEqual(X_transformed.shape, (len(self.X_test), n_components))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=3)
        
        params = model.get_params()
        self.assertIsInstance(params, dict)
        self.assertIn('n_components', params)
        
    def test_is_fitted(self):
        """Test is_fitted method"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=3)
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y_dummy)
        self.assertTrue(model.is_fitted())
        
    # Negative test cases
    def test_empty_data(self):
        """Test with empty data - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=3)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(np.array([]).reshape(0, 10), np.array([]).reshape(0, 1))
            
    def test_zero_components(self):
        """Test with zero components - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=0)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y_dummy)
            
    def test_negative_components(self):
        """Test with negative components - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=-1)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y_dummy)
            
    def test_more_components_than_features(self):
        """Test with more components than features - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=20)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y_dummy)
            
    def test_not_fitted_transform(self):
        """Test transform without fitting - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=3)
        
        with self.assertRaises((RuntimeError, ValueError)):
            model.transform(self.X_test)
            
    def test_wrong_feature_count(self):
        """Test transform with wrong feature count - should raise error"""
        import ingenuityml.decomposition as ing_decomp
        
        model = ing_decomp.PCA(n_components=3)
        model.fit(self.X, self.y_dummy)
        
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

