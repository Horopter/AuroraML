#!/usr/bin/env python3
"""
Test Suite for IngenuityML DBSCAN Algorithm
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

class TestDBSCAN(unittest.TestCase):
    """Test DBSCAN algorithm - Positive and Negative Cases"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create well-separated clusters
        n_samples = 100
        self.X = np.vstack([
            np.random.randn(n_samples // 3, 2).astype(np.float64) + np.array([2, 2]),
            np.random.randn(n_samples // 3, 2).astype(np.float64) - np.array([2, 2]),
            np.random.randn(n_samples - 2 * (n_samples // 3), 2).astype(np.float64)
        ])
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    # Positive test cases
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(self.X)
        
        self.assertEqual(len(labels), len(self.X))
        self.assertIsInstance(labels, np.ndarray)
        self.assertTrue(np.all(labels >= -1))  # -1 is noise label
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.DBSCAN(eps=0.5, min_samples=5)
        
        params = model.get_params()
        self.assertIsInstance(params, dict)
        self.assertIn('eps', params)
        self.assertIn('min_samples', params)
        
    def test_different_eps(self):
        """Test with different eps values"""
        import ingenuityml.cluster as ing_cluster
        
        for eps in [0.3, 0.5, 1.0]:
            model = ing_cluster.DBSCAN(eps=eps, min_samples=5)
            labels = model.fit_predict(self.X)
            self.assertEqual(len(labels), len(self.X))
            
    def test_different_min_samples(self):
        """Test with different min_samples values"""
        import ingenuityml.cluster as ing_cluster
        
        for min_samples in [3, 5, 10]:
            model = ing_cluster.DBSCAN(eps=0.5, min_samples=min_samples)
            labels = model.fit_predict(self.X)
            self.assertEqual(len(labels), len(self.X))
            
    def test_labels_method(self):
        """Test labels method"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.DBSCAN(eps=0.5, min_samples=5)
        model.fit(self.X, self.y_dummy)
        
        labels = model.labels()
        self.assertEqual(len(labels), len(self.X))
        self.assertTrue(np.all(labels >= -1))
        
    # Negative test cases
    def test_empty_data(self):
        """Test with empty data - should raise error"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.DBSCAN(eps=0.5, min_samples=5)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit_predict(np.array([]).reshape(0, 2))
            
    def test_negative_eps(self):
        """Test with negative eps - should raise error"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.DBSCAN(eps=-0.1, min_samples=5)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit_predict(self.X)
            
    def test_zero_min_samples(self):
        """Test with zero min_samples - should raise error"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.DBSCAN(eps=0.5, min_samples=0)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit_predict(self.X)
            
    def test_negative_min_samples(self):
        """Test with negative min_samples - should raise error"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.DBSCAN(eps=0.5, min_samples=-1)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit_predict(self.X)

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

