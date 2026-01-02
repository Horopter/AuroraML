#!/usr/bin/env python3
"""
Test Suite for AuroraML AgglomerativeClustering Algorithm
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

class TestAgglomerativeClustering(unittest.TestCase):
    """Test AgglomerativeClustering algorithm - Positive and Negative Cases"""
    
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
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.AgglomerativeClustering(n_clusters=3)
        labels = model.fit_predict(self.X)
        
        self.assertEqual(len(labels), len(self.X))
        self.assertIsInstance(labels, np.ndarray)
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.AgglomerativeClustering(n_clusters=3)
        
        params = model.get_params()
        self.assertIsInstance(params, dict)
        self.assertIn('n_clusters', params)
        
    def test_different_n_clusters(self):
        """Test with different numbers of clusters"""
        import auroraml.cluster as aml_cluster
        
        for n_clusters in [2, 3, 4]:
            model = aml_cluster.AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(self.X)
            self.assertEqual(len(labels), len(self.X))
            self.assertTrue(np.all(labels >= 0))
            self.assertTrue(np.all(labels < n_clusters))
            
    def test_labels_method(self):
        """Test labels method"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.AgglomerativeClustering(n_clusters=3)
        model.fit(self.X, self.y_dummy)
        
        labels = model.labels()
        self.assertEqual(len(labels), len(self.X))
        self.assertTrue(np.all(labels >= 0))
        
    # Negative test cases
    def test_empty_data(self):
        """Test with empty data - should raise error"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.AgglomerativeClustering(n_clusters=3)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit_predict(np.array([]).reshape(0, 2))
            
    def test_zero_clusters(self):
        """Test with zero clusters - should raise error"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.AgglomerativeClustering(n_clusters=0)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit_predict(self.X)
            
    def test_negative_clusters(self):
        """Test with negative clusters - should raise error"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.AgglomerativeClustering(n_clusters=-1)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit_predict(self.X)
            
    def test_more_clusters_than_samples(self):
        """Test with more clusters than samples - should raise error"""
        import auroraml.cluster as aml_cluster
        
        X_small = self.X[:5]
        
        model = aml_cluster.AgglomerativeClustering(n_clusters=10)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit_predict(X_small)

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

