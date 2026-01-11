#!/usr/bin/env python3
"""
Test Suite for IngenuityML KMeans Algorithm
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

class TestKMeans(unittest.TestCase):
    """Test KMeans algorithm - Positive and Negative Cases"""
    
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
        
        model = ing_cluster.KMeans(n_clusters=3, random_state=42)
        model.fit(self.X, self.y_dummy)
        
        labels = model.predict_labels(self.X)
        self.assertEqual(len(labels), len(self.X))
        self.assertIsInstance(labels, np.ndarray)
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))
        
    def test_cluster_centers(self):
        """Test cluster centers"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.KMeans(n_clusters=3, random_state=42)
        model.fit(self.X, self.y_dummy)
        
        centers = model.cluster_centers()
        self.assertEqual(centers.shape, (3, 2))
        self.assertFalse(np.any(np.isnan(centers)))
        self.assertFalse(np.any(np.isinf(centers)))
        
    def test_inertia(self):
        """Test inertia calculation"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.KMeans(n_clusters=3, random_state=42)
        model.fit(self.X, self.y_dummy)
        
        inertia = model.inertia()
        self.assertGreater(inertia, 0.0)
        self.assertFalse(np.isnan(inertia))
        self.assertFalse(np.isinf(inertia))
        
    def test_different_k(self):
        """Test with different numbers of clusters"""
        import ingenuityml.cluster as ing_cluster
        
        for k in [2, 3, 4]:
            model = ing_cluster.KMeans(n_clusters=k, random_state=42)
            model.fit(self.X, self.y_dummy)
            labels = model.predict_labels(self.X)
            self.assertEqual(len(labels), len(self.X))
            self.assertTrue(np.all(labels >= 0))
            self.assertTrue(np.all(labels < k))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.KMeans(n_clusters=3, random_state=42)
        
        params = model.get_params()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
        
    def test_is_fitted(self):
        """Test is_fitted method"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.KMeans(n_clusters=3)
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y_dummy)
        self.assertTrue(model.is_fitted())
        
    # Negative test cases
    def test_empty_data(self):
        """Test with empty data - should raise error"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.KMeans(n_clusters=3)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(np.array([]).reshape(0, 2), np.array([]).reshape(0, 1))
            
    def test_zero_clusters(self):
        """Test with zero clusters - should raise error"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.KMeans(n_clusters=0)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y_dummy)
            
    def test_negative_clusters(self):
        """Test with negative clusters - should raise error"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.KMeans(n_clusters=-1)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y_dummy)
            
    def test_more_clusters_than_samples(self):
        """Test with more clusters than samples - should raise error"""
        import ingenuityml.cluster as ing_cluster
        
        X_small = self.X[:5]
        y_small = self.y_dummy[:5]
        
        model = ing_cluster.KMeans(n_clusters=10)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(X_small, y_small)
            
    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        import ingenuityml.cluster as ing_cluster
        
        model = ing_cluster.KMeans(n_clusters=3)
        
        with self.assertRaises((RuntimeError, ValueError)):
            model.predict_labels(self.X)

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

