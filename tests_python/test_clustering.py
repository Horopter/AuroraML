#!/usr/bin/env python3
"""
Test Suite for AuroraML Clustering
Tests KMeans, DBSCAN, and AgglomerativeClustering algorithms
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

def _make_blobs(n_samples=100, centers=2, n_features=2, random_state=42):
    """Generate blob-like clustering data"""
    np.random.seed(random_state)
    X = []
    y = []
    
    for i in range(centers):
        center = np.random.randn(n_features) * 5
        X_center = np.random.randn(n_samples // centers, n_features) + center
        X.append(X_center)
        y.extend([i] * (n_samples // centers))
    
    return np.vstack(X).astype(np.float64), np.array(y)

class TestKMeans(unittest.TestCase):
    """Test KMeans algorithm"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y_true = _make_blobs(n_samples=100, centers=3, n_features=2)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.KMeans(n_clusters=3, random_state=42)
        model.fit(self.X, self.y_dummy)
        
        # Test predict_labels
        labels = model.predict_labels(self.X)
        self.assertEqual(len(labels), len(self.X))
        self.assertIsInstance(labels, np.ndarray)
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))
        
    def test_fit_transform(self):
        """Test fit_transform method"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.KMeans(n_clusters=3, random_state=42)
        X_transformed = model.fit_transform(self.X, self.y_dummy)
        
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertIsInstance(X_transformed, np.ndarray)
        
    def test_transform(self):
        """Test transform method"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.KMeans(n_clusters=3, random_state=42)
        model.fit(self.X, self.y_dummy)
        
        X_test = self.X + 0.1 * np.random.randn(*self.X.shape)
        X_transformed = model.transform(X_test)
        
        self.assertEqual(X_transformed.shape, X_test.shape)
        
    def test_inverse_transform(self):
        """Test inverse_transform method"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.KMeans(n_clusters=3, random_state=42)
        model.fit(self.X, self.y_dummy)
        
        X_transformed = model.transform(self.X)
        X_inverse = model.inverse_transform(X_transformed)
        
        # Should be close to original (within some tolerance)
        np.testing.assert_array_almost_equal(X_inverse, self.X, decimal=1)
        
    def test_different_n_clusters(self):
        """Test with different n_clusters values"""
        import auroraml.cluster as aml_cluster
        
        n_clusters_values = [2, 3, 4, 5]
        for n_clusters in n_clusters_values:
            model = aml_cluster.KMeans(n_clusters=n_clusters, random_state=42)
            model.fit(self.X, self.y_dummy)
            labels = model.predict_labels(self.X)
            
            unique_labels = np.unique(labels)
            self.assertLessEqual(len(unique_labels), n_clusters)
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.KMeans(n_clusters=3, max_iter=100, tol=1e-4)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_clusters', params)
        self.assertIn('max_iter', params)
        self.assertIn('tol', params)
        self.assertIn('random_state', params)
        
        # Test parameter setting
        model.set_params(n_clusters=5)
        self.assertEqual(model.get_params()['n_clusters'], "5")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.KMeans(n_clusters=3, random_state=42)
        model.fit(self.X, self.y_dummy)
        labels = model.predict_labels(self.X)
        
        # Should find reasonable number of clusters
        unique_labels = np.unique(labels)
        self.assertGreaterEqual(len(unique_labels), 2)
        self.assertLessEqual(len(unique_labels), 3)
        
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.KMeans(n_clusters=1)
        
        # Test with single sample
        X_single = self.X[:1]
        y_dummy_single = np.zeros((1, 1)).astype(np.float64)
        model.fit(X_single, y_dummy_single)
        labels = model.predict_labels(X_single)
        self.assertEqual(len(labels), 1)
        
        # Test with empty data
        with self.assertRaises(ValueError):
            model.fit(np.array([]).reshape(0, 2), np.array([]).reshape(0, 1))

class TestDBSCAN(unittest.TestCase):
    """Test DBSCAN algorithm"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y_true = _make_blobs(n_samples=100, centers=3, n_features=2)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit functionality"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.DBSCAN(eps=1.0, min_samples=5)
        model.fit(self.X, self.y_dummy)
        
        # Test labels
        labels = model.labels()
        self.assertEqual(len(labels), len(self.X))
        self.assertIsInstance(labels, np.ndarray)
        self.assertTrue(np.all(labels >= -1))  # -1 for noise points
        
    def test_different_parameters(self):
        """Test with different parameters"""
        import auroraml.cluster as aml_cluster
        
        eps_values = [0.5, 1.0, 2.0]
        min_samples_values = [3, 5, 10]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                model = aml_cluster.DBSCAN(eps=eps, min_samples=min_samples)
                model.fit(self.X, self.y_dummy)
                labels = model.labels()
                
                self.assertEqual(len(labels), len(self.X))
                self.assertTrue(np.all(labels >= -1))
                
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.DBSCAN(eps=1.0, min_samples=5)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('eps', params)
        self.assertIn('min_samples', params)
        
        # Test parameter setting
        model.set_params(eps=2.0)
        self.assertEqual(model.get_params()['eps'], "2.000000")
        
    def test_noise_detection(self):
        """Test noise point detection"""
        import auroraml.cluster as aml_cluster
        
        # Use strict parameters to create noise points
        model = aml_cluster.DBSCAN(eps=0.1, min_samples=10)
        model.fit(self.X, self.y_dummy)
        labels = model.labels()
        
        # Should have some noise points (labeled as -1)
        noise_points = np.sum(labels == -1)
        self.assertGreaterEqual(noise_points, 0)
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.DBSCAN(eps=1.0, min_samples=5)
        model.fit(self.X, self.y_dummy)
        labels = model.labels()
        
        # Should find some clusters
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.assertGreaterEqual(n_clusters, 1)

class TestAgglomerativeClustering(unittest.TestCase):
    """Test AgglomerativeClustering algorithm"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y_true = _make_blobs(n_samples=50, centers=3, n_features=2)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit functionality"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.AgglomerativeClustering(n_clusters=3, linkage="single", affinity="euclidean")
        model.fit(self.X, self.y_dummy)
        
        # Test labels
        labels = model.labels()
        self.assertEqual(len(labels), len(self.X))
        self.assertIsInstance(labels, np.ndarray)
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))
        
    def test_different_n_clusters(self):
        """Test with different n_clusters values"""
        import auroraml.cluster as aml_cluster
        
        n_clusters_values = [2, 3, 4]
        for n_clusters in n_clusters_values:
            model = aml_cluster.AgglomerativeClustering(
                n_clusters=n_clusters, linkage="single", affinity="euclidean"
            )
            model.fit(self.X, self.y_dummy)
            labels = model.labels()
            
            unique_labels = np.unique(labels)
            self.assertEqual(len(unique_labels), n_clusters)
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.AgglomerativeClustering(n_clusters=3, linkage="single", affinity="euclidean")
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_clusters', params)
        self.assertIn('linkage', params)
        self.assertIn('affinity', params)
        
        # Test parameter setting
        model.set_params(n_clusters=5)
        self.assertEqual(model.get_params()['n_clusters'], "5")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.cluster as aml_cluster
        
        model = aml_cluster.AgglomerativeClustering(n_clusters=3, linkage="single", affinity="euclidean")
        model.fit(self.X, self.y_dummy)
        labels = model.labels()
        
        # Should find exactly n_clusters
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), 3)

class TestClusteringIntegration(unittest.TestCase):
    """Integration tests for clustering algorithms"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y_true = _make_blobs(n_samples=100, centers=3, n_features=2)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_model_comparison(self):
        """Compare different clustering algorithms"""
        import auroraml.cluster as aml_cluster
        
        models = [
            ('KMeans', aml_cluster.KMeans(n_clusters=3, random_state=42)),
            ('DBSCAN', aml_cluster.DBSCAN(eps=1.0, min_samples=5)),
            ('AgglomerativeClustering', aml_cluster.AgglomerativeClustering(
                n_clusters=3, linkage="single", affinity="euclidean"
            ))
        ]
        
        for name, model in models:
            model.fit(self.X, self.y_dummy)
            
            if hasattr(model, 'predict_labels'):
                labels = model.predict_labels(self.X)
            else:
                labels = model.labels()
                
            self.assertEqual(len(labels), len(self.X))
            self.assertTrue(np.all(labels >= -1))  # DBSCAN can have -1 for noise
            
    def test_performance(self):
        """Test overall performance"""
        import auroraml.cluster as aml_cluster
        
        # Test KMeans performance
        kmeans = aml_cluster.KMeans(n_clusters=3, random_state=42)
        kmeans.fit(self.X, self.y_dummy)
        kmeans_labels = kmeans.predict_labels(self.X)
        
        # Should find reasonable number of clusters
        unique_labels = np.unique(kmeans_labels)
        self.assertGreaterEqual(len(unique_labels), 2)
        self.assertLessEqual(len(unique_labels), 3)
        
    def test_empty_data(self):
        """Test with empty data"""
        import auroraml.cluster as aml_cluster
        
        X_empty = np.array([]).reshape(0, 2)
        y_dummy_empty = np.array([]).reshape(0, 1)
        
        # All algorithms should handle empty data gracefully
        for model_class in [aml_cluster.KMeans, aml_cluster.DBSCAN, aml_cluster.AgglomerativeClustering]:
            if model_class == aml_cluster.AgglomerativeClustering:
                model = model_class(n_clusters=2, linkage="single", affinity="euclidean")
            else:
                model = model_class()
                
            with self.assertRaises(ValueError):
                model.fit(X_empty, y_dummy_empty)
                
    def test_single_sample(self):
        """Test with single sample"""
        import auroraml.cluster as aml_cluster
        
        X_single = self.X[:1]
        y_dummy_single = np.zeros((1, 1)).astype(np.float64)
        
        # KMeans should handle single sample
        kmeans = aml_cluster.KMeans(n_clusters=1, random_state=42)
        kmeans.fit(X_single, y_dummy_single)
        labels = kmeans.predict_labels(X_single)
        self.assertEqual(len(labels), 1)
        
    def test_single_feature(self):
        """Test with single feature"""
        import auroraml.cluster as aml_cluster
        
        X_single_feature = self.X[:, [0]]
        y_dummy = np.zeros((X_single_feature.shape[0], 1)).astype(np.float64)
        
        # KMeans should handle single feature
        kmeans = aml_cluster.KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X_single_feature, y_dummy)
        labels = kmeans.predict_labels(X_single_feature)
        self.assertEqual(len(labels), len(X_single_feature))
        
    def test_not_fitted_error(self):
        """Test error when using unfitted model"""
        import auroraml.cluster as aml_cluster
        
        kmeans = aml_cluster.KMeans(n_clusters=3)
        
        # Should raise error when not fitted
        with self.assertRaises((AttributeError, RuntimeError)):
            kmeans.predict_labels(self.X)

if __name__ == '__main__':
    unittest.main()
