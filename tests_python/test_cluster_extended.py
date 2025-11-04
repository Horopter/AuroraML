"""Tests for extended clustering methods"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml

class TestClusterExtended(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create two clusters
        self.X1 = np.random.randn(50, 2).astype(np.float64) + np.array([2, 2])
        self.X2 = np.random.randn(50, 2).astype(np.float64) - np.array([2, 2])
        self.X = np.vstack([self.X1, self.X2]).astype(np.float64)

    def test_mini_batch_kmeans(self):
        """Test MiniBatchKMeans"""
        kmeans = auroraml.cluster.MiniBatchKMeans(
            n_clusters=2, batch_size=20, random_state=42
        )
        kmeans.fit(self.X, None)
        
        labels = kmeans.fit_predict(self.X)
        self.assertEqual(len(labels), self.X.shape[0])
        
        predictions = kmeans.predict(self.X)
        self.assertEqual(len(predictions), self.X.shape[0])
        
        centers = kmeans.cluster_centers()
        self.assertEqual(centers.shape, (2, 2))

    def test_spectral_clustering(self):
        """Test SpectralClustering"""
        spectral = auroraml.cluster.SpectralClustering(
            n_clusters=2, affinity="rbf", gamma=1.0, random_state=42
        )
        labels = spectral.fit_predict(self.X)
        
        self.assertEqual(len(labels), self.X.shape[0])
        # Should have 2 clusters
        unique_labels = np.unique(labels)
        self.assertLessEqual(len(unique_labels), 2)
        
        labels_attr = spectral.labels()
        self.assertEqual(len(labels_attr), self.X.shape[0])

if __name__ == '__main__':
    unittest.main()

