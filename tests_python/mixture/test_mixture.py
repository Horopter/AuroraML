"""Tests for Mixture module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml
import random

class TestMixture(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create data from two Gaussians
        self.X1 = np.random.randn(50, 2).astype(np.float64) + np.array([2, 2])
        self.X2 = np.random.randn(50, 2).astype(np.float64) - np.array([2, 2])
        self.X = np.vstack([self.X1, self.X2]).astype(np.float64)

    def test_gaussian_mixture(self):
        """Test GaussianMixture"""
        gm = auroraml.mixture.GaussianMixture(
            n_components=2, max_iter=50, random_state=42
        )
        gm.fit(self.X, None)
        
        predictions = gm.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        proba = gm.predict_proba(self.X)
        self.assertEqual(proba.shape, (self.X.shape[0], 2))
        
        scores = gm.score_samples(self.X)
        self.assertEqual(scores.shape[0], self.X.shape[0])
        
        means = gm.means()
        self.assertEqual(len(means), 2)
        
        covariances = gm.covariances()
        self.assertEqual(len(covariances), 2)

if __name__ == '__main__':
    # Shuffle tests within this file
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    test_methods = [test for test in suite]
    random.seed(42)  # Reproducible shuffle
    random.shuffle(test_methods)
    
    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)

