"""Tests for Outlier Detection module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml
import random

class TestOutlierDetection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        # Add some outliers
        self.X_outliers = np.vstack([
            self.X,
            np.random.randn(5, 4) * 5 + 10  # Outliers
        ]).astype(np.float64)

    def test_isolation_forest(self):
        """Test IsolationForest"""
        iso = auroraml.outlier_detection.IsolationForest(
            n_estimators=10, contamination=0.1, random_state=42
        )
        iso.fit(self.X_outliers, None)
        
        predictions = iso.predict(self.X_outliers)
        self.assertEqual(predictions.shape[0], self.X_outliers.shape[0])
        # -1 = outlier, 1 = inlier
        self.assertTrue(all(p in [-1, 1] for p in predictions))
        
        scores = iso.decision_function(self.X_outliers)
        self.assertEqual(scores.shape[0], self.X_outliers.shape[0])
        
        # Test fit_predict
        predictions2 = iso.fit_predict(self.X_outliers)
        self.assertEqual(predictions2.shape[0], self.X_outliers.shape[0])

    def test_local_outlier_factor(self):
        """Test LocalOutlierFactor"""
        lof = auroraml.outlier_detection.LocalOutlierFactor(
            n_neighbors=10, contamination=0.1
        )
        lof.fit(self.X_outliers, None)
        
        predictions = lof.predict(self.X_outliers)
        self.assertEqual(predictions.shape[0], self.X_outliers.shape[0])
        self.assertTrue(all(p in [-1, 1] for p in predictions))
        
        scores = lof.decision_function(self.X_outliers)
        self.assertEqual(scores.shape[0], self.X_outliers.shape[0])
        
        # Test fit_predict
        predictions2 = lof.fit_predict(self.X_outliers)
        self.assertEqual(predictions2.shape[0], self.X_outliers.shape[0])

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

