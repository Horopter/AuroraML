"""Tests for Discriminant Analysis module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml

class TestDiscriminantAnalysis(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (np.random.randn(100) > 0).astype(np.float64)
        self.y_class = (np.random.randn(100) > 0).astype(np.int32)

    def test_quadratic_discriminant_analysis(self):
        """Test QuadraticDiscriminantAnalysis"""
        qda = auroraml.discriminant_analysis.QuadraticDiscriminantAnalysis(
            regularization=0.0
        )
        qda.fit(self.X, self.y_class)
        
        predictions = qda.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        proba = qda.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])
        
        classes = qda.classes()
        self.assertGreater(len(classes), 0)

if __name__ == '__main__':
    unittest.main()

