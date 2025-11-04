"""Tests for ExtraTree"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml

class TestExtraTree(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randn(100).astype(np.float64)
        self.y_class = (np.random.randn(100) > 0).astype(np.int32)

    def test_extratree_classifier(self):
        """Test ExtraTreeClassifier"""
        clf = auroraml.tree.ExtraTreeClassifier(
            max_depth=3, random_state=42
        )
        clf.fit(self.X, self.y_class)
        
        predictions = clf.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        proba = clf.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])
        
        classes = clf.classes()
        self.assertGreater(len(classes), 0)

    def test_extratree_regressor(self):
        """Test ExtraTreeRegressor"""
        reg = auroraml.tree.ExtraTreeRegressor(
            max_depth=3, random_state=42
        )
        reg.fit(self.X, self.y)
        
        predictions = reg.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

if __name__ == '__main__':
    unittest.main()

