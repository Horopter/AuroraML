"""Tests for Calibration module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml

class TestCalibration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (np.random.randn(100) > 0).astype(np.float64)
        self.y_class = (np.random.randn(100) > 0).astype(np.int32)

    def test_calibrated_classifier_cv(self):
        """Test CalibratedClassifierCV"""
        base_clf = auroraml.tree.DecisionTreeClassifier(max_depth=3)
        
        calibrated = auroraml.calibration.CalibratedClassifierCV(
            base_estimator=base_clf, method="sigmoid", cv=3
        )
        calibrated.fit(self.X, self.y_class)
        
        predictions = calibrated.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        proba = calibrated.predict_proba(self.X)
        self.assertEqual(proba.shape, (self.X.shape[0], 2))
        
        classes = calibrated.classes()
        self.assertGreater(len(classes), 0)

if __name__ == '__main__':
    unittest.main()

