"""Tests for Semi-supervised module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ingenuityml
import random

class TestSemiSupervised(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        # Create labels with some unlabeled (-1)
        self.y = (np.random.randn(100) > 0).astype(np.int32)
        self.y[50:] = -1  # Make half unlabeled

    def test_label_propagation(self):
        """Test LabelPropagation"""
        lp = ingenuityml.semi_supervised.LabelPropagation(
            gamma=20.0, max_iter=10, kernel="rbf"
        )
        lp.fit(self.X, self.y.astype(np.float64))
        
        predictions = lp.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        proba = lp.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])
        
        classes = lp.classes()
        self.assertGreater(len(classes), 0)

    def test_label_spreading(self):
        """Test LabelSpreading"""
        ls = ingenuityml.semi_supervised.LabelSpreading(
            alpha=0.2, gamma=20.0, max_iter=10, kernel="rbf"
        )
        ls.fit(self.X, self.y.astype(np.float64))
        
        predictions = ls.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        proba = ls.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])
        
        classes = ls.classes()
        self.assertGreater(len(classes), 0)

    def test_self_training_classifier(self):
        """Test SelfTrainingClassifier"""
        st = ingenuityml.semi_supervised.SelfTrainingClassifier(
            n_neighbors=3, threshold=0.6, max_iter=5
        )
        st.fit(self.X, self.y.astype(np.float64))

        predictions = st.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

        proba = st.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])

        classes = st.classes()
        self.assertGreater(len(classes), 0)

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
