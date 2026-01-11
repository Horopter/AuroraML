#!/usr/bin/env python3
"""
Test Suite for IngenuityML NearestCentroid
Tests NearestCentroid classifier
"""

import sys
import os
import unittest
import numpy as np
import random

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestNearestCentroid(unittest.TestCase):
    """Test NearestCentroid algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create well-separated clusters for testing
        n_samples = 100
        n_features = 4
        
        self.X = np.random.randn(n_samples, n_features).astype(np.float64)
        # Create 3 classes with different centers
        self.y = np.zeros(n_samples, dtype=np.int32)
        self.y[33:66] = 1
        self.y[66:] = 2
        
        # Shift centers for different classes
        self.X[33:66] += 2.0  # Class 1
        self.X[66:] -= 2.0    # Class 2
        
        self.X_test = np.random.randn(20, n_features).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import ingenuityml
        
        model = ingenuityml.NearestCentroid()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        unique_classes = np.unique(self.y)
        self.assertTrue(np.all(np.isin(predictions, unique_classes)))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import ingenuityml
        
        model = ingenuityml.NearestCentroid()
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        n_classes = len(np.unique(self.y))
        self.assertEqual(probabilities.shape, (len(self.X_test), n_classes))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        
    def test_decision_function(self):
        """Test decision function"""
        import ingenuityml
        
        model = ingenuityml.NearestCentroid()
        model.fit(self.X, self.y)
        decision = model.decision_function(self.X_test)
        
        self.assertEqual(len(decision), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(decision)))
        
    def test_centroid_properties(self):
        """Test that centroids are computed correctly"""
        import ingenuityml
        
        model = ingenuityml.NearestCentroid()
        model.fit(self.X, self.y)
        
        # Check that predictions are reasonable
        predictions = model.predict(self.X)
        # Should have reasonable accuracy on training data
        accuracy = np.mean(predictions == self.y)
        self.assertGreater(accuracy, 0.5)  # Should be better than random
        
    def test_binary_classification(self):
        """Test with binary classification"""
        import ingenuityml
        
        y_binary = (self.X[:, 0] > 0).astype(np.int32)
        model = ingenuityml.NearestCentroid()
        model.fit(self.X, y_binary)
        predictions = model.predict(self.X_test)
        
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_get_params(self):
        """Test parameter retrieval"""
        import ingenuityml
        
        model = ingenuityml.NearestCentroid()
        params = model.get_params()
        self.assertIsInstance(params, dict)
        
    def test_is_fitted(self):
        """Test fitted state"""
        import ingenuityml
        
        model = ingenuityml.NearestCentroid()
        self.assertFalse(model.is_fitted())
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    def test_multiclass_classification(self):
        """Test with multiclass classification"""
        import ingenuityml
        import ingenuityml.metrics as ing_metrics
        
        model = ingenuityml.NearestCentroid()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        # Should have reasonable accuracy
        accuracy = ing_metrics.accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.3)  # Better than random for 3 classes

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

