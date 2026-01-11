#!/usr/bin/env python3
"""
Test Suite for IngenuityML DummyClassifier
Tests DummyClassifier for baseline comparisons
"""

import sys
import os
import unittest
import numpy as np
import random

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestDummyClassifier(unittest.TestCase):
    """Test DummyClassifier algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randint(0, 3, 100).astype(np.int32)  # 3 classes
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_most_frequent_strategy(self):
        """Test most_frequent strategy"""
        import ingenuityml.ensemble as ing_ensemble
        
        model = ing_ensemble.DummyClassifier(strategy="most_frequent")
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        # All predictions should be the most frequent class
        most_frequent = np.bincount(self.y).argmax()
        self.assertTrue(np.all(predictions == most_frequent))
        
    def test_uniform_strategy(self):
        """Test uniform strategy"""
        import ingenuityml.ensemble as ing_ensemble
        
        model = ing_ensemble.DummyClassifier(strategy="uniform")
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        # Predictions should be valid class labels
        unique_classes = np.unique(self.y)
        self.assertTrue(np.all(np.isin(predictions, unique_classes)))
        
    def test_predict_proba_most_frequent(self):
        """Test probability prediction with most_frequent strategy"""
        import ingenuityml.ensemble as ing_ensemble
        
        model = ing_ensemble.DummyClassifier(strategy="most_frequent")
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        n_classes = len(np.unique(self.y))
        self.assertEqual(probabilities.shape, (len(self.X_test), n_classes))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        
    def test_predict_proba_uniform(self):
        """Test probability prediction with uniform strategy"""
        import ingenuityml.ensemble as ing_ensemble
        
        model = ing_ensemble.DummyClassifier(strategy="uniform")
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        n_classes = len(np.unique(self.y))
        self.assertEqual(probabilities.shape, (len(self.X_test), n_classes))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        # Uniform strategy should give equal probabilities
        expected_prob = 1.0 / n_classes
        self.assertTrue(np.allclose(probabilities, expected_prob, atol=1e-6))
        
    def test_get_params(self):
        """Test parameter retrieval"""
        import ingenuityml.ensemble as ing_ensemble
        
        model = ing_ensemble.DummyClassifier(strategy="most_frequent")
        params = model.get_params()
        self.assertIn('strategy', params)
        self.assertEqual(params['strategy'], 'most_frequent')
        
    def test_is_fitted(self):
        """Test fitted state"""
        import ingenuityml.ensemble as ing_ensemble
        
        model = ing_ensemble.DummyClassifier()
        self.assertFalse(model.is_fitted())
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    def test_binary_classification(self):
        """Test with binary classification"""
        import ingenuityml.ensemble as ing_ensemble
        
        y_binary = np.random.randint(0, 2, 100).astype(np.int32)
        model = ing_ensemble.DummyClassifier(strategy="most_frequent")
        model.fit(self.X, y_binary)
        predictions = model.predict(self.X_test)
        
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

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

