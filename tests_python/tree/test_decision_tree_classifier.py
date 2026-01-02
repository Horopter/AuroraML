#!/usr/bin/env python3
"""
Test Suite for AuroraML DecisionTreeClassifier Algorithm
Includes positive and negative test cases
All tests run in shuffled order with 5-minute timeout
"""

import sys
import os
import unittest
import random
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestDecisionTreeClassifier(unittest.TestCase):
    """Test DecisionTreeClassifier algorithm - Positive and Negative Cases"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    # Positive test cases
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-5))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        
    def test_different_parameters(self):
        """Test with different parameters"""
        import auroraml.tree as aml_tree
        
        depths = [3, 5, 10]
        for depth in depths:
            model = aml_tree.DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
        
        params = model.get_params()
        self.assertIn('max_depth', params)
        self.assertIn('min_samples_split', params)
        self.assertIn('criterion', params)
        
        model.set_params({'max_depth': '10'})
        self.assertEqual(model.get_params()['max_depth'], "10")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.tree as aml_tree
        import auroraml.metrics as aml_metrics
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        accuracy = aml_metrics.accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.7)
        
    def test_feature_importance(self):
        """Test feature importance"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(self.X, self.y)
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            self.assertEqual(len(importance), self.X.shape[1])
            self.assertTrue(np.all(importance >= 0))
            self.assertAlmostEqual(np.sum(importance), 1.0, places=5)
        
    def test_is_fitted(self):
        """Test is_fitted method"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier()
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    # Negative test cases
    def test_empty_data(self):
        """Test with empty data - should raise error"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(np.array([]).reshape(0, 4), np.array([]))
            
    def test_dimension_mismatch(self):
        """Test with dimension mismatch - should raise error"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y[:-1])
            
    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5)
        
        with self.assertRaises((RuntimeError, ValueError)):
            model.predict(self.X_test)
            
    def test_wrong_feature_count(self):
        """Test predict with wrong feature count - should raise error"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(self.X, self.y)
        
        X_wrong = np.random.randn(20, 6).astype(np.float64)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.predict(X_wrong)
            
    def test_negative_max_depth(self):
        """Test with negative max_depth - edge case"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=-1)
        
        # Should either work (unlimited depth) or raise error
        try:
            model.fit(self.X, self.y)
            pred = model.predict(self.X_test)
            self.assertEqual(len(pred), len(self.X_test))
        except (ValueError, RuntimeError):
            pass
            
    def test_zero_min_samples_split(self):
        """Test with zero min_samples_split - should raise error"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(min_samples_split=0)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y)

if __name__ == '__main__':
    # Shuffle tests within this file
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    test_methods = [test for test in suite]
    random.seed(42)
    random.shuffle(test_methods)
    
    shuffled_suite = unittest.TestSuite(test_methods)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(shuffled_suite)

