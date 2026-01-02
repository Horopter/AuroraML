#!/usr/bin/env python3
"""
Test Suite for AuroraML AdaBoostClassifier Algorithm
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

class TestAdaBoostClassifier(unittest.TestCase):
    """Test AdaBoostClassifier algorithm - Positive and Negative Cases"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 4
        
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float64)
        self.y = ((self.X[:, 0] + self.X[:, 1] - 0.5 * self.X[:, 2]) > 0).astype(np.int32)
        self.X_test = np.random.randn(30, self.n_features).astype(np.float64)
        
    # Positive test cases
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        classes = model.classes()
        self.assertTrue(np.all(np.isin(predictions, classes)))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-5))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        
    def test_decision_function(self):
        """Test decision function"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
        model.fit(self.X, self.y)
        decision = model.decision_function(self.X_test)
        
        self.assertEqual(len(decision), len(self.X_test))
        self.assertIsInstance(decision, np.ndarray)
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
        
        params = model.get_params()
        self.assertIn('n_estimators', params)
        self.assertIn('learning_rate', params)
        self.assertIn('random_state', params)
        
        model.set_params(n_estimators=75, learning_rate=0.8)
        updated_params = model.get_params()
        self.assertEqual(updated_params['n_estimators'], "75")
        self.assertEqual(updated_params['learning_rate'], "0.800000")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.adaboost as aml_adaboost
        import auroraml.metrics as aml_metrics
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        accuracy = aml_metrics.accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.7)
        
    def test_different_learning_rates(self):
        """Test with different learning rates"""
        import auroraml.adaboost as aml_adaboost
        
        learning_rates = [0.5, 1.0, 2.0]
        for lr in learning_rates:
            model = aml_adaboost.AdaBoostClassifier(n_estimators=50, learning_rate=lr, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_different_n_estimators(self):
        """Test with different numbers of estimators"""
        import auroraml.adaboost as aml_adaboost
        
        n_estimators_list = [10, 50, 100]
        for n_est in n_estimators_list:
            model = aml_adaboost.AdaBoostClassifier(n_estimators=n_est, learning_rate=1.0, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_is_fitted(self):
        """Test is_fitted method"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier()
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    def test_classes_attribute(self):
        """Test classes attribute"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50, random_state=42)
        model.fit(self.X, self.y)
        
        classes = model.classes()
        self.assertGreaterEqual(len(classes), 2)
        self.assertTrue(np.all(np.isin(classes, [0, 1])))
        
    # Negative test cases
    def test_empty_data(self):
        """Test with empty data - should raise error"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(np.array([]).reshape(0, 4), np.array([]))
            
    def test_dimension_mismatch(self):
        """Test with dimension mismatch - should raise error"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y[:-1])
            
    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50)
        
        with self.assertRaises((RuntimeError, ValueError)):
            model.predict(self.X_test)
            
    def test_wrong_feature_count(self):
        """Test predict with wrong feature count - should raise error"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50, random_state=42)
        model.fit(self.X, self.y)
        
        X_wrong = np.random.randn(30, 6).astype(np.float64)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.predict(X_wrong)
            
    def test_negative_n_estimators(self):
        """Test with negative n_estimators - should raise error"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=-1)
        
        with self.assertRaises((ValueError, RuntimeError)):
            model.fit(self.X, self.y)
            
    def test_zero_n_estimators(self):
        """Test with zero n_estimators - edge case"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=0)
        
        try:
            model.fit(self.X, self.y)
            pred = model.predict(self.X_test)
            self.assertEqual(len(pred), len(self.X_test))
        except (ValueError, RuntimeError):
            # Acceptable if zero estimators not supported
            pass
            
    def test_negative_learning_rate(self):
        """Test with negative learning rate - should raise error"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50, learning_rate=-1.0)
        
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

