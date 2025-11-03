#!/usr/bin/env python3
"""
Test Suite for AuroraML AdaBoost Algorithms
Tests AdaBoostClassifier and AdaBoostRegressor
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestAdaBoostClassifier(unittest.TestCase):
    """Test AdaBoostClassifier algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 4
        
        # Create classification data
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float64)
        self.y = ((self.X[:, 0] + self.X[:, 1] - 0.5 * self.X[:, 2]) > 0).astype(np.int32)
        self.X_test = np.random.randn(30, self.n_features).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        
        # Check predictions are valid classes
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
        
        # Test parameter setting
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

class TestAdaBoostRegressor(unittest.TestCase):
    """Test AdaBoostRegressor algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 4
        
        # Create regression data
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float64)
        self.y = (2.0 * self.X[:, 0] + 1.5 * self.X[:, 1] - 
                  0.8 * self.X[:, 2] + 0.1 * self.X[:, 3] + 
                  0.05 * np.random.randn(self.n_samples)).astype(np.float64)
        self.X_test = np.random.randn(30, self.n_features).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertFalse(np.any(np.isnan(predictions)))
        self.assertFalse(np.any(np.isinf(predictions)))
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostRegressor(n_estimators=50, learning_rate=1.0, 
                                               loss='linear', random_state=42)
        
        params = model.get_params()
        self.assertIn('n_estimators', params)
        self.assertIn('learning_rate', params)
        self.assertIn('loss', params)
        self.assertIn('random_state', params)
        
        # Test parameter setting
        model.set_params(n_estimators=75, loss='square')
        updated_params = model.get_params()
        self.assertEqual(updated_params['n_estimators'], "75")
        self.assertEqual(updated_params['loss'], "square")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.adaboost as aml_adaboost
        import auroraml.metrics as aml_metrics
        
        model = aml_adaboost.AdaBoostRegressor(n_estimators=100, learning_rate=1.0, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        mse = aml_metrics.mean_squared_error(self.y, predictions)
        self.assertLess(mse, 10.0)
        
        r2 = aml_metrics.r2_score(self.y, predictions)
        self.assertGreater(r2, 0.5)
        
    def test_different_losses(self):
        """Test with different loss functions"""
        import auroraml.adaboost as aml_adaboost
        
        losses = ['linear', 'square', 'exponential']
        for loss in losses:
            model = aml_adaboost.AdaBoostRegressor(n_estimators=50, learning_rate=1.0, 
                                                  loss=loss, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            self.assertFalse(np.any(np.isnan(predictions)))
            
    def test_different_learning_rates(self):
        """Test with different learning rates"""
        import auroraml.adaboost as aml_adaboost
        
        learning_rates = [0.5, 1.0, 2.0]
        for lr in learning_rates:
            model = aml_adaboost.AdaBoostRegressor(n_estimators=50, learning_rate=lr, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_is_fitted(self):
        """Test is_fitted method"""
        import auroraml.adaboost as aml_adaboost
        
        model = aml_adaboost.AdaBoostRegressor()
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    def test_consistency(self):
        """Test model consistency with same random seed"""
        import auroraml.adaboost as aml_adaboost
        
        model1 = aml_adaboost.AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42)
        model2 = aml_adaboost.AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42)
        
        model1.fit(self.X, self.y)
        model2.fit(self.X, self.y)
        
        pred1 = model1.predict(self.X_test)
        pred2 = model2.predict(self.X_test)
        
        np.testing.assert_allclose(pred1, pred2, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()

