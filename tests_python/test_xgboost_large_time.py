#!/usr/bin/env python3
"""
Test Suite for AuroraML XGBoost Algorithms
Tests XGBClassifier and XGBRegressor
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestXGBClassifier(unittest.TestCase):
    """Test XGBClassifier algorithm"""
    
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
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        
        # Check predictions are valid classes
        classes = model.classes()
        self.assertTrue(np.all(np.isin(predictions, classes)))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-5))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        
    def test_decision_function(self):
        """Test decision function"""
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(self.X, self.y)
        decision = model.decision_function(self.X_test)
        
        self.assertEqual(len(decision), len(self.X_test))
        self.assertIsInstance(decision, np.ndarray)
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=6,
                                     gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
                                     min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
                                     random_state=42)
        
        params = model.get_params()
        self.assertIn('n_estimators', params)
        self.assertIn('learning_rate', params)
        self.assertIn('max_depth', params)
        self.assertIn('gamma', params)
        self.assertIn('reg_alpha', params)
        self.assertIn('reg_lambda', params)
        self.assertIn('min_child_weight', params)
        self.assertIn('subsample', params)
        self.assertIn('colsample_bytree', params)
        
        # Test parameter setting
        model.set_params(n_estimators=75, reg_lambda=3.0)
        updated_params = model.get_params()
        self.assertEqual(updated_params['n_estimators'], "75")
        self.assertEqual(updated_params['reg_lambda'], "3.000000")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.xgboost as aml_xgb
        import auroraml.metrics as aml_metrics
        
        model = aml_xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        accuracy = aml_metrics.accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.7)
        
    def test_regularization(self):
        """Test with different regularization parameters"""
        import auroraml.xgboost as aml_xgb
        
        # Low regularization
        model_low = aml_xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=6,
                                          reg_lambda=0.1, random_state=42)
        model_low.fit(self.X, self.y)
        
        # High regularization
        model_high = aml_xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=6,
                                          reg_lambda=10.0, random_state=42)
        model_high.fit(self.X, self.y)
        
        pred_low = model_low.predict(self.X_test)
        pred_high = model_high.predict(self.X_test)
        
        self.assertEqual(len(pred_low), len(self.X_test))
        self.assertEqual(len(pred_high), len(self.X_test))
        
    def test_subsampling(self):
        """Test with subsampling"""
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=6,
                                     subsample=0.5, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
    def test_is_fitted(self):
        """Test is_fitted method"""
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBClassifier()
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    def test_classes_attribute(self):
        """Test classes attribute"""
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBClassifier(n_estimators=50, random_state=42)
        model.fit(self.X, self.y)
        
        classes = model.classes()
        self.assertGreaterEqual(len(classes), 2)

class TestXGBRegressor(unittest.TestCase):
    """Test XGBRegressor algorithm"""
    
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
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertFalse(np.any(np.isnan(predictions)))
        self.assertFalse(np.any(np.isinf(predictions)))
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=6,
                                    gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
                                    min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
                                    random_state=42)
        
        params = model.get_params()
        self.assertIn('n_estimators', params)
        self.assertIn('learning_rate', params)
        self.assertIn('max_depth', params)
        
        # Test parameter setting
        model.set_params(n_estimators=75, reg_lambda=3.0)
        updated_params = model.get_params()
        self.assertEqual(updated_params['n_estimators'], "75")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.xgboost as aml_xgb
        import auroraml.metrics as aml_metrics
        
        model = aml_xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        mse = aml_metrics.mean_squared_error(self.y, predictions)
        self.assertLess(mse, 5.0)
        
        r2 = aml_metrics.r2_score(self.y, predictions)
        self.assertGreater(r2, 0.6)
        
    def test_regularization(self):
        """Test with different regularization parameters"""
        import auroraml.xgboost as aml_xgb
        
        # Low regularization
        model_low = aml_xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=6,
                                        reg_lambda=0.1, random_state=42)
        model_low.fit(self.X, self.y)
        
        # High regularization
        model_high = aml_xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=6,
                                         reg_lambda=10.0, random_state=42)
        model_high.fit(self.X, self.y)
        
        pred_low = model_low.predict(self.X_test)
        pred_high = model_high.predict(self.X_test)
        
        self.assertEqual(len(pred_low), len(self.X_test))
        self.assertEqual(len(pred_high), len(self.X_test))
        
    def test_subsampling(self):
        """Test with subsampling"""
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=6,
                                    subsample=0.7, colsample_bytree=0.7, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
    def test_is_fitted(self):
        """Test is_fitted method"""
        import auroraml.xgboost as aml_xgb
        
        model = aml_xgb.XGBRegressor()
        self.assertFalse(model.is_fitted())
        
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
        
    def test_consistency(self):
        """Test model consistency with same random seed"""
        import auroraml.xgboost as aml_xgb
        
        model1 = aml_xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=6, random_state=42)
        model2 = aml_xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=6, random_state=42)
        
        model1.fit(self.X, self.y)
        model2.fit(self.X, self.y)
        
        pred1 = model1.predict(self.X_test)
        pred2 = model2.predict(self.X_test)
        
        np.testing.assert_allclose(pred1, pred2, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()

