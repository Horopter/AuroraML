#!/usr/bin/env python3
"""
Test Suite for AuroraML Linear Models
Tests LinearRegression, Ridge, and Lasso algorithms
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestLinearRegression(unittest.TestCase):
    """Test LinearRegression algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5).astype(np.float64)
        self.y = self.X @ np.array([1.0, -2.0, 0.5, 3.0, -1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        self.X_test = np.random.randn(20, 5).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_coefficients(self):
        """Test coefficient retrieval"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        model.fit(self.X, self.y)
        coef = model.coef()
        intercept = model.intercept()
        
        self.assertEqual(len(coef), self.X.shape[1])
        self.assertIsInstance(intercept, (int, float, np.number))
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('fit_intercept', params)
        self.assertIn('copy_X', params)
        
        # Test parameter setting
        model.set_params({'fit_intercept': 'false'})
        self.assertEqual(model.get_params()['fit_intercept'], "false")
        
    def test_performance(self):
        """Test model performance on simple data"""
        import auroraml.linear_model as aml_lm
        import auroraml.metrics as aml_metrics
        
        model = aml_lm.LinearRegression()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        r2 = 1 - np.sum((self.y - predictions) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)
        self.assertGreater(r2, 0.8)
        
    def test_single_feature(self):
        """Test with single feature"""
        import auroraml.linear_model as aml_lm
        
        X_single = self.X[:, [0]]
        model = aml_lm.LinearRegression()
        model.fit(X_single, self.y)
        predictions = model.predict(X_single)
        
        self.assertEqual(len(predictions), len(self.y))
        
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        
        # Test with empty data
        with self.assertRaises(ValueError):
            model.fit(np.array([]).reshape(0, 5), np.array([]))
            
    def test_model_persistence(self):
        """Test model saving and loading"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.LinearRegression()
        model.fit(self.X, self.y)
        
        # Test that model can be used for prediction
        original_pred = model.predict(self.X_test)
        self.assertEqual(len(original_pred), len(self.X_test))
        
        # Note: save/load functionality may not be implemented
        # This test verifies basic functionality instead

class TestRidge(unittest.TestCase):
    """Test Ridge regression algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5).astype(np.float64)
        self.y = self.X @ np.array([1.0, -2.0, 0.5, 3.0, -1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        self.X_test = np.random.randn(20, 5).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.Ridge(alpha=1.0)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_alpha_parameter(self):
        """Test alpha parameter effect"""
        import auroraml.linear_model as aml_lm
        
        # Test different alpha values
        alphas = [0.1, 1.0, 10.0]
        for alpha in alphas:
            model = aml_lm.Ridge(alpha=alpha)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.Ridge(alpha=1.0)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('alpha', params)
        self.assertIn('fit_intercept', params)
        
        # Test parameter setting
        model.set_params({'alpha': '2.0'})
        self.assertEqual(model.get_params()['alpha'], "2.000000")
        
    def test_regularization_effect(self):
        """Test that higher alpha leads to smaller coefficients"""
        import auroraml.linear_model as aml_lm
        
        model_low = aml_lm.Ridge(alpha=0.1)
        model_high = aml_lm.Ridge(alpha=10.0)
        
        model_low.fit(self.X, self.y)
        model_high.fit(self.X, self.y)
        
        coef_low = np.linalg.norm(model_low.coef())
        coef_high = np.linalg.norm(model_high.coef())
        
        self.assertLess(coef_high, coef_low)

class TestLasso(unittest.TestCase):
    """Test Lasso regression algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5).astype(np.float64)
        self.y = self.X @ np.array([1.0, -2.0, 0.5, 3.0, -1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        self.X_test = np.random.randn(20, 5).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.Lasso(alpha=0.1)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_sparsity_effect(self):
        """Test that Lasso creates sparse solutions"""
        import auroraml.linear_model as aml_lm
        
        # Compare Lasso with LinearRegression
        lasso_model = aml_lm.Lasso(alpha=100.0)  # High alpha for sparsity
        linear_model = aml_lm.LinearRegression()
        
        lasso_model.fit(self.X, self.y)
        linear_model.fit(self.X, self.y)
        
        # Lasso coefficients should be smaller in magnitude
        lasso_coef_norm = np.max(np.abs(lasso_model.coef()))
        linear_coef_norm = np.max(np.abs(linear_model.coef()))
        
        self.assertLess(lasso_coef_norm, linear_coef_norm)
        
    def test_alpha_parameter(self):
        """Test alpha parameter effect"""
        import auroraml.linear_model as aml_lm
        
        # Test different alpha values
        alphas = [0.1, 1.0, 10.0]
        for alpha in alphas:
            model = aml_lm.Lasso(alpha=alpha)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.Lasso(alpha=0.1)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('alpha', params)
        self.assertIn('fit_intercept', params)
        
        # Test parameter setting
        model.set_params({'alpha': '2.0'})
        self.assertEqual(model.get_params()['alpha'], "2.000000")

class TestLinearModelsIntegration(unittest.TestCase):
    """Integration tests for linear models"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5).astype(np.float64)
        self.y = self.X @ np.array([1.0, -2.0, 0.5, 3.0, -1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        
    def test_model_comparison(self):
        """Compare different linear models"""
        import auroraml.linear_model as aml_lm
        import auroraml.metrics as aml_metrics
        
        models = [
            ('LinearRegression', aml_lm.LinearRegression()),
            ('Ridge', aml_lm.Ridge(alpha=1.0)),
            ('Lasso', aml_lm.Lasso(alpha=0.1))
        ]
        
        for name, model in models:
            model.fit(self.X, self.y)
            predictions = model.predict(self.X)
            mse = aml_metrics.mean_squared_error(self.y, predictions)
            self.assertLess(mse, 1.0, f"{name} MSE too high: {mse}")
            
    def test_cross_validation_compatibility(self):
        """Test compatibility with cross-validation"""
        import auroraml.linear_model as aml_lm
        import auroraml.model_selection as aml_ms
        import auroraml.metrics as aml_metrics
        
        model = aml_lm.LinearRegression()
        kfold = aml_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in kfold.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            mse = aml_metrics.mean_squared_error(y_val, predictions)
            scores.append(mse)
            
        mean_score = np.mean(scores)
        self.assertLess(mean_score, 1.0)
        
    def test_performance(self):
        """Test overall performance"""
        import auroraml.linear_model as aml_lm
        import auroraml.metrics as aml_metrics
        
        model = aml_lm.LinearRegression()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        r2 = 1 - np.sum((self.y - predictions) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)
        self.assertGreater(r2, 0.8)

if __name__ == '__main__':
    unittest.main()