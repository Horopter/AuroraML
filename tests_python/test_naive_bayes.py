#!/usr/bin/env python3
"""
Test Suite for AuroraML Naive Bayes
Tests GaussianNB algorithm
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestGaussianNB(unittest.TestCase):
    """Test GaussianNB algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create classification data with two classes
        n_samples = 100
        self.X = np.random.randn(n_samples, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.naive_bayes as aml_nb
        
        model = aml_nb.GaussianNB()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.naive_bayes as aml_nb
        
        model = aml_nb.GaussianNB()
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        
    def test_predict_log_proba(self):
        """Test log probability prediction"""
        import auroraml.naive_bayes as aml_nb
        
        model = aml_nb.GaussianNB()
        model.fit(self.X, self.y)
        log_probabilities = model.predict_log_proba(self.X_test)
        
        self.assertEqual(log_probabilities.shape, (len(self.X_test), 2))
        
        # Convert back to probabilities and check consistency
        probabilities = np.exp(log_probabilities)
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.naive_bayes as aml_nb
        
        model = aml_nb.GaussianNB()
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('var_smoothing', params)
        
        # Test parameter setting
        model.set_params(var_smoothing=1e-8)
        self.assertEqual(model.get_params()['var_smoothing'], "0.000000")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.naive_bayes as aml_nb
        import auroraml.metrics as aml_metrics
        
        model = aml_nb.GaussianNB()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        accuracy = aml_metrics.accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.5)  # Should be better than random
        
    def test_class_prior(self):
        """Test class prior estimation"""
        import auroraml.naive_bayes as aml_nb
        
        model = aml_nb.GaussianNB()
        model.fit(self.X, self.y)
        
        # Check that class priors are estimated
        if hasattr(model, 'class_prior_'):
            priors = model.class_prior_
            self.assertEqual(len(priors), 2)
            self.assertAlmostEqual(np.sum(priors), 1.0, places=5)
            self.assertTrue(np.all(priors >= 0))
            
    def test_feature_statistics(self):
        """Test feature statistics estimation"""
        import auroraml.naive_bayes as aml_nb
        
        model = aml_nb.GaussianNB()
        model.fit(self.X, self.y)
        
        # Check that means and variances are estimated
        if hasattr(model, 'theta_') and hasattr(model, 'sigma_'):
            means = model.theta_
            variances = model.sigma_
            
            self.assertEqual(means.shape, (2, self.X.shape[1]))  # 2 classes, 4 features
            self.assertEqual(variances.shape, (2, self.X.shape[1]))
            self.assertTrue(np.all(variances > 0))  # Variances should be positive
            
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.naive_bayes as aml_nb
        
        model = aml_nb.GaussianNB()
        
        # Test with single sample per class
        X_single = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_single = np.array([0, 1])
        model.fit(X_single, y_single)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Test with empty data
        with self.assertRaises(ValueError):
            model.fit(np.array([]).reshape(0, 4), np.array([]))
            
    def test_model_persistence(self):
        """Test model saving and loading"""
        import auroraml.naive_bayes as aml_nb
        
        model = aml_nb.GaussianNB()
        model.fit(self.X, self.y)
        
        # Save model
        filename = "test_gaussian_nb.bin"
        model.save(filename)
        
        # Load model
        loaded_model = aml_nb.GaussianNB()
        loaded_model.load(filename)
        
        # Compare predictions
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Clean up
        os.remove(filename)

class TestGaussianNBIntegration(unittest.TestCase):
    """Integration tests for GaussianNB"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_cross_validation_compatibility(self):
        """Test compatibility with cross-validation"""
        import auroraml.naive_bayes as aml_nb
        import auroraml.model_selection as aml_ms
        import auroraml.metrics as aml_metrics
        
        model = aml_nb.GaussianNB()
        kfold = aml_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in kfold.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            accuracy = aml_metrics.accuracy_score(y_val, predictions)
            scores.append(accuracy)
            
        mean_score = np.mean(scores)
        self.assertGreater(mean_score, 0.5)
        
    def test_probability_consistency(self):
        """Test consistency between predict and predict_proba"""
        import auroraml.naive_bayes as aml_nb
        
        model = aml_nb.GaussianNB()
        model.fit(self.X, self.y)
        
        predictions = model.predict(self.X_test)
        probabilities = model.predict_proba(self.X_test)
        
        # Predictions should match argmax of probabilities
        predicted_classes = np.argmax(probabilities, axis=1)
        np.testing.assert_array_equal(predictions, predicted_classes)
        
    def test_different_class_distributions(self):
        """Test with different class distributions"""
        import auroraml.naive_bayes as aml_nb
        import auroraml.metrics as aml_metrics
        
        # Create imbalanced dataset
        n_samples = 100
        X_imbalanced = np.random.randn(n_samples, 4).astype(np.float64)
        y_imbalanced = np.concatenate([np.zeros(80), np.ones(20)])
        
        model = aml_nb.GaussianNB()
        model.fit(X_imbalanced, y_imbalanced)
        predictions = model.predict(X_imbalanced)
        
        accuracy = aml_metrics.accuracy_score(y_imbalanced, predictions)
        self.assertGreater(accuracy, 0.5)
        
    def test_feature_importance_estimation(self):
        """Test that the model can handle different feature scales"""
        import auroraml.naive_bayes as aml_nb
        import auroraml.metrics as aml_metrics
        
        # Create data with different feature scales
        X_scaled = np.column_stack([
            np.random.randn(100) * 0.1,  # Small scale
            np.random.randn(100) * 10.0,  # Large scale
            np.random.randn(100) * 1.0,   # Normal scale
            np.random.randn(100) * 0.01   # Very small scale
        ]).astype(np.float64)
        
        y_scaled = (X_scaled[:, 0] + X_scaled[:, 1] > 0).astype(np.int32)
        
        model = aml_nb.GaussianNB()
        model.fit(X_scaled, y_scaled)
        predictions = model.predict(X_scaled)
        
        accuracy = aml_metrics.accuracy_score(y_scaled, predictions)
        self.assertGreater(accuracy, 0.5)
        
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning with GridSearchCV"""
        import auroraml.naive_bayes as aml_nb
        import auroraml.model_selection as aml_ms
        
        model = aml_nb.GaussianNB()
        param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
        kfold = aml_ms.KFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = aml_ms.GridSearchCV(
            model, param_grid, cv=kfold, scoring='accuracy'
        )
        
        grid_search.fit(self.X, self.y)
        
        # Check that best parameters are found
        self.assertIn('var_smoothing', grid_search.best_params_)
        self.assertGreaterEqual(grid_search.best_score_, 0.0)
        self.assertLessEqual(grid_search.best_score_, 1.0)

if __name__ == '__main__':
    unittest.main()
