#!/usr/bin/env python3
import random
"""
Test Suite for AuroraML SVM
Tests LinearSVC algorithm
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestLinearSVC(unittest.TestCase):
    """Test LinearSVC algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create classification data
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.svm as aml_svm
        
        model = aml_svm.LinearSVC(C=1.0, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_decision_function(self):
        """Test decision function"""
        import auroraml.svm as aml_svm
        
        model = aml_svm.LinearSVC(C=1.0, random_state=42)
        model.fit(self.X, self.y)
        decision_scores = model.decision_function(self.X_test)
        
        self.assertEqual(len(decision_scores), len(self.X_test))
        self.assertIsInstance(decision_scores, np.ndarray)
        
    def test_different_parameters(self):
        """Test with different parameters"""
        import auroraml.svm as aml_svm
        
        # Test different C values
        C_values = [0.1, 1.0, 10.0]
        for C in C_values:
            model = aml_svm.LinearSVC(C=C, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.svm as aml_svm
        
        model = aml_svm.LinearSVC(C=1.0, max_iter=1000, random_state=42)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('C', params)
        self.assertIn('max_iter', params)
        self.assertIn('random_state', params)
        
        # Test parameter setting
        model.set_params({'C': '2.0'})
        self.assertEqual(model.get_params()['C'], "2.000000")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.svm as aml_svm
        import auroraml.metrics as aml_metrics
        
        model = aml_svm.LinearSVC(C=1.0, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        accuracy = aml_metrics.accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.5)  # Should be better than random
        
    def test_coefficients(self):
        """Test coefficient retrieval"""
        import auroraml.svm as aml_svm
        
        model = aml_svm.LinearSVC(C=1.0, random_state=42)
        model.fit(self.X, self.y)
        
        # Check that coefficients are available
        if hasattr(model, 'coef_'):
            coef = model.coef_
            self.assertEqual(len(coef), self.X.shape[1])
            self.assertIsInstance(coef, np.ndarray)
            
        if hasattr(model, 'intercept_'):
            intercept = model.intercept_
            self.assertIsInstance(intercept, (int, float, np.number))
            
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.svm as aml_svm
        
        model = aml_svm.LinearSVC(C=1.0)
        
        # Test with single sample
        X_single = self.X[:1]
        y_single = self.y[:1]
        model.fit(X_single, y_single)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Test with empty data
        with self.assertRaises(ValueError):
            model.fit(np.array([]).reshape(0, 4), np.array([]))
            
    def test_model_persistence(self):
        """Test model saving and loading"""
        import auroraml.svm as aml_svm
        
        model = aml_svm.LinearSVC(C=1.0, random_state=42)
        model.fit(self.X, self.y)
        
        # Save model
        filename = "test_linear_svc.bin"
        model.save(filename)
        
        # Load model
        loaded_model = aml_svm.LinearSVC()
        loaded_model.load(filename)
        
        # Compare predictions
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Clean up
        os.remove(filename)

class TestLinearSVCIntegration(unittest.TestCase):
    """Integration tests for LinearSVC"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_cross_validation_compatibility(self):
        """Test compatibility with cross-validation"""
        import auroraml.svm as aml_svm
        import auroraml.model_selection as aml_ms
        import auroraml.metrics as aml_metrics
        
        model = aml_svm.LinearSVC(C=1.0, random_state=42)
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
        
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning with GridSearchCV"""
        import auroraml.svm as aml_svm
        import auroraml.model_selection as aml_ms
        
        model = aml_svm.LinearSVC(random_state=42)
        param_grid = {'C': [0.1, 1.0, 10.0]}
        kfold = aml_ms.KFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = aml_ms.GridSearchCV(
            model, param_grid, cv=kfold, scoring='accuracy'
        )
        
        grid_search.fit(self.X, self.y)
        
        # Check that best parameters are found
        self.assertIn('C', grid_search.best_params_)
        self.assertIn(grid_search.best_params_['C'], [0.1, 1.0, 10.0])
        self.assertGreaterEqual(grid_search.best_score_, 0.0)
        self.assertLessEqual(grid_search.best_score_, 1.0)
        
    def test_decision_function_consistency(self):
        """Test consistency between decision_function and predict"""
        import auroraml.svm as aml_svm
        
        model = aml_svm.LinearSVC(C=1.0, random_state=42)
        model.fit(self.X, self.y)
        
        decision_scores = model.decision_function(self.X_test)
        predictions = model.predict(self.X_test)
        
        # Predictions should match sign of decision scores
        predicted_from_scores = (decision_scores > 0).astype(int)
        np.testing.assert_array_equal(predictions, predicted_from_scores)
        
    def test_different_class_distributions(self):
        """Test with different class distributions"""
        import auroraml.svm as aml_svm
        import auroraml.metrics as aml_metrics
        
        # Create imbalanced dataset
        n_samples = 100
        X_imbalanced = np.random.randn(n_samples, 4).astype(np.float64)
        y_imbalanced = np.concatenate([np.zeros(80), np.ones(20)])
        
        model = aml_svm.LinearSVC(C=1.0, random_state=42)
        model.fit(X_imbalanced, y_imbalanced)
        predictions = model.predict(X_imbalanced)
        
        accuracy = aml_metrics.accuracy_score(y_imbalanced, predictions)
        self.assertGreater(accuracy, 0.5)
        
    def test_regularization_effect(self):
        """Test effect of regularization parameter C"""
        import auroraml.svm as aml_svm
        import auroraml.metrics as aml_metrics
        
        # Test with different C values
        C_values = [0.1, 1.0, 10.0]
        accuracies = []
        
        for C in C_values:
            model = aml_svm.LinearSVC(C=C, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X)
            accuracy = aml_metrics.accuracy_score(self.y, predictions)
            accuracies.append(accuracy)
            
        # All C values should give reasonable performance
        for acc in accuracies:
            self.assertGreater(acc, 0.5)
    
    def test_convergence(self):
        """Test that model converges with sufficient iterations"""
        import auroraml.svm as aml_svm
        
        # Test with high max_iter to ensure convergence
        model = aml_svm.LinearSVC(C=1.0, max_iter=10000, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        # Should complete without errors
        self.assertEqual(len(predictions), len(self.X_test))
        
    def test_feature_importance(self):
        """Test that model can learn feature importance"""
        import auroraml.svm as aml_svm
        
        # Create data where first feature is most important
        X_important = np.random.randn(100, 4).astype(np.float64)
        y_important = (X_important[:, 0] > 0).astype(np.int32)
        
        model = aml_svm.LinearSVC(C=1.0, random_state=42)
        model.fit(X_important, y_important)
        
        # Check that coefficients are available
        if hasattr(model, 'coef_'):
            coef = model.coef_
            # First coefficient should be largest in magnitude
            self.assertGreaterEqual(np.abs(coef[0]), np.abs(coef[1]))
            self.assertGreaterEqual(np.abs(coef[0]), np.abs(coef[2]))
            self.assertGreaterEqual(np.abs(coef[0]), np.abs(coef[3]))

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
