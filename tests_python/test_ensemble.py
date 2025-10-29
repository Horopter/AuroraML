#!/usr/bin/env python3
"""
Test Suite for AuroraML Ensemble Methods
Tests RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestRandomForestClassifier(unittest.TestCase):
    """Test RandomForestClassifier algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create classification data
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.ensemble as aml_ensemble
        
        model = aml_ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.ensemble as aml_ensemble
        
        model = aml_ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        
    def test_different_parameters(self):
        """Test with different parameters"""
        import auroraml.ensemble as aml_ensemble
        
        # Test different n_estimators
        n_estimators_values = [5, 10, 20]
        for n_estimators in n_estimators_values:
            model = aml_ensemble.RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.ensemble as aml_ensemble
        
        model = aml_ensemble.RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_estimators', params)
        self.assertIn('max_depth', params)
        self.assertIn('random_state', params)
        
        # Test parameter setting
        model.set_params(n_estimators=20)
        self.assertEqual(model.get_params()['n_estimators'], "20")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.ensemble as aml_ensemble
        import auroraml.metrics as aml_metrics
        
        model = aml_ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        accuracy = aml_metrics.accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.7)
        
    def test_feature_importance(self):
        """Test feature importance"""
        import auroraml.ensemble as aml_ensemble
        
        model = aml_ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        
        # Check that feature importance is available
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            self.assertEqual(len(importance), self.X.shape[1])
            self.assertTrue(np.all(importance >= 0))
            self.assertAlmostEqual(np.sum(importance), 1.0, places=5)
            
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.ensemble as aml_ensemble
        
        model = aml_ensemble.RandomForestClassifier(n_estimators=5)
        
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
        import auroraml.ensemble as aml_ensemble
        
        model = aml_ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        
        # Save model
        filename = "test_rf_classifier.bin"
        model.save(filename)
        
        # Load model
        loaded_model = aml_ensemble.RandomForestClassifier()
        loaded_model.load(filename)
        
        # Compare predictions
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Clean up
        os.remove(filename)

class TestRandomForestRegressor(unittest.TestCase):
    """Test RandomForestRegressor algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create regression data
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = self.X @ np.array([1.0, -2.0, 0.5, 1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.ensemble as aml_ensemble
        
        model = aml_ensemble.RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_different_parameters(self):
        """Test with different parameters"""
        import auroraml.ensemble as aml_ensemble
        
        # Test different n_estimators
        n_estimators_values = [5, 10, 20]
        for n_estimators in n_estimators_values:
            model = aml_ensemble.RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.ensemble as aml_ensemble
        
        model = aml_ensemble.RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_estimators', params)
        self.assertIn('max_depth', params)
        self.assertIn('random_state', params)
        
        # Test parameter setting
        model.set_params(n_estimators=20)
        self.assertEqual(model.get_params()['n_estimators'], "20")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.ensemble as aml_ensemble
        import auroraml.metrics as aml_metrics
        
        model = aml_ensemble.RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        mse = aml_metrics.mean_squared_error(self.y, predictions)
        self.assertLess(mse, 1.0)
        
    def test_feature_importance(self):
        """Test feature importance"""
        import auroraml.ensemble as aml_ensemble
        
        model = aml_ensemble.RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        
        # Check that feature importance is available
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            self.assertEqual(len(importance), self.X.shape[1])
            self.assertTrue(np.all(importance >= 0))
            self.assertAlmostEqual(np.sum(importance), 1.0, places=5)

class TestGradientBoostingClassifier(unittest.TestCase):
    """Test GradientBoostingClassifier algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create classification data
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.gradient_boosting as aml_gb
        
        model = aml_gb.GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.gradient_boosting as aml_gb
        
        model = aml_gb.GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        
    def test_different_parameters(self):
        """Test with different parameters"""
        import auroraml.gradient_boosting as aml_gb
        
        # Test different n_estimators
        n_estimators_values = [5, 10, 20]
        for n_estimators in n_estimators_values:
            model = aml_gb.GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.gradient_boosting as aml_gb
        
        model = aml_gb.GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, random_state=42)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_estimators', params)
        self.assertIn('learning_rate', params)
        self.assertIn('random_state', params)
        
        # Test parameter setting
        model.set_params(n_estimators=20)
        self.assertEqual(model.get_params()['n_estimators'], "20")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.gradient_boosting as aml_gb
        import auroraml.metrics as aml_metrics
        
        model = aml_gb.GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        accuracy = aml_metrics.accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.7)

class TestGradientBoostingRegressor(unittest.TestCase):
    """Test GradientBoostingRegressor algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create regression data
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = self.X @ np.array([1.0, -2.0, 0.5, 1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.gradient_boosting as aml_gb
        
        model = aml_gb.GradientBoostingRegressor(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_different_parameters(self):
        """Test with different parameters"""
        import auroraml.gradient_boosting as aml_gb
        
        # Test different n_estimators
        n_estimators_values = [5, 10, 20]
        for n_estimators in n_estimators_values:
            model = aml_gb.GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.gradient_boosting as aml_gb
        
        model = aml_gb.GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, random_state=42)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_estimators', params)
        self.assertIn('learning_rate', params)
        self.assertIn('random_state', params)
        
        # Test parameter setting
        model.set_params(n_estimators=20)
        self.assertEqual(model.get_params()['n_estimators'], "20")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.gradient_boosting as aml_gb
        import auroraml.metrics as aml_metrics
        
        model = aml_gb.GradientBoostingRegressor(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        mse = aml_metrics.mean_squared_error(self.y, predictions)
        self.assertLess(mse, 1.0)

class TestEnsembleIntegration(unittest.TestCase):
    """Integration tests for ensemble methods"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y_class = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.y_reg = self.X @ np.array([1.0, -2.0, 0.5, 1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        
    def test_classifier_vs_regressor(self):
        """Compare classifier and regressor behavior"""
        import auroraml.ensemble as aml_ensemble
        import auroraml.gradient_boosting as aml_gb
        import auroraml.metrics as aml_metrics
        
        # Test RandomForest
        rf_clf = aml_ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
        rf_reg = aml_ensemble.RandomForestRegressor(n_estimators=10, random_state=42)
        
        rf_clf.fit(self.X, self.y_class)
        rf_reg.fit(self.X, self.y_reg)
        
        clf_pred = rf_clf.predict(self.X)
        reg_pred = rf_reg.predict(self.X)
        
        clf_accuracy = aml_metrics.accuracy_score(self.y_class, clf_pred)
        reg_mse = aml_metrics.mean_squared_error(self.y_reg, reg_pred)
        
        self.assertGreater(clf_accuracy, 0.7)
        self.assertLess(reg_mse, 1.0)
        
        # Test GradientBoosting
        gb_clf = aml_gb.GradientBoostingClassifier(n_estimators=10, random_state=42)
        gb_reg = aml_gb.GradientBoostingRegressor(n_estimators=10, random_state=42)
        
        gb_clf.fit(self.X, self.y_class)
        gb_reg.fit(self.X, self.y_reg)
        
        gb_clf_pred = gb_clf.predict(self.X)
        gb_reg_pred = gb_reg.predict(self.X)
        
        gb_clf_accuracy = aml_metrics.accuracy_score(self.y_class, gb_clf_pred)
        gb_reg_mse = aml_metrics.mean_squared_error(self.y_reg, gb_reg_pred)
        
        self.assertGreater(gb_clf_accuracy, 0.7)
        self.assertLess(gb_reg_mse, 1.0)
        
    def test_cross_validation_compatibility(self):
        """Test compatibility with cross-validation"""
        import auroraml.ensemble as aml_ensemble
        import auroraml.model_selection as aml_ms
        import auroraml.metrics as aml_metrics
        
        # Test RandomForest
        rf_clf = aml_ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
        kfold = aml_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in kfold.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y_class[train_idx], self.y_class[val_idx]
            
            rf_clf.fit(X_train, y_train)
            predictions = rf_clf.predict(X_val)
            accuracy = aml_metrics.accuracy_score(y_val, predictions)
            scores.append(accuracy)
            
        mean_score = np.mean(scores)
        self.assertGreater(mean_score, 0.5)
        
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning with GridSearchCV"""
        import auroraml.ensemble as aml_ensemble
        import auroraml.model_selection as aml_ms
        
        # Test RandomForest
        rf_clf = aml_ensemble.RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [5, 10], 'max_depth': [3, 5]}
        kfold = aml_ms.KFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = aml_ms.GridSearchCV(
            rf_clf, param_grid, cv=kfold, scoring='accuracy'
        )
        
        grid_search.fit(self.X, self.y_class)
        
        # Check that best parameters are found
        self.assertIn('n_estimators', grid_search.best_params_)
        self.assertIn('max_depth', grid_search.best_params_)
        self.assertGreaterEqual(grid_search.best_score_, 0.0)
        self.assertLessEqual(grid_search.best_score_, 1.0)
        
    def test_ensemble_diversity(self):
        """Test that ensemble methods show diversity"""
        import auroraml.ensemble as aml_ensemble
        import auroraml.gradient_boosting as aml_gb
        
        # Test RandomForest vs GradientBoosting
        rf_clf = aml_ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
        gb_clf = aml_gb.GradientBoostingClassifier(n_estimators=10, random_state=42)
        
        rf_clf.fit(self.X, self.y_class)
        gb_clf.fit(self.X, self.y_class)
        
        rf_pred = rf_clf.predict(self.X)
        gb_pred = gb_clf.predict(self.X)
        
        # Both should perform well
        self.assertEqual(len(rf_pred), len(self.y_class))
        self.assertEqual(len(gb_pred), len(self.y_class))
        
    def test_feature_importance_consistency(self):
        """Test that feature importance is consistent"""
        import auroraml.ensemble as aml_ensemble
        
        # Test RandomForest
        rf_clf = aml_ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
        rf_clf.fit(self.X, self.y_class)
        
        if hasattr(rf_clf, 'feature_importances_'):
            importance = rf_clf.feature_importances_
            self.assertEqual(len(importance), self.X.shape[1])
            self.assertTrue(np.all(importance >= 0))
            self.assertAlmostEqual(np.sum(importance), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
