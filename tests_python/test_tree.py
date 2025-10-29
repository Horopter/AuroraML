#!/usr/bin/env python3
"""
Test Suite for AuroraML Tree Algorithms
Tests DecisionTreeClassifier and DecisionTreeRegressor algorithms
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestDecisionTreeClassifier(unittest.TestCase):
    """Test DecisionTreeClassifier algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create classification data
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        
    def test_different_parameters(self):
        """Test with different parameters"""
        import auroraml.tree as aml_tree
        
        # Test different max_depth values
        depths = [3, 5, 10]
        for depth in depths:
            model = aml_tree.DecisionTreeClassifier(max_depth=depth)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5, min_samples_split=2)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('max_depth', params)
        self.assertIn('min_samples_split', params)
        self.assertIn('min_samples_leaf', params)
        self.assertIn('criterion', params)
        
        # Test parameter setting
        model.set_params({'max_depth': '10'})
        self.assertEqual(model.get_params()['max_depth'], "10")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.tree as aml_tree
        import auroraml.metrics as aml_metrics
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        accuracy = aml_metrics.accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.7)
        
    def test_feature_importance(self):
        """Test feature importance"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5)
        model.fit(self.X, self.y)
        
        # Feature importance should be available
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            self.assertEqual(len(importance), self.X.shape[1])
            self.assertTrue(np.all(importance >= 0))
            self.assertAlmostEqual(np.sum(importance), 1.0, places=5)
        
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=1)
        
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
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeClassifier(max_depth=5)
        model.fit(self.X, self.y)
        
        # Save model
        filename = "test_dt_classifier.bin"
        model.save(filename)
        
        # Load model
        loaded_model = aml_tree.DecisionTreeClassifier()
        loaded_model.load(filename)
        
        # Compare predictions
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Clean up
        os.remove(filename)

class TestDecisionTreeRegressor(unittest.TestCase):
    """Test DecisionTreeRegressor algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create regression data
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = self.X @ np.array([1.0, -2.0, 0.5, 1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeRegressor(max_depth=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_different_parameters(self):
        """Test with different parameters"""
        import auroraml.tree as aml_tree
        
        # Test different max_depth values
        depths = [3, 5, 10]
        for depth in depths:
            model = aml_tree.DecisionTreeRegressor(max_depth=depth)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeRegressor(max_depth=5, min_samples_split=2)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('max_depth', params)
        self.assertIn('min_samples_split', params)
        self.assertIn('min_samples_leaf', params)
        self.assertIn('criterion', params)
        
        # Test parameter setting
        model.set_params({'max_depth': '10'})
        self.assertEqual(model.get_params()['max_depth'], "10")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.tree as aml_tree
        import auroraml.metrics as aml_metrics
        
        model = aml_tree.DecisionTreeRegressor(max_depth=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        mse = aml_metrics.mean_squared_error(self.y, predictions)
        self.assertLess(mse, 1.0)
        
    def test_feature_importance(self):
        """Test feature importance"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeRegressor(max_depth=5)
        model.fit(self.X, self.y)
        
        # Feature importance should be available
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            self.assertEqual(len(importance), self.X.shape[1])
            self.assertTrue(np.all(importance >= 0))
            self.assertAlmostEqual(np.sum(importance), 1.0, places=5)
        
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeRegressor(max_depth=1)
        
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
        import auroraml.tree as aml_tree
        
        model = aml_tree.DecisionTreeRegressor(max_depth=5)
        model.fit(self.X, self.y)
        
        # Save model
        filename = "test_dt_regressor.bin"
        model.save(filename)
        
        # Load model
        loaded_model = aml_tree.DecisionTreeRegressor()
        loaded_model.load(filename)
        
        # Compare predictions
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=10)
        
        # Clean up
        os.remove(filename)

class TestTreeIntegration(unittest.TestCase):
    """Integration tests for tree algorithms"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y_class = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.y_reg = self.X @ np.array([1.0, -2.0, 0.5, 1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        
    def test_classifier_vs_regressor(self):
        """Compare classifier and regressor behavior"""
        import auroraml.tree as aml_tree
        import auroraml.metrics as aml_metrics
        
        # Test classifier
        clf = aml_tree.DecisionTreeClassifier(max_depth=5)
        clf.fit(self.X, self.y_class)
        clf_pred = clf.predict(self.X)
        clf_accuracy = aml_metrics.accuracy_score(self.y_class, clf_pred)
        
        # Test regressor
        reg = aml_tree.DecisionTreeRegressor(max_depth=5)
        reg.fit(self.X, self.y_reg)
        reg_pred = reg.predict(self.X)
        reg_mse = aml_metrics.mean_squared_error(self.y_reg, reg_pred)
        
        self.assertGreater(clf_accuracy, 0.7)
        self.assertLess(reg_mse, 1.0)
        
    def test_depth_effect(self):
        """Test effect of max_depth parameter"""
        import auroraml.tree as aml_tree
        import auroraml.metrics as aml_metrics
        
        depths = [1, 3, 5, 10]
        accuracies = []
        
        for depth in depths:
            model = aml_tree.DecisionTreeClassifier(max_depth=depth)
            model.fit(self.X, self.y_class)
            predictions = model.predict(self.X)
            accuracy = aml_metrics.accuracy_score(self.y_class, predictions)
            accuracies.append(accuracy)
            
        # Deeper trees should generally perform better (or at least not worse)
        self.assertGreaterEqual(accuracies[-1], accuracies[0])
        
    def test_cross_validation_compatibility(self):
        """Test compatibility with cross-validation"""
        import auroraml.tree as aml_tree
        import auroraml.model_selection as aml_ms
        import auroraml.metrics as aml_metrics
        
        # Test classifier
        clf = aml_tree.DecisionTreeClassifier(max_depth=5)
        kfold = aml_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in kfold.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y_class[train_idx], self.y_class[val_idx]
            
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_val)
            accuracy = aml_metrics.accuracy_score(y_val, predictions)
            scores.append(accuracy)
            
        mean_score = np.mean(scores)
        self.assertGreater(mean_score, 0.5)
        
    def test_overfitting_behavior(self):
        """Test overfitting behavior with different depths"""
        import auroraml.tree as aml_tree
        import auroraml.metrics as aml_metrics
        
        # Split data for train/test
        split_idx = int(0.8 * len(self.X))
        X_train, X_test = self.X[:split_idx], self.X[split_idx:]
        y_train, y_test = self.y_class[:split_idx], self.y_class[split_idx:]
        
        # Test with different depths
        depths = [1, 3, 10]
        train_scores = []
        test_scores = []
        
        for depth in depths:
            model = aml_tree.DecisionTreeClassifier(max_depth=depth)
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_acc = aml_metrics.accuracy_score(y_train, train_pred)
            test_acc = aml_metrics.accuracy_score(y_test, test_pred)
            
            train_scores.append(train_acc)
            test_scores.append(test_acc)
            
        # Deeper trees should fit training data better
        self.assertGreaterEqual(train_scores[-1], train_scores[0])

if __name__ == '__main__':
    unittest.main()
