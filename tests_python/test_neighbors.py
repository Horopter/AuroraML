#!/usr/bin/env python3
"""
Test Suite for AuroraML Neighbors
Tests KNeighborsClassifier and KNeighborsRegressor algorithms
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestKNeighborsClassifier(unittest.TestCase):
    """Test KNeighborsClassifier algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create classification data
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.KNeighborsClassifier(n_neighbors=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.KNeighborsClassifier(n_neighbors=5)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        
    def test_different_k_values(self):
        """Test with different k values"""
        import auroraml.neighbors as aml_neighbors
        
        k_values = [1, 3, 5, 7]
        for k in k_values:
            model = aml_neighbors.KNeighborsClassifier(n_neighbors=k)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.KNeighborsClassifier(n_neighbors=5)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_neighbors', params)
        self.assertIn('weights', params)
        self.assertIn('algorithm', params)
        
        # Test parameter setting
        model.set_params({'n_neighbors': '7'})
        self.assertEqual(model.get_params()['n_neighbors'], "7")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.neighbors as aml_neighbors
        import auroraml.metrics as aml_metrics
        
        model = aml_neighbors.KNeighborsClassifier(n_neighbors=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        accuracy = aml_metrics.accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.7)
        
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.KNeighborsClassifier(n_neighbors=1)
        
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
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.KNeighborsClassifier(n_neighbors=5)
        model.fit(self.X, self.y)
        
        # Save model
        filename = "test_knn_classifier.bin"
        model.save(filename)
        
        # Load model
        loaded_model = aml_neighbors.KNeighborsClassifier()
        loaded_model.load(filename)
        
        # Compare predictions
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Clean up
        os.remove(filename)

class TestKNeighborsRegressor(unittest.TestCase):
    """Test KNeighborsRegressor algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create regression data
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = self.X @ np.array([1.0, -2.0, 0.5, 1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.KNeighborsRegressor(n_neighbors=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_different_k_values(self):
        """Test with different k values"""
        import auroraml.neighbors as aml_neighbors
        
        k_values = [1, 3, 5, 7]
        for k in k_values:
            model = aml_neighbors.KNeighborsRegressor(n_neighbors=k)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.KNeighborsRegressor(n_neighbors=5)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_neighbors', params)
        self.assertIn('weights', params)
        self.assertIn('algorithm', params)
        
        # Test parameter setting
        model.set_params({'n_neighbors': '7'})
        self.assertEqual(model.get_params()['n_neighbors'], "7")
        
    def test_performance(self):
        """Test model performance"""
        import auroraml.neighbors as aml_neighbors
        import auroraml.metrics as aml_metrics
        
        model = aml_neighbors.KNeighborsRegressor(n_neighbors=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        mse = aml_metrics.mean_squared_error(self.y, predictions)
        self.assertLess(mse, 1.0)
        
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.KNeighborsRegressor(n_neighbors=1)
        
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
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.KNeighborsRegressor(n_neighbors=5)
        model.fit(self.X, self.y)
        
        # Save model
        filename = "test_knn_regressor.bin"
        model.save(filename)
        
        # Load model
        loaded_model = aml_neighbors.KNeighborsRegressor()
        loaded_model.load(filename)
        
        # Compare predictions
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=10)
        
        # Clean up
        os.remove(filename)

class TestNeighborsIntegration(unittest.TestCase):
    """Integration tests for neighbors algorithms"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y_class = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.y_reg = self.X @ np.array([1.0, -2.0, 0.5, 1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        
    def test_classifier_vs_regressor(self):
        """Compare classifier and regressor behavior"""
        import auroraml.neighbors as aml_neighbors
        import auroraml.metrics as aml_metrics
        
        # Test classifier
        clf = aml_neighbors.KNeighborsClassifier(n_neighbors=5)
        clf.fit(self.X, self.y_class)
        clf_pred = clf.predict(self.X)
        clf_accuracy = aml_metrics.accuracy_score(self.y_class, clf_pred)
        
        # Test regressor
        reg = aml_neighbors.KNeighborsRegressor(n_neighbors=5)
        reg.fit(self.X, self.y_reg)
        reg_pred = reg.predict(self.X)
        reg_mse = aml_metrics.mean_squared_error(self.y_reg, reg_pred)
        
        self.assertGreater(clf_accuracy, 0.7)
        self.assertLess(reg_mse, 1.0)
        
    def test_cross_validation_compatibility(self):
        """Test compatibility with cross-validation"""
        import auroraml.neighbors as aml_neighbors
        import auroraml.model_selection as aml_ms
        import auroraml.metrics as aml_metrics
        
        # Test classifier
        clf = aml_neighbors.KNeighborsClassifier(n_neighbors=5)
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
        
    def test_performance_comparison(self):
        """Compare performance with different k values"""
        import auroraml.neighbors as aml_neighbors
        import auroraml.metrics as aml_metrics
        
        k_values = [1, 3, 5, 7, 10]
        accuracies = []
        
        for k in k_values:
            model = aml_neighbors.KNeighborsClassifier(n_neighbors=k)
            model.fit(self.X, self.y_class)
            predictions = model.predict(self.X)
            accuracy = aml_metrics.accuracy_score(self.y_class, predictions)
            accuracies.append(accuracy)
            
        # All k values should give reasonable performance
        for acc in accuracies:
            self.assertGreater(acc, 0.5)

if __name__ == '__main__':
    unittest.main()
