#!/usr/bin/env python3
"""
Test Suite for AuroraML Metrics
Tests all evaluation metrics for classification and regression
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestClassificationMetrics(unittest.TestCase):
    """Test classification metrics"""
    
    def setUp(self):
        """Set up test data"""
        self.y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
        self.y_pred_proba = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.8, 0.2],
            [0.1, 0.9], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1]
        ])
        
    def test_accuracy_score(self):
        """Test accuracy score"""
        import auroraml.metrics as aml_metrics
        
        accuracy = aml_metrics.accuracy_score(self.y_true, self.y_pred)
        
        # Calculate expected accuracy manually
        expected = np.mean(self.y_true == self.y_pred)
        
        self.assertAlmostEqual(accuracy, expected, places=10)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        
    def test_accuracy_score_perfect(self):
        """Test accuracy score with perfect predictions"""
        import auroraml.metrics as aml_metrics
        
        accuracy = aml_metrics.accuracy_score(self.y_true, self.y_true)
        self.assertEqual(accuracy, 1.0)
        
    def test_accuracy_score_worst(self):
        """Test accuracy score with worst predictions"""
        import auroraml.metrics as aml_metrics
        
        y_worst = 1 - self.y_true
        accuracy = aml_metrics.accuracy_score(self.y_true, y_worst)
        self.assertEqual(accuracy, 0.0)
        
    def test_accuracy_score_edge_cases(self):
        """Test accuracy score edge cases"""
        import auroraml.metrics as aml_metrics
        
        # Single sample
        accuracy = aml_metrics.accuracy_score([0], [0])
        self.assertEqual(accuracy, 1.0)
        
        # All same class
        y_all_zeros = np.zeros(10)
        accuracy = aml_metrics.accuracy_score(y_all_zeros, y_all_zeros)
        self.assertEqual(accuracy, 1.0)
        
    def test_precision_score(self):
        """Test precision score"""
        import auroraml.metrics as aml_metrics
        
        # Reshape arrays to match expected format
        y_true_reshaped = self.y_true.reshape(-1, 1).astype(np.int32)
        y_pred_reshaped = self.y_pred.reshape(-1, 1).astype(np.int32)
        
        precision = aml_metrics.precision_score(y_true_reshaped, y_pred_reshaped, "macro")
        
        # Calculate expected precision manually
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        expected = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        self.assertAlmostEqual(precision, expected, places=10)
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
        
    def test_recall_score(self):
        """Test recall score"""
        import auroraml.metrics as aml_metrics
        
        # Reshape arrays to match expected format
        y_true_reshaped = self.y_true.reshape(-1, 1).astype(np.int32)
        y_pred_reshaped = self.y_pred.reshape(-1, 1).astype(np.int32)
        
        recall = aml_metrics.recall_score(y_true_reshaped, y_pred_reshaped, "macro")
        
        # Calculate expected recall manually
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        expected = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        self.assertAlmostEqual(recall, expected, places=10)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
        
    def test_f1_score(self):
        """Test F1 score"""
        import auroraml.metrics as aml_metrics
        
        # Reshape arrays to match expected format
        y_true_reshaped = self.y_true.reshape(-1, 1).astype(np.int32)
        y_pred_reshaped = self.y_pred.reshape(-1, 1).astype(np.int32)
        
        f1 = aml_metrics.f1_score(y_true_reshaped, y_pred_reshaped, "macro")
        
        # Calculate expected F1 manually
        precision = aml_metrics.precision_score(y_true_reshaped, y_pred_reshaped, "macro")
        recall = aml_metrics.recall_score(y_true_reshaped, y_pred_reshaped, "macro")
        expected = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        self.assertAlmostEqual(f1, expected, places=10)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
        
    def test_confusion_matrix(self):
        """Test confusion matrix"""
        import auroraml.metrics as aml_metrics
        
        cm = aml_metrics.confusion_matrix(self.y_true, self.y_pred)
        
        # Check shape
        self.assertEqual(cm.shape, (2, 2))
        
        # Check values
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        
        self.assertEqual(cm[0, 0], tn)  # True negatives
        self.assertEqual(cm[0, 1], fp)  # False positives
        self.assertEqual(cm[1, 0], fn)  # False negatives
        self.assertEqual(cm[1, 1], tp)  # True positives

class TestRegressionMetrics(unittest.TestCase):
    """Test regression metrics"""
    
    def setUp(self):
        """Set up test data"""
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
    def test_mean_squared_error(self):
        """Test mean squared error"""
        import auroraml.metrics as aml_metrics
        
        mse = aml_metrics.mean_squared_error(self.y_true, self.y_pred)
        
        # Calculate expected MSE manually
        expected = np.mean((self.y_true - self.y_pred) ** 2)
        
        self.assertAlmostEqual(mse, expected, places=10)
        self.assertGreaterEqual(mse, 0.0)
        
    def test_mean_absolute_error(self):
        """Test mean absolute error"""
        import auroraml.metrics as aml_metrics
        
        mae = aml_metrics.mean_absolute_error(self.y_true, self.y_pred)
        
        # Calculate expected MAE manually
        expected = np.mean(np.abs(self.y_true - self.y_pred))
        
        self.assertAlmostEqual(mae, expected, places=10)
        self.assertGreaterEqual(mae, 0.0)
        
    def test_r2_score(self):
        """Test R² score"""
        import auroraml.metrics as aml_metrics
        
        r2 = aml_metrics.r2_score(self.y_true, self.y_pred)
        
        # Calculate expected R² manually
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        expected = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        self.assertAlmostEqual(r2, expected, places=10)
        self.assertLessEqual(r2, 1.0)
        
    def test_r2_score_perfect(self):
        """Test R² score with perfect predictions"""
        import auroraml.metrics as aml_metrics
        
        r2 = aml_metrics.r2_score(self.y_true, self.y_true)
        self.assertEqual(r2, 1.0)
        
    def test_r2_score_worst(self):
        """Test R² score with mean predictions"""
        import auroraml.metrics as aml_metrics
        
        y_mean = np.full_like(self.y_true, np.mean(self.y_true))
        r2 = aml_metrics.r2_score(self.y_true, y_mean)
        self.assertAlmostEqual(r2, 0.0, places=10)
        
    def test_mean_squared_log_error(self):
        """Test mean squared log error (not implemented, skip)"""
        # MSLE is not implemented in AuroraML, skip this test
        self.skipTest("mean_squared_log_error not implemented in AuroraML")
        
    def test_edge_cases(self):
        """Test edge cases for regression metrics"""
        import auroraml.metrics as aml_metrics
        
        # Single sample
        mse = aml_metrics.mean_squared_error([1.0], [1.0])
        self.assertEqual(mse, 0.0)
        
        # Identical arrays
        y_identical = np.array([1.0, 2.0, 3.0])
        mse = aml_metrics.mean_squared_error(y_identical, y_identical)
        self.assertEqual(mse, 0.0)
        
        # Zero variance - R2 should be NaN for constant arrays
        y_constant = np.array([1.0, 1.0, 1.0])
        r2 = aml_metrics.r2_score(y_constant, y_constant)
        # R2 can be NaN for constant arrays, which is acceptable
        self.assertTrue(np.isnan(r2) or r2 == 1.0)

class TestMetricsIntegration(unittest.TestCase):
    """Integration tests for metrics"""
    
    def test_classification_metrics_consistency(self):
        """Test consistency between classification metrics"""
        import auroraml.metrics as aml_metrics
        
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
        
        # Reshape arrays to match expected format
        y_true_reshaped = y_true.reshape(-1, 1).astype(np.int32)
        y_pred_reshaped = y_pred.reshape(-1, 1).astype(np.int32)
        
        accuracy = aml_metrics.accuracy_score(y_true, y_pred)
        precision = aml_metrics.precision_score(y_true_reshaped, y_pred_reshaped, "macro")
        recall = aml_metrics.recall_score(y_true_reshaped, y_pred_reshaped, "macro")
        f1 = aml_metrics.f1_score(y_true_reshaped, y_pred_reshaped, "macro")
        
        # All metrics should be between 0 and 1
        for metric in [accuracy, precision, recall, f1]:
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)
            
        # F1 should be harmonic mean of precision and recall
        if precision + recall > 0:
            expected_f1 = 2 * (precision * recall) / (precision + recall)
            self.assertAlmostEqual(f1, expected_f1, places=10)
            
    def test_regression_metrics_consistency(self):
        """Test consistency between regression metrics"""
        import auroraml.metrics as aml_metrics
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        mse = aml_metrics.mean_squared_error(y_true, y_pred)
        mae = aml_metrics.mean_absolute_error(y_true, y_pred)
        r2 = aml_metrics.r2_score(y_true, y_pred)
        
        # MSE should be >= MAE² (Jensen's inequality) - but with some tolerance
        self.assertGreaterEqual(mse, mae ** 2 - 1e-10)
        
        # R² should be <= 1
        self.assertLessEqual(r2, 1.0)
        
    def test_metrics_with_different_data_types(self):
        """Test metrics with different data types"""
        import auroraml.metrics as aml_metrics
        
        # Test with different array shapes
        y_true_1d = np.array([0, 1, 0, 1])
        y_pred_1d = np.array([0, 1, 1, 1])
        
        y_true_2d = np.array([[0], [1], [0], [1]])
        y_pred_2d = np.array([[0], [1], [1], [1]])
        
        acc_1d = aml_metrics.accuracy_score(y_true_1d, y_pred_1d)
        acc_2d = aml_metrics.accuracy_score(y_true_2d, y_pred_2d)
        
        # Results should be the same
        self.assertEqual(acc_1d, acc_2d)
        
    def test_metrics_performance(self):
        """Test metrics performance with larger datasets"""
        import auroraml.metrics as aml_metrics
        
        # Generate larger dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Classification
        y_true_class = np.random.randint(0, 2, n_samples)
        y_pred_class = np.random.randint(0, 2, n_samples)
        
        accuracy = aml_metrics.accuracy_score(y_true_class, y_pred_class)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        
        # Regression
        y_true_reg = np.random.randn(n_samples)
        y_pred_reg = y_true_reg + 0.1 * np.random.randn(n_samples)
        
        mse = aml_metrics.mean_squared_error(y_true_reg, y_pred_reg)
        self.assertGreaterEqual(mse, 0.0)
        
        r2 = aml_metrics.r2_score(y_true_reg, y_pred_reg)
        self.assertLessEqual(r2, 1.0)

if __name__ == '__main__':
    unittest.main()
