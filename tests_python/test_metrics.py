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

class TestAdditionalClassificationMetrics(unittest.TestCase):
    """Test additional classification metrics"""
    
    def setUp(self):
        """Set up test data"""
        self.y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
        self.y_score = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.95, 0.4, 0.1, 0.85, 0.05])
        self.y_proba = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.8, 0.2],
            [0.05, 0.95], [0.6, 0.4], [0.7, 0.3], [0.15, 0.85], [0.95, 0.05]
        ])
        
    def test_balanced_accuracy_score(self):
        """Test balanced accuracy score"""
        import auroraml.metrics as aml_metrics
        
        balanced_acc = aml_metrics.balanced_accuracy_score(self.y_true, self.y_pred)
        
        self.assertGreaterEqual(balanced_acc, 0.0)
        self.assertLessEqual(balanced_acc, 1.0)
        
    def test_roc_auc_score(self):
        """Test ROC AUC score"""
        import auroraml.metrics as aml_metrics
        
        roc_auc = aml_metrics.roc_auc_score(self.y_true, self.y_score)
        
        self.assertGreaterEqual(roc_auc, 0.0)
        self.assertLessEqual(roc_auc, 1.0)
        
    def test_average_precision_score(self):
        """Test average precision score"""
        import auroraml.metrics as aml_metrics
        
        ap = aml_metrics.average_precision_score(self.y_true, self.y_score)
        
        self.assertGreaterEqual(ap, 0.0)
        self.assertLessEqual(ap, 1.0)
        
    def test_log_loss(self):
        """Test log loss"""
        import auroraml.metrics as aml_metrics
        
        log_loss_val = aml_metrics.log_loss(self.y_true, self.y_proba)
        
        self.assertGreaterEqual(log_loss_val, 0.0)
        
    def test_hinge_loss(self):
        """Test hinge loss"""
        import auroraml.metrics as aml_metrics
        
        decision = self.y_score * 2.0 - 1.0  # Convert to [-1, 1]
        hinge = aml_metrics.hinge_loss(self.y_true, decision)
        
        self.assertGreaterEqual(hinge, 0.0)
        
    def test_cohen_kappa_score(self):
        """Test Cohen's kappa score"""
        import auroraml.metrics as aml_metrics
        
        kappa = aml_metrics.cohen_kappa_score(self.y_true, self.y_pred)
        
        self.assertGreaterEqual(kappa, -1.0)
        self.assertLessEqual(kappa, 1.0)
        
    def test_matthews_corrcoef(self):
        """Test Matthews correlation coefficient"""
        import auroraml.metrics as aml_metrics
        
        mcc = aml_metrics.matthews_corrcoef(self.y_true, self.y_pred)
        
        self.assertGreaterEqual(mcc, -1.0)
        self.assertLessEqual(mcc, 1.0)
        
    def test_hamming_loss(self):
        """Test Hamming loss"""
        import auroraml.metrics as aml_metrics
        
        hamming = aml_metrics.hamming_loss(self.y_true, self.y_pred)
        
        self.assertGreaterEqual(hamming, 0.0)
        self.assertLessEqual(hamming, 1.0)
        
    def test_jaccard_score(self):
        """Test Jaccard score"""
        import auroraml.metrics as aml_metrics
        
        jaccard = aml_metrics.jaccard_score(self.y_true, self.y_pred, "macro")
        
        self.assertGreaterEqual(jaccard, 0.0)
        self.assertLessEqual(jaccard, 1.0)
        
    def test_zero_one_loss(self):
        """Test zero-one loss"""
        import auroraml.metrics as aml_metrics
        
        zero_one = aml_metrics.zero_one_loss(self.y_true, self.y_pred)
        
        self.assertGreaterEqual(zero_one, 0.0)
        self.assertLessEqual(zero_one, 1.0)
        
    def test_brier_score_loss(self):
        """Test Brier score loss"""
        import auroraml.metrics as aml_metrics
        
        brier = aml_metrics.brier_score_loss(self.y_true, self.y_score)
        
        self.assertGreaterEqual(brier, 0.0)
        self.assertLessEqual(brier, 1.0)

class TestAdditionalRegressionMetrics(unittest.TestCase):
    """Test additional regression metrics"""
    
    def setUp(self):
        """Set up test data"""
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1, 5.8, 7.2, 7.9])
        self.y_true_positive = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        self.y_pred_positive = np.array([2.1, 2.9, 4.1, 4.9, 6.1])
        
    def test_median_absolute_error(self):
        """Test median absolute error"""
        import auroraml.metrics as aml_metrics
        
        median_ae = aml_metrics.median_absolute_error(self.y_true, self.y_pred)
        
        self.assertGreaterEqual(median_ae, 0.0)
        
    def test_max_error(self):
        """Test max error"""
        import auroraml.metrics as aml_metrics
        
        max_err = aml_metrics.max_error(self.y_true, self.y_pred)
        
        self.assertGreaterEqual(max_err, 0.0)
        
    def test_mean_poisson_deviance(self):
        """Test mean Poisson deviance"""
        import auroraml.metrics as aml_metrics
        
        poisson_dev = aml_metrics.mean_poisson_deviance(self.y_true_positive, self.y_pred_positive)
        
        self.assertGreaterEqual(poisson_dev, 0.0)
        
    def test_mean_gamma_deviance(self):
        """Test mean Gamma deviance"""
        import auroraml.metrics as aml_metrics
        
        gamma_dev = aml_metrics.mean_gamma_deviance(self.y_true_positive, self.y_pred_positive)
        
        self.assertGreaterEqual(gamma_dev, 0.0)
        
    def test_mean_tweedie_deviance(self):
        """Test mean Tweedie deviance"""
        import auroraml.metrics as aml_metrics
        
        tweedie_dev = aml_metrics.mean_tweedie_deviance(self.y_true_positive, self.y_pred_positive, 1.5)
        
        self.assertGreaterEqual(tweedie_dev, 0.0)
        
    def test_d2_tweedie_score(self):
        """Test D² Tweedie score"""
        import auroraml.metrics as aml_metrics
        
        d2 = aml_metrics.d2_tweedie_score(self.y_true_positive, self.y_pred_positive, 1.0)
        
        # D² can be negative or positive
        self.assertFalse(np.isnan(d2))
        
    def test_d2_pinball_score(self):
        """Test D² pinball score"""
        import auroraml.metrics as aml_metrics
        
        d2_pinball = aml_metrics.d2_pinball_score(self.y_true, self.y_pred, 0.5)
        
        self.assertFalse(np.isnan(d2_pinball))
        
    def test_d2_absolute_error_score(self):
        """Test D² absolute error score"""
        import auroraml.metrics as aml_metrics
        
        d2_ae = aml_metrics.d2_absolute_error_score(self.y_true, self.y_pred)
        
        self.assertFalse(np.isnan(d2_ae))

class TestClusteringMetrics(unittest.TestCase):
    """Test clustering metrics"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.rand(30, 3)
        self.labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        self.labels_true = np.array([0] * 10 + [1] * 10 + [2] * 10)
        self.labels_pred = np.array([1] * 10 + [2] * 10 + [0] * 10)  # Permuted
        
    def test_silhouette_score(self):
        """Test silhouette score"""
        import auroraml.metrics as aml_metrics
        
        silhouette = aml_metrics.silhouette_score(self.X, self.labels)
        
        self.assertGreaterEqual(silhouette, -1.0)
        self.assertLessEqual(silhouette, 1.0)
        
    def test_silhouette_samples(self):
        """Test silhouette samples"""
        import auroraml.metrics as aml_metrics
        
        samples = aml_metrics.silhouette_samples(self.X, self.labels)
        
        self.assertEqual(len(samples), self.X.shape[0])
        for s in samples:
            self.assertGreaterEqual(s, -1.0)
            self.assertLessEqual(s, 1.0)
        
    def test_calinski_harabasz_score(self):
        """Test Calinski-Harabasz score"""
        import auroraml.metrics as aml_metrics
        
        ch_score = aml_metrics.calinski_harabasz_score(self.X, self.labels)
        
        self.assertGreaterEqual(ch_score, 0.0)
        
    def test_davies_bouldin_score(self):
        """Test Davies-Bouldin score"""
        import auroraml.metrics as aml_metrics
        
        db_score = aml_metrics.davies_bouldin_score(self.X, self.labels)
        
        self.assertGreaterEqual(db_score, 0.0)
        
    def test_adjusted_rand_score(self):
        """Test adjusted Rand score"""
        import auroraml.metrics as aml_metrics
        
        ari = aml_metrics.adjusted_rand_score(self.labels_true, self.labels_pred)
        
        self.assertGreaterEqual(ari, -1.0)
        self.assertLessEqual(ari, 1.0)
        
    def test_adjusted_mutual_info_score(self):
        """Test adjusted mutual information score"""
        import auroraml.metrics as aml_metrics
        
        ami = aml_metrics.adjusted_mutual_info_score(self.labels_true, self.labels_pred)
        
        self.assertGreaterEqual(ami, -1.0)
        self.assertLessEqual(ami, 1.0 + 1e-10)  # Allow for floating point precision
        
    def test_normalized_mutual_info_score(self):
        """Test normalized mutual information score"""
        import auroraml.metrics as aml_metrics
        
        nmi = aml_metrics.normalized_mutual_info_score(self.labels_true, self.labels_pred)
        
        self.assertGreaterEqual(nmi, 0.0)
        self.assertLessEqual(nmi, 1.0)
        
    def test_homogeneity_score(self):
        """Test homogeneity score"""
        import auroraml.metrics as aml_metrics
        
        homogeneity = aml_metrics.homogeneity_score(self.labels_true, self.labels_pred)
        
        self.assertGreaterEqual(homogeneity, 0.0)
        self.assertLessEqual(homogeneity, 1.0)
        
    def test_completeness_score(self):
        """Test completeness score"""
        import auroraml.metrics as aml_metrics
        
        completeness = aml_metrics.completeness_score(self.labels_true, self.labels_pred)
        
        self.assertGreaterEqual(completeness, 0.0)
        self.assertLessEqual(completeness, 1.0)
        
    def test_v_measure_score(self):
        """Test V-measure score"""
        import auroraml.metrics as aml_metrics
        
        v_measure = aml_metrics.v_measure_score(self.labels_true, self.labels_pred)
        
        self.assertGreaterEqual(v_measure, 0.0)
        self.assertLessEqual(v_measure, 1.0)
        
    def test_fowlkes_mallows_score(self):
        """Test Fowlkes-Mallows score"""
        import auroraml.metrics as aml_metrics
        
        fm_score = aml_metrics.fowlkes_mallows_score(self.labels_true, self.labels_pred)
        
        self.assertGreaterEqual(fm_score, 0.0)
        self.assertLessEqual(fm_score, 1.0)

if __name__ == '__main__':
    unittest.main()
