"""Tests for Feature Selection module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auroraml
import random

class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 10).astype(np.float64)
        self.y = (np.random.randn(100) > 0).astype(np.float64)
        self.y_class = (np.random.randn(100) > 0).astype(np.int32)

    def test_variance_threshold(self):
        """Test VarianceThreshold"""
        # Create data with low variance feature
        X = self.X.copy()
        X[:, 0] = 1.0  # Constant feature
        
        vt = auroraml.feature_selection.VarianceThreshold(threshold=0.01)
        X_transformed = vt.fit_transform(X, self.y)
        
        # Should remove constant feature
        self.assertEqual(X_transformed.shape[1], X.shape[1] - 1)
        support = vt.get_support()
        self.assertEqual(len(support), X.shape[1] - 1)

    def test_select_k_best(self):
        """Test SelectKBest"""
        def score_func(X_feature, y):
            # Simple correlation-based score
            corr = np.corrcoef(X_feature, y)[0, 1]
            return np.abs(corr) if not np.isnan(corr) else 0.0
        
        selector = auroraml.feature_selection.SelectKBest(score_func=score_func, k=3)
        X_transformed = selector.fit_transform(self.X, self.y)
        
        self.assertEqual(X_transformed.shape[1], 3)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        
        scores = selector.scores()
        self.assertEqual(len(scores), self.X.shape[1])

    def test_select_percentile(self):
        """Test SelectPercentile"""
        def score_func(X_feature, y):
            corr = np.corrcoef(X_feature, y)[0, 1]
            return np.abs(corr) if not np.isnan(corr) else 0.0
        
        selector = auroraml.feature_selection.SelectPercentile(
            score_func=score_func, percentile=30
        )
        X_transformed = selector.fit_transform(self.X, self.y)
        
        # Should select top 30% of features
        expected_features = max(1, int(0.3 * self.X.shape[1]))
        self.assertEqual(X_transformed.shape[1], expected_features)

    def test_select_fpr(self):
        """Test SelectFpr"""
        selector = auroraml.feature_selection.SelectFpr(
            score_func=auroraml.feature_selection.scores.f_classif, alpha=0.1
        )
        X_transformed = selector.fit_transform(self.X, self.y_class)

        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertGreater(X_transformed.shape[1], 0)
        self.assertLessEqual(X_transformed.shape[1], self.X.shape[1])
        self.assertEqual(len(selector.scores()), self.X.shape[1])

    def test_select_fdr(self):
        """Test SelectFdr"""
        selector = auroraml.feature_selection.SelectFdr(
            score_func=auroraml.feature_selection.scores.f_classif, alpha=0.1
        )
        X_transformed = selector.fit_transform(self.X, self.y_class)

        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertGreater(X_transformed.shape[1], 0)
        self.assertLessEqual(X_transformed.shape[1], self.X.shape[1])
        self.assertEqual(len(selector.scores()), self.X.shape[1])

    def test_select_fwe(self):
        """Test SelectFwe"""
        selector = auroraml.feature_selection.SelectFwe(
            score_func=auroraml.feature_selection.scores.f_classif, alpha=0.1
        )
        X_transformed = selector.fit_transform(self.X, self.y_class)

        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertGreater(X_transformed.shape[1], 0)
        self.assertLessEqual(X_transformed.shape[1], self.X.shape[1])
        self.assertEqual(len(selector.scores()), self.X.shape[1])

    def test_generic_univariate_select(self):
        """Test GenericUnivariateSelect"""
        selector = auroraml.feature_selection.GenericUnivariateSelect(
            score_func=auroraml.feature_selection.scores.f_classif, mode="percentile", param=20.0
        )
        X_transformed = selector.fit_transform(self.X, self.y_class)

        expected_features = max(1, int(0.2 * self.X.shape[1]))
        self.assertEqual(X_transformed.shape[1], expected_features)

    def test_select_from_model(self):
        """Test SelectFromModel"""
        estimator = auroraml.linear_model.LinearRegression()
        selector = auroraml.feature_selection.SelectFromModel(
            estimator=estimator, threshold=0.0, max_features=3
        )
        X_transformed = selector.fit_transform(self.X, self.y)

        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertGreater(X_transformed.shape[1], 0)
        self.assertLessEqual(X_transformed.shape[1], 3)
        self.assertEqual(len(selector.importances()), self.X.shape[1])

    def test_rfe(self):
        """Test RFE"""
        estimator = auroraml.linear_model.LinearRegression()
        selector = auroraml.feature_selection.RFE(
            estimator=estimator, n_features_to_select=3, step=2
        )
        X_transformed = selector.fit_transform(self.X, self.y)

        self.assertEqual(X_transformed.shape[1], 3)

    def test_rfecv(self):
        """Test RFECV"""
        X_small = self.X[:40, :6]
        y_small = self.y_class[:40]

        estimator = auroraml.linear_model.LogisticRegression()
        cv = auroraml.model_selection.KFold(n_splits=3, shuffle=True, random_state=42)
        selector = auroraml.feature_selection.RFECV(
            estimator=estimator, cv=cv, step=1, scoring="accuracy", min_features_to_select=2
        )
        X_transformed = selector.fit_transform(X_small, y_small)

        self.assertGreaterEqual(X_transformed.shape[1], 2)
        self.assertLessEqual(X_transformed.shape[1], X_small.shape[1])

    def test_sequential_feature_selector(self):
        """Test SequentialFeatureSelector"""
        X_small = self.X[:40, :6]
        y_small = self.y_class[:40]

        estimator = auroraml.linear_model.LogisticRegression()
        cv = auroraml.model_selection.KFold(n_splits=3, shuffle=True, random_state=42)
        selector = auroraml.feature_selection.SequentialFeatureSelector(
            estimator=estimator, cv=cv, n_features_to_select=3, direction="forward", scoring="accuracy"
        )
        X_transformed = selector.fit_transform(X_small, y_small)

        self.assertEqual(X_transformed.shape[1], 3)

    def test_scoring_functions(self):
        """Test scoring functions"""
        X_feature = self.X[:, 0]
        y_class = self.y_class.astype(np.int32)
        
        # Test f_classif
        score1 = auroraml.feature_selection.scores.f_classif(X_feature, y_class)
        self.assertIsInstance(score1, float)
        
        # Test f_regression
        score2 = auroraml.feature_selection.scores.f_regression(X_feature, self.y)
        self.assertIsInstance(score2, float)
        
        # Test chi2
        score3 = auroraml.feature_selection.scores.chi2(X_feature, y_class)
        self.assertIsInstance(score3, float)

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
