#!/usr/bin/env python3
"""
Test Suite for AuroraML RidgeClassifier and RidgeClassifierCV
Tests Ridge-based classification algorithms
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestRidgeClassifier(unittest.TestCase):
    """Test RidgeClassifier algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.RidgeClassifier(alpha=1.0)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.RidgeClassifier(alpha=1.0)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape[0], len(self.X_test))
        self.assertEqual(probabilities.shape[1], 2)  # Binary classification
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        
    def test_different_alpha_values(self):
        """Test with different alpha values"""
        import auroraml.linear_model as aml_lm
        
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            model = aml_lm.RidgeClassifier(alpha=alpha)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_coef_and_intercept(self):
        """Test coefficient and intercept access"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.RidgeClassifier(alpha=1.0)
        model.fit(self.X, self.y)
        
        coef = model.coef()
        intercept = model.intercept()
        
        self.assertEqual(len(coef), self.X.shape[1])
        self.assertIsInstance(intercept, (float, np.floating))
        
    def test_get_params(self):
        """Test parameter retrieval"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.RidgeClassifier(alpha=2.0, fit_intercept=True)
        params = model.get_params()
        self.assertIn('alpha', params)
        self.assertIn('fit_intercept', params)
        
    def test_is_fitted(self):
        """Test fitted state"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.RidgeClassifier()
        self.assertFalse(model.is_fitted())
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())

class TestRidgeClassifierCV(unittest.TestCase):
    """Test RidgeClassifierCV algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.RidgeClassifierCV(cv=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.RidgeClassifierCV(cv=5)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape[0], len(self.X_test))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        
    def test_custom_cv_folds(self):
        """Test with different CV fold values"""
        import auroraml.linear_model as aml_lm
        
        for cv in [3, 5, 10]:
            model = aml_lm.RidgeClassifierCV(cv=cv)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_custom_alpha_values(self):
        """Test with custom alpha values"""
        import auroraml.linear_model as aml_lm
        
        alphas = [0.1, 1.0, 10.0, 100.0]
        model = aml_lm.RidgeClassifierCV(cv=5, alphas=alphas)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
    def test_get_params(self):
        """Test parameter retrieval"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.RidgeClassifierCV(cv=5)
        params = model.get_params()
        self.assertIn('cv', params)
        self.assertIn('scoring', params)
        
    def test_is_fitted(self):
        """Test fitted state"""
        import auroraml.linear_model as aml_lm
        
        model = aml_lm.RidgeClassifierCV(cv=5)
        self.assertFalse(model.is_fitted())
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())

if __name__ == '__main__':
    unittest.main()

