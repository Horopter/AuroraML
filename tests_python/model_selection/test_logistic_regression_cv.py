#!/usr/bin/env python3
"""
Test Suite for AuroraML LogisticRegressionCV
Tests LogisticRegressionCV with cross-validation
"""

import sys
import os
import unittest
import numpy as np
import random

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestLogisticRegressionCV(unittest.TestCase):
    """Test LogisticRegressionCV algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create binary classification data
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.model_selection as aml_ms
        
        model = aml_ms.LogisticRegressionCV(cv=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.model_selection as aml_ms
        
        model = aml_ms.LogisticRegressionCV(cv=5)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape[0], len(self.X_test))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        
    def test_custom_cv_folds(self):
        """Test with different CV fold values"""
        import auroraml.model_selection as aml_ms
        
        for cv in [3, 5, 10]:
            model = aml_ms.LogisticRegressionCV(cv=cv)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
    def test_custom_C_values(self):
        """Test with custom C values"""
        import auroraml.model_selection as aml_ms
        
        Cs = [0.01, 0.1, 1.0, 10.0]
        model = aml_ms.LogisticRegressionCV(cv=5, Cs=Cs)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
    def test_get_params(self):
        """Test parameter retrieval"""
        import auroraml.model_selection as aml_ms
        
        model = aml_ms.LogisticRegressionCV(cv=5)
        params = model.get_params()
        self.assertIn('cv', params)
        self.assertIn('scoring', params)
        
    def test_is_fitted(self):
        """Test fitted state"""
        import auroraml.model_selection as aml_ms
        
        model = aml_ms.LogisticRegressionCV(cv=5)
        self.assertFalse(model.is_fitted())
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())

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

