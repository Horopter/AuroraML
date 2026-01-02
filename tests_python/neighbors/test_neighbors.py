#!/usr/bin/env python3
"""
Test Suite for AuroraML Neighbors
Tests KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, and RadiusNeighborsRegressor
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

class TestRadiusNeighborsClassifier(unittest.TestCase):
    """Test RadiusNeighborsClassifier algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.RadiusNeighborsClassifier(radius=1.0)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    def test_predict_proba(self):
        """Test probability prediction"""
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.RadiusNeighborsClassifier(radius=1.0)
        model.fit(self.X, self.y)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape[0], len(self.X_test))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
        
    def test_different_radius_values(self):
        """Test with different radius values"""
        import auroraml.neighbors as aml_neighbors
        
        for radius in [0.5, 1.0, 2.0]:
            model = aml_neighbors.RadiusNeighborsClassifier(radius=radius)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))

class TestRadiusNeighborsRegressor(unittest.TestCase):
    """Test RadiusNeighborsRegressor algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (self.X[:, 0] + self.X[:, 1] + np.random.randn(100) * 0.1).astype(np.float64)
        self.X_test = np.random.randn(20, 4).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        import auroraml.neighbors as aml_neighbors
        
        model = aml_neighbors.RadiusNeighborsRegressor(radius=1.0)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue(np.all(np.isfinite(predictions)))
        
    def test_different_radius_values(self):
        """Test with different radius values"""
        import auroraml.neighbors as aml_neighbors
        
        for radius in [0.5, 1.0, 2.0]:
            model = aml_neighbors.RadiusNeighborsRegressor(radius=radius)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))

if __name__ == '__main__':
    unittest.main()

