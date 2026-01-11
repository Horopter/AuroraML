"""Tests for Compose module (ColumnTransformer, TransformedTargetRegressor)"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ingenuityml
import random

class TestCompose(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = np.random.randn(100).astype(np.float64)

    def test_column_transformer(self):
        """Test ColumnTransformer"""
        scaler1 = ingenuityml.preprocessing.StandardScaler()
        scaler2 = ingenuityml.preprocessing.MinMaxScaler()
        
        transformers = [
            ("scaler1", scaler1, [0, 1]),
            ("scaler2", scaler2, [2, 3])
        ]
        ct = ingenuityml.compose.ColumnTransformer(transformers, remainder="drop")
        
        ct.fit(self.X, self.y)
        X_transformed = ct.transform(self.X)
        
        # Should have 4 columns (2 + 2)
        self.assertEqual(X_transformed.shape[1], 4)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])

    def test_column_transformer_remainder(self):
        """Test ColumnTransformer with remainder passthrough"""
        scaler = ingenuityml.preprocessing.StandardScaler()
        transformers = [("scaler", scaler, [0])]
        ct = ingenuityml.compose.ColumnTransformer(transformers, remainder="passthrough")
        
        ct.fit(self.X, self.y)
        X_transformed = ct.transform(self.X)
        
        # Should have 4 columns (1 transformed + 3 passthrough)
        self.assertEqual(X_transformed.shape[1], 4)

    def test_transformed_target_regressor(self):
        """Test TransformedTargetRegressor"""
        regressor = ingenuityml.linear_model.LinearRegression()
        transformer = ingenuityml.preprocessing.StandardScaler()
        
        ttr = ingenuityml.compose.TransformedTargetRegressor(
            regressor=regressor,
            transformer=transformer
        )
        
        ttr.fit(self.X, self.y)
        predictions = ttr.predict(self.X)
        
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        self.assertTrue(ttr.is_fitted())

    def test_transformed_target_regressor_no_transformer(self):
        """Test TransformedTargetRegressor without transformer"""
        regressor = ingenuityml.linear_model.LinearRegression()
        ttr = ingenuityml.compose.TransformedTargetRegressor(regressor=regressor)
        
        ttr.fit(self.X, self.y)
        predictions = ttr.predict(self.X)
        
        self.assertEqual(predictions.shape[0], self.X.shape[0])

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

