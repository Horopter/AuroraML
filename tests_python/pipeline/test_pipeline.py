"""Tests for Pipeline and FeatureUnion modules"""
import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ingenuityml
import random

class TestPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (np.random.randn(100) > 0).astype(np.float64)
        self.y_class = (np.random.randn(100) > 0).astype(np.int32)

    def test_pipeline_classification(self):
        """Test Pipeline with classifier"""
        scaler = ingenuityml.preprocessing.StandardScaler()
        clf = ingenuityml.tree.DecisionTreeClassifier(max_depth=3)
        
        steps = [
            ("scaler", scaler),
            ("classifier", clf)
        ]
        pipeline = ingenuityml.pipeline.Pipeline(steps)
        
        pipeline.fit(self.X, self.y_class)
        self.assertTrue(pipeline.is_fitted())
        
        predictions = pipeline.predict_classes(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        
        proba = pipeline.predict_proba(self.X)
        self.assertEqual(proba.shape, (self.X.shape[0], 2))

    def test_pipeline_regression(self):
        """Test Pipeline with regressor"""
        scaler = ingenuityml.preprocessing.StandardScaler()
        reg = ingenuityml.linear_model.LinearRegression()
        
        steps = [
            ("scaler", scaler),
            ("regressor", reg)
        ]
        pipeline = ingenuityml.pipeline.Pipeline(steps)
        
        pipeline.fit(self.X, self.y)
        predictions = pipeline.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

    def test_pipeline_transform(self):
        """Test Pipeline transform method"""
        scaler1 = ingenuityml.preprocessing.StandardScaler()
        scaler2 = ingenuityml.preprocessing.MinMaxScaler()
        
        steps = [
            ("scaler1", scaler1),
            ("scaler2", scaler2)
        ]
        pipeline = ingenuityml.pipeline.Pipeline(steps)
        
        X_transformed = pipeline.fit_transform(self.X, self.y)
        self.assertEqual(X_transformed.shape, self.X.shape)

    def test_feature_union(self):
        """Test FeatureUnion"""
        scaler1 = ingenuityml.preprocessing.StandardScaler()
        scaler2 = ingenuityml.preprocessing.MinMaxScaler()
        
        transformers = [
            ("scaler1", scaler1),
            ("scaler2", scaler2)
        ]
        union = ingenuityml.pipeline.FeatureUnion(transformers)
        
        X_transformed = union.fit_transform(self.X, self.y)
        # Should have double the columns
        self.assertEqual(X_transformed.shape[1], self.X.shape[1] * 2)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])

    def test_feature_union_get_transformer(self):
        """Test FeatureUnion get_transformer"""
        scaler = ingenuityml.preprocessing.StandardScaler()
        transformers = [("scaler", scaler)]
        union = ingenuityml.pipeline.FeatureUnion(transformers)
        
        union.fit(self.X, self.y)
        retrieved = union.get_transformer("scaler")
        self.assertIsNotNone(retrieved)
        
        names = union.get_transformer_names()
        self.assertIn("scaler", names)

    def test_pipeline_get_step(self):
        """Test Pipeline get_step"""
        scaler = ingenuityml.preprocessing.StandardScaler()
        clf = ingenuityml.tree.DecisionTreeClassifier()
        steps = [("scaler", scaler), ("classifier", clf)]
        pipeline = ingenuityml.pipeline.Pipeline(steps)
        
        pipeline.fit(self.X, self.y_class)
        retrieved = pipeline.get_step("scaler")
        self.assertIsNotNone(retrieved)
        
        names = pipeline.get_step_names()
        self.assertIn("scaler", names)
        self.assertIn("classifier", names)

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

