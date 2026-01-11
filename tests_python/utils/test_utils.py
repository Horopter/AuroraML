"""Tests for Utils module"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ingenuityml
import random

class TestUtils(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y = (np.random.randn(100) > 0).astype(np.int32)
        self.y_multiclass = (np.random.randint(0, 3, 100)).astype(np.int32)

    def test_multiclass_utilities(self):
        """Test multiclass utilities"""
        # Test is_multiclass
        is_multi = ingenuityml.utils.multiclass.is_multiclass(self.y_multiclass)
        self.assertTrue(is_multi)
        
        is_multi_binary = ingenuityml.utils.multiclass.is_multiclass(self.y)
        self.assertFalse(is_multi_binary)
        
        # Test unique_labels
        unique = ingenuityml.utils.multiclass.unique_labels(self.y_multiclass)
        self.assertGreater(len(unique), 2)
        
        # Test type_of_target
        target_type = ingenuityml.utils.multiclass.type_of_target(self.y)
        self.assertIn(target_type, ["binary", "multiclass", "continuous"])

    def test_resample(self):
        """Test resampling utilities"""
        # Test resample
        X_resampled, y_resampled = ingenuityml.utils.resample.resample(
            self.X, self.y.astype(np.float64), n_samples=50, random_state=42
        )
        self.assertEqual(X_resampled.shape[0], 50)
        self.assertEqual(y_resampled.shape[0], 50)
        
        # Test shuffle
        X_shuffled = self.X.copy()
        y_shuffled = self.y.astype(np.float64).copy()
        ingenuityml.utils.resample.shuffle(X_shuffled, y_shuffled, random_state=42)
        # Should have same shape
        self.assertEqual(X_shuffled.shape, self.X.shape)

    def test_validation(self):
        """Test validation utilities"""
        # Test check_finite
        is_finite = ingenuityml.utils.validation.check_finite(self.X)
        self.assertTrue(is_finite)
        
        # Test check_has_nan
        X_with_nan = self.X.copy()
        X_with_nan[0, 0] = np.nan
        has_nan = ingenuityml.utils.validation.check_has_nan(X_with_nan)
        self.assertTrue(has_nan)
        
        # Test check_has_inf
        X_with_inf = self.X.copy()
        X_with_inf[0, 0] = np.inf
        has_inf = ingenuityml.utils.validation.check_has_inf(X_with_inf)
        self.assertTrue(has_inf)

    def test_class_weight(self):
        """Test class weight utilities"""
        weights = ingenuityml.utils.class_weight.compute_class_weight("balanced", self.y)
        self.assertIsInstance(weights, dict)
        self.assertGreater(len(weights), 0)
        
        sample_weights = ingenuityml.utils.class_weight.compute_sample_weight(self.y, weights)
        self.assertIsInstance(sample_weights, dict)

    def test_array_utilities(self):
        """Test array utilities"""
        # Test issparse
        is_sparse = ingenuityml.utils.array.issparse(self.X)
        self.assertFalse(is_sparse)  # Eigen matrices are always dense
        
        # Test shape
        shape = ingenuityml.utils.array.shape(self.X)
        self.assertEqual(shape, (self.X.shape[0], self.X.shape[1]))

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

