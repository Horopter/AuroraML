#!/usr/bin/env python3
import random
"""
Test Suite for AuroraML Preprocessing
Tests StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestStandardScaler(unittest.TestCase):
    """Test StandardScaler algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.StandardScaler()
        X_scaled = scaler.fit_transform(self.X, self.y_dummy)
        
        self.assertEqual(X_scaled.shape, self.X.shape)
        self.assertIsInstance(X_scaled, np.ndarray)
        
    def test_fit_transform(self):
        """Test fit_transform method"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.StandardScaler()
        X_scaled = scaler.fit_transform(self.X, self.y_dummy)
        
        # Check that data is approximately standardized
        mean_scaled = np.mean(X_scaled, axis=0)
        std_scaled = np.std(X_scaled, axis=0)
        
        np.testing.assert_array_almost_equal(mean_scaled, np.zeros(4), decimal=1)
        np.testing.assert_array_almost_equal(std_scaled, np.ones(4), decimal=1)
        
    def test_inverse_transform(self):
        """Test inverse transform"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.StandardScaler()
        X_scaled = scaler.fit_transform(self.X, self.y_dummy)
        X_inverse = scaler.inverse_transform(X_scaled)
        
        np.testing.assert_array_almost_equal(X_inverse, self.X, decimal=10)
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.StandardScaler()
        
        # Test default parameters
        params = scaler.get_params()
        self.assertIn('with_mean', params)
        self.assertIn('with_std', params)
        
        # Test parameter setting
        scaler.set_params(with_mean=False)
        self.assertEqual(scaler.get_params()['with_mean'], "false")
        
    def test_with_mean_false(self):
        """Test StandardScaler with with_mean=False"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(self.X, self.y_dummy)
        
        # Should still be fitted
        self.assertTrue(hasattr(scaler, 'mean_'))
        self.assertTrue(hasattr(scaler, 'scale_'))
        
        # Inverse transform should work
        X_inverse = scaler.inverse_transform(X_scaled)
        self.assertEqual(X_inverse.shape, self.X.shape)

class TestMinMaxScaler(unittest.TestCase):
    """Test MinMaxScaler algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X, self.y_dummy)
        
        self.assertEqual(X_scaled.shape, self.X.shape)
        self.assertIsInstance(X_scaled, np.ndarray)
        
    def test_fit_transform(self):
        """Test fit_transform method"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X, self.y_dummy)
        
        # Check that data is in [0, 1] range
        self.assertTrue(np.all(X_scaled >= 0))
        self.assertTrue(np.all(X_scaled <= 1))
        
    def test_inverse_transform(self):
        """Test inverse transform"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X, self.y_dummy)
        X_inverse = scaler.inverse_transform(X_scaled)
        
        np.testing.assert_array_almost_equal(X_inverse, self.X, decimal=10)
        
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.MinMaxScaler()
        
        # Test default parameters
        params = scaler.get_params()
        self.assertIn('feature_range_min', params)
        self.assertIn('feature_range_max', params)
        
        # Test parameter setting
        scaler.set_params(feature_range_min=-1.0, feature_range_max=1.0)
        self.assertEqual(scaler.get_params()['feature_range_min'], "-1.000000")
        self.assertEqual(scaler.get_params()['feature_range_max'], "1.000000")

class TestRobustScaler(unittest.TestCase):
    """Test RobustScaler algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.RobustScaler()
        X_scaled = scaler.fit_transform(self.X, self.y_dummy)
        
        self.assertEqual(X_scaled.shape, self.X.shape)
        self.assertIsInstance(X_scaled, np.ndarray)
        
    def test_fit_transform(self):
        """Test fit_transform method"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.RobustScaler()
        X_scaled = scaler.fit_transform(self.X, self.y_dummy)
        
        # Check that data is approximately centered and scaled
        median_scaled = np.median(X_scaled, axis=0)
        np.testing.assert_array_almost_equal(median_scaled, np.zeros(4), decimal=1)
        
    def test_inverse_transform(self):
        """Test inverse transform"""
        import auroraml.preprocessing as aml_pp
        
        scaler = aml_pp.RobustScaler()
        X_scaled = scaler.fit_transform(self.X, self.y_dummy)
        X_inverse = scaler.inverse_transform(X_scaled)
        
        np.testing.assert_array_almost_equal(X_inverse, self.X, decimal=10)

class TestLabelEncoder(unittest.TestCase):
    """Test LabelEncoder algorithm"""
    
    def setUp(self):
        """Set up test data"""
        self.y_labels = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.float64)
        self.y_dummy = np.zeros((len(self.y_labels), 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import auroraml.preprocessing as aml_pp
        
        encoder = aml_pp.LabelEncoder()
        y_encoded = encoder.fit_transform(self.y_dummy, self.y_labels)
        
        self.assertEqual(len(y_encoded), len(self.y_labels))
        self.assertIsInstance(y_encoded, np.ndarray)
        
    def test_inverse_transform(self):
        """Test inverse transform"""
        import auroraml.preprocessing as aml_pp
        
        encoder = aml_pp.LabelEncoder()
        y_encoded = encoder.fit_transform(self.y_dummy, self.y_labels)
        y_decoded = encoder.inverse_transform(y_encoded)
        
        # Check that we get back the original labels (length should match)
        self.assertEqual(len(y_decoded), len(self.y_labels))
        
    def test_unique_labels(self):
        """Test with unique labels"""
        import auroraml.preprocessing as aml_pp
        
        unique_labels = np.array([0, 1, 2], dtype=np.float64)
        y_dummy = np.zeros((len(unique_labels), 1)).astype(np.float64)
        
        encoder = aml_pp.LabelEncoder()
        y_encoded = encoder.fit_transform(y_dummy, unique_labels)
        
        # Should have 3 unique encoded values
        unique_encoded = np.unique(y_encoded)
        self.assertEqual(len(unique_encoded), 3)

class TestOneHotEncoder(unittest.TestCase):
    """Test OneHotEncoder algorithm"""
    
    def setUp(self):
        """Set up test data"""
        self.X_categorical = np.array([[0, 0], [1, 1], [2, 2], [0, 0]], dtype=np.float64)
        self.y_dummy = np.zeros((self.X_categorical.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import auroraml.preprocessing as aml_pp
        
        encoder = aml_pp.OneHotEncoder()
        X_encoded = encoder.fit_transform(self.X_categorical, self.y_dummy)
        
        self.assertIsInstance(X_encoded, np.ndarray)
        self.assertEqual(X_encoded.shape[0], self.X_categorical.shape[0])
        
    def test_inverse_transform(self):
        """Test inverse transform"""
        import auroraml.preprocessing as aml_pp
        
        encoder = aml_pp.OneHotEncoder()
        X_encoded = encoder.fit_transform(self.X_categorical, self.y_dummy)
        X_decoded = encoder.inverse_transform(X_encoded)
        
        # Should get back original shape
        self.assertEqual(X_decoded.shape, self.X_categorical.shape)

class TestOrdinalEncoder(unittest.TestCase):
    """Test OrdinalEncoder algorithm"""
    
    def setUp(self):
        """Set up test data"""
        self.X_categorical = np.array([[0, 0], [1, 1], [2, 2], [0, 0]], dtype=np.float64)
        self.y_dummy = np.zeros((self.X_categorical.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import auroraml.preprocessing as aml_pp
        
        encoder = aml_pp.OrdinalEncoder()
        X_encoded = encoder.fit_transform(self.X_categorical, self.y_dummy)
        
        self.assertEqual(X_encoded.shape, self.X_categorical.shape)
        self.assertIsInstance(X_encoded, np.ndarray)
        
    def test_inverse_transform(self):
        """Test inverse transform"""
        import auroraml.preprocessing as aml_pp
        
        encoder = aml_pp.OrdinalEncoder()
        X_encoded = encoder.fit_transform(self.X_categorical, self.y_dummy)
        X_decoded = encoder.inverse_transform(X_encoded)
        
        # Should get back original shape
        self.assertEqual(X_decoded.shape, self.X_categorical.shape)

class TestPreprocessingIntegration(unittest.TestCase):
    """Integration tests for preprocessing algorithms"""
    
    def test_pipeline_compatibility(self):
        """Test that different scalers can be used in sequence"""
        import auroraml.preprocessing as aml_pp
        
        np.random.seed(42)
        X = np.random.randn(100, 4).astype(np.float64)
        y_dummy = np.zeros((X.shape[0], 1)).astype(np.float64)
        
        # Test StandardScaler -> MinMaxScaler pipeline
        scaler1 = aml_pp.StandardScaler()
        scaler2 = aml_pp.MinMaxScaler()
        
        X_scaled1 = scaler1.fit_transform(X, y_dummy)
        X_scaled2 = scaler2.fit_transform(X_scaled1, y_dummy)
        
        self.assertEqual(X_scaled2.shape, X.shape)
        self.assertTrue(np.all(X_scaled2 >= 0))
        self.assertTrue(np.all(X_scaled2 <= 1))
        
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.preprocessing as aml_pp
        
        # Test with single sample
        X_single = np.random.randn(1, 4).astype(np.float64)
        y_dummy_single = np.zeros((1, 1)).astype(np.float64)
        
        scaler = aml_pp.StandardScaler()
        X_scaled = scaler.fit_transform(X_single, y_dummy_single)
        
        self.assertEqual(X_scaled.shape, X_single.shape)
        
        # Test with empty data
        with self.assertRaises(ValueError):
            scaler.fit(np.array([]).reshape(0, 4), np.array([]).reshape(0, 1))
            
    def test_model_persistence(self):
        """Test model saving and loading"""
        import auroraml.preprocessing as aml_pp
        
        np.random.seed(42)
        X = np.random.randn(100, 4).astype(np.float64)
        y_dummy = np.zeros((X.shape[0], 1)).astype(np.float64)
        
        scaler = aml_pp.StandardScaler()
        scaler.fit(X, y_dummy)
        
        # Save model
        filename = "test_scaler.bin"
        scaler.save(filename)
        
        # Load model
        loaded_scaler = aml_pp.StandardScaler()
        loaded_scaler.load(filename)
        
        # Compare transformations
        original_transformed = scaler.transform(X, y_dummy)
        loaded_transformed = loaded_scaler.transform(X, y_dummy)
        
        np.testing.assert_array_almost_equal(original_transformed, loaded_transformed, decimal=10)
        
        # Clean up
        os.remove(filename)

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
