#!/usr/bin/env python3
import random
"""
Test Suite for AuroraML Decomposition
Tests PCA, TruncatedSVD, and LDA algorithms
"""

import sys
import os
import unittest
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

class TestPCA(unittest.TestCase):
    """Test PCA algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 10).astype(np.float64)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.PCA(n_components=3)
        model.fit(self.X, self.y_dummy)
        X_transformed = model.transform(self.X)
        
        self.assertEqual(X_transformed.shape, (100, 3))
        self.assertIsInstance(X_transformed, np.ndarray)
        
    def test_fit_transform(self):
        """Test fit_transform method"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.PCA(n_components=3)
        X_transformed = model.fit_transform(self.X, self.y_dummy)
        
        self.assertEqual(X_transformed.shape, (100, 3))
        self.assertIsInstance(X_transformed, np.ndarray)
        
    def test_inverse_transform(self):
        """Test inverse_transform method"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.PCA(n_components=3)
        model.fit(self.X, self.y_dummy)
        X_transformed = model.transform(self.X)
        X_inverse = model.inverse_transform(X_transformed)
        
        # Should be close to original (within some tolerance)
        np.testing.assert_array_almost_equal(X_inverse, self.X, decimal=1)
        
    def test_different_n_components(self):
        """Test with different n_components values"""
        import auroraml.decomposition as aml_decomp
        
        n_components_values = [1, 3, 5, 8]
        for n_components in n_components_values:
            model = aml_decomp.PCA(n_components=n_components)
            model.fit(self.X, self.y_dummy)
            X_transformed = model.transform(self.X)
            
            self.assertEqual(X_transformed.shape, (100, n_components))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.PCA(n_components=3, whiten=True)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_components', params)
        self.assertIn('whiten', params)
        
        # Test parameter setting
        model.set_params(n_components=5)
        self.assertEqual(model.get_params()['n_components'], "5")
        
    def test_explained_variance(self):
        """Test explained variance ratio"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.PCA(n_components=3)
        model.fit(self.X, self.y_dummy)
        
        # Check that explained variance ratio is available
        if hasattr(model, 'explained_variance_ratio_'):
            explained_var = model.explained_variance_ratio_
            self.assertEqual(len(explained_var), 3)
            self.assertTrue(np.all(explained_var >= 0))
            self.assertTrue(np.all(explained_var <= 1))
            self.assertAlmostEqual(np.sum(explained_var), 1.0, places=5)
            
    def test_performance(self):
        """Test model performance"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.PCA(n_components=3)
        model.fit(self.X, self.y_dummy)
        X_transformed = model.transform(self.X)
        
        # Transformed data should have correct shape
        self.assertEqual(X_transformed.shape, (100, 3))
        
        # Components should be orthogonal
        if hasattr(model, 'components_'):
            components = model.components_
            dot_product = np.dot(components, components.T)
            np.testing.assert_array_almost_equal(dot_product, np.eye(3), decimal=10)

class TestTruncatedSVD(unittest.TestCase):
    """Test TruncatedSVD algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 10).astype(np.float64)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.TruncatedSVD(n_components=3)
        model.fit(self.X, self.y_dummy)
        X_transformed = model.transform(self.X)
        
        self.assertEqual(X_transformed.shape, (100, 3))
        self.assertIsInstance(X_transformed, np.ndarray)
        
    def test_fit_transform(self):
        """Test fit_transform method"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.TruncatedSVD(n_components=3)
        X_transformed = model.fit_transform(self.X, self.y_dummy)
        
        self.assertEqual(X_transformed.shape, (100, 3))
        self.assertIsInstance(X_transformed, np.ndarray)
        
    def test_inverse_transform(self):
        """Test inverse_transform method"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.TruncatedSVD(n_components=3)
        model.fit(self.X, self.y_dummy)
        X_transformed = model.transform(self.X)
        X_inverse = model.inverse_transform(X_transformed)
        
        # Should be close to original (within some tolerance)
        np.testing.assert_array_almost_equal(X_inverse, self.X, decimal=1)
        
    def test_different_n_components(self):
        """Test with different n_components values"""
        import auroraml.decomposition as aml_decomp
        
        n_components_values = [1, 3, 5, 8]
        for n_components in n_components_values:
            model = aml_decomp.TruncatedSVD(n_components=n_components)
            model.fit(self.X, self.y_dummy)
            X_transformed = model.transform(self.X)
            
            self.assertEqual(X_transformed.shape, (100, n_components))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.TruncatedSVD(n_components=3, n_iter=5)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_components', params)
        self.assertIn('n_iter', params)
        
        # Test parameter setting
        model.set_params(n_components=5)
        self.assertEqual(model.get_params()['n_components'], "5")

class TestLDA(unittest.TestCase):
    """Test LDA algorithm"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 10).astype(np.float64)
        self.y = np.random.randint(0, 3, 100).astype(np.int32)  # 3 classes
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_basic_functionality(self):
        """Test basic fit and transform functionality"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.LDA(n_components=2)
        model.fit(self.X, self.y_dummy, self.y)
        X_transformed = model.transform(self.X)
        
        self.assertEqual(X_transformed.shape, (100, 2))
        self.assertIsInstance(X_transformed, np.ndarray)
        
    def test_fit_transform(self):
        """Test fit_transform method"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.LDA(n_components=2)
        X_transformed = model.fit_transform(self.X, self.y_dummy, self.y)
        
        self.assertEqual(X_transformed.shape, (100, 2))
        self.assertIsInstance(X_transformed, np.ndarray)
        
    def test_different_n_components(self):
        """Test with different n_components values"""
        import auroraml.decomposition as aml_decomp
        
        # LDA can have at most n_classes - 1 components
        n_components_values = [1, 2]
        for n_components in n_components_values:
            model = aml_decomp.LDA(n_components=n_components)
            model.fit(self.X, self.y_dummy, self.y)
            X_transformed = model.transform(self.X)
            
            self.assertEqual(X_transformed.shape, (100, n_components))
            
    def test_parameters(self):
        """Test parameter getter and setter"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.LDA(n_components=2)
        
        # Test default parameters
        params = model.get_params()
        self.assertIn('n_components', params)
        
        # Test parameter setting
        model.set_params(n_components=1)
        self.assertEqual(model.get_params()['n_components'], "1")
        
    def test_class_separation(self):
        """Test that LDA improves class separation"""
        import auroraml.decomposition as aml_decomp
        
        model = aml_decomp.LDA(n_components=2)
        model.fit(self.X, self.y_dummy, self.y)
        X_transformed = model.transform(self.X)
        
        # Check that transformed data has correct shape
        self.assertEqual(X_transformed.shape, (100, 2))
        
        # Check that different classes are separated in transformed space
        unique_classes = np.unique(self.y)
        for i, class1 in enumerate(unique_classes):
            for class2 in unique_classes[i+1:]:
                class1_data = X_transformed[self.y == class1]
                class2_data = X_transformed[self.y == class2]
                
                # Classes should be separated (different means)
                mean1 = np.mean(class1_data, axis=0)
                mean2 = np.mean(class2_data, axis=0)
                self.assertGreater(np.linalg.norm(mean1 - mean2), 0)

class TestDecompositionIntegration(unittest.TestCase):
    """Integration tests for decomposition algorithms"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 10).astype(np.float64)
        self.y = np.random.randint(0, 3, 100).astype(np.int32)
        self.y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        
    def test_model_comparison(self):
        """Compare different decomposition algorithms"""
        import auroraml.decomposition as aml_decomp
        
        models = [
            ('PCA', aml_decomp.PCA(n_components=3)),
            ('TruncatedSVD', aml_decomp.TruncatedSVD(n_components=3)),
            ('LDA', aml_decomp.LDA(n_components=2))
        ]
        
        for name, model in models:
            if name == 'LDA':
                model.fit(self.X, self.y_dummy, self.y)
            else:
                model.fit(self.X, self.y_dummy)
                
            X_transformed = model.transform(self.X)
            self.assertEqual(X_transformed.shape[0], len(self.X))
            
    def test_dimensionality_reduction(self):
        """Test that dimensionality reduction works correctly"""
        import auroraml.decomposition as aml_decomp
        
        # Test PCA
        pca = aml_decomp.PCA(n_components=5)
        pca.fit(self.X, self.y_dummy)
        X_pca = pca.transform(self.X)
        
        self.assertEqual(X_pca.shape, (100, 5))
        self.assertLess(X_pca.shape[1], self.X.shape[1])
        
        # Test TruncatedSVD
        svd = aml_decomp.TruncatedSVD(n_components=5)
        svd.fit(self.X, self.y_dummy)
        X_svd = svd.transform(self.X)
        
        self.assertEqual(X_svd.shape, (100, 5))
        self.assertLess(X_svd.shape[1], self.X.shape[1])
        
    def test_reconstruction_quality(self):
        """Test reconstruction quality"""
        import auroraml.decomposition as aml_decomp
        
        # Test PCA reconstruction
        pca = aml_decomp.PCA(n_components=5)
        pca.fit(self.X, self.y_dummy)
        X_transformed = pca.transform(self.X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Reconstruction should be close to original
        reconstruction_error = np.mean((self.X - X_reconstructed) ** 2)
        self.assertLess(reconstruction_error, 1.0)
        
    def test_edge_cases(self):
        """Test edge cases"""
        import auroraml.decomposition as aml_decomp
        
        # Test with single sample
        X_single = self.X[:1]
        y_dummy_single = np.zeros((1, 1)).astype(np.float64)
        
        pca = aml_decomp.PCA(n_components=1)
        pca.fit(X_single, y_dummy_single)
        X_transformed = pca.transform(X_single)
        self.assertEqual(X_transformed.shape, (1, 1))
        
        # Test with empty data
        with self.assertRaises(ValueError):
            pca.fit(np.array([]).reshape(0, 10), np.array([]).reshape(0, 1))
            
    def test_single_feature(self):
        """Test with single feature"""
        import auroraml.decomposition as aml_decomp
        
        X_single_feature = self.X[:, [0]]
        y_dummy = np.zeros((X_single_feature.shape[0], 1)).astype(np.float64)
        
        pca = aml_decomp.PCA(n_components=1)
        pca.fit(X_single_feature, y_dummy)
        X_transformed = pca.transform(X_single_feature)
        self.assertEqual(X_transformed.shape, (100, 1))
        
    def test_not_fitted_error(self):
        """Test error when using unfitted model"""
        import auroraml.decomposition as aml_decomp
        
        pca = aml_decomp.PCA(n_components=3)
        
        # Should raise error when not fitted
        with self.assertRaises((AttributeError, RuntimeError)):
            pca.transform(self.X)

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
