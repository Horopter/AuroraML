"""Tests for extended preprocessing utilities"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ingenuityml
import random

class TestPreprocessingExtended(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4).astype(np.float64)

    def test_max_abs_scaler(self):
        """Test MaxAbsScaler"""
        scaler = ingenuityml.preprocessing.MaxAbsScaler()
        scaler.fit(self.X, None)
        
        X_transformed = scaler.transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)
        
        # Values should be in [-1, 1]
        self.assertTrue(np.all(np.abs(X_transformed) <= 1.0 + 1e-10))
        
        max_abs = scaler.max_abs()
        self.assertEqual(len(max_abs), self.X.shape[1])
        
        # Test inverse transform
        X_inverse = scaler.inverse_transform(X_transformed)
        np.testing.assert_array_almost_equal(X_inverse, self.X, decimal=5)

    def test_binarizer(self):
        """Test Binarizer"""
        binarizer = ingenuityml.preprocessing.Binarizer(threshold=0.0)
        binarizer.fit(self.X, None)
        
        X_binary = binarizer.transform(self.X)
        self.assertEqual(X_binary.shape, self.X.shape)
        
        # Should be binary (0 or 1)
        self.assertTrue(np.all((X_binary == 0) | (X_binary == 1)))
        
        # Test fit_transform
        X_binary2 = binarizer.fit_transform(self.X, None)
        self.assertTrue(np.all((X_binary2 == 0) | (X_binary2 == 1)))

    def test_label_binarizer(self):
        """Test LabelBinarizer"""
        y = np.array([0, 1, 2, 0, 1, 2], dtype=np.float64)
        X_dummy = y.reshape(-1, 1)
        lb = ingenuityml.preprocessing.LabelBinarizer()
        lb.fit(X_dummy, y)
        y_bin = lb.transform(X_dummy)
        self.assertEqual(y_bin.shape, (y.shape[0], 3))
        y_inv = lb.inverse_transform(y_bin)
        self.assertEqual(y_inv.shape, (y.shape[0], 1))

    def test_multilabel_binarizer(self):
        """Test MultiLabelBinarizer"""
        X_labels = np.array([
            [0, 1, np.nan],
            [1, 2, np.nan],
            [0, 2, 1],
        ], dtype=np.float64)
        y_dummy = np.zeros((X_labels.shape[0], 1)).astype(np.float64)
        mlb = ingenuityml.preprocessing.MultiLabelBinarizer()
        mlb.fit(X_labels, y_dummy)
        X_bin = mlb.transform(X_labels)
        self.assertEqual(X_bin.shape, (X_labels.shape[0], 3))
        X_inv = mlb.inverse_transform(X_bin)
        self.assertEqual(X_inv.shape[0], X_labels.shape[0])

    def test_kbins_discretizer(self):
        """Test KBinsDiscretizer"""
        y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        kb_ordinal = ingenuityml.preprocessing.KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
        kb_ordinal.fit(self.X, y_dummy)
        X_ord = kb_ordinal.transform(self.X)
        self.assertEqual(X_ord.shape, self.X.shape)

        kb_onehot = ingenuityml.preprocessing.KBinsDiscretizer(n_bins=3, encode="onehot", strategy="quantile")
        kb_onehot.fit(self.X, y_dummy)
        X_oh = kb_onehot.transform(self.X)
        self.assertEqual(X_oh.shape[0], self.X.shape[0])
        self.assertEqual(X_oh.shape[1], self.X.shape[1] * 3)

    def test_quantile_transformer(self):
        """Test QuantileTransformer"""
        y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        qt = ingenuityml.preprocessing.QuantileTransformer(n_quantiles=10, output_distribution="uniform")
        qt.fit(self.X, y_dummy)
        X_q = qt.transform(self.X)
        self.assertTrue(np.all(X_q >= 0.0))
        self.assertTrue(np.all(X_q <= 1.0))

        qt_norm = ingenuityml.preprocessing.QuantileTransformer(n_quantiles=10, output_distribution="normal")
        X_qn = qt_norm.fit_transform(self.X, y_dummy)
        self.assertFalse(np.any(np.isnan(X_qn)))

    def test_power_transformer(self):
        """Test PowerTransformer"""
        y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        pt = ingenuityml.preprocessing.PowerTransformer(method="yeo-johnson", standardize=True)
        pt.fit(self.X, y_dummy)
        X_t = pt.transform(self.X)
        self.assertEqual(X_t.shape, self.X.shape)
        X_inv = pt.inverse_transform(X_t)
        self.assertEqual(X_inv.shape, self.X.shape)

    def test_function_transformer(self):
        """Test FunctionTransformer"""
        X_nonneg = np.abs(self.X)
        ft = ingenuityml.preprocessing.FunctionTransformer(func="log1p", inverse_func="expm1")
        ft.fit(X_nonneg, None)
        X_t = ft.transform(X_nonneg)
        self.assertEqual(X_t.shape, X_nonneg.shape)
        X_inv = ft.inverse_transform(X_t)
        self.assertEqual(X_inv.shape, X_nonneg.shape)

    def test_spline_transformer(self):
        """Test SplineTransformer"""
        y_dummy = np.zeros((self.X.shape[0], 1)).astype(np.float64)
        st = ingenuityml.preprocessing.SplineTransformer(n_knots=4, degree=3, include_bias=True)
        st.fit(self.X, y_dummy)
        X_s = st.transform(self.X)
        self.assertEqual(X_s.shape[0], self.X.shape[0])
        self.assertGreater(X_s.shape[1], self.X.shape[1])
        X_inv = st.inverse_transform(X_s)
        self.assertEqual(X_inv.shape, self.X.shape)

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
