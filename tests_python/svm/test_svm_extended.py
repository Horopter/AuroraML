#!/usr/bin/env python3
"""
Test extended SVM algorithms (LinearSVR, NuSVC, NuSVR)
"""

import os
import sys
import unittest

import numpy as np

# Add the build directory to Python path
build_path = os.path.join(os.path.dirname(__file__), "..", "build")
sys.path.insert(0, build_path)

try:
    import ingenuityml
except ImportError as e:
    print(f"Failed to import ingenuityml: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


class TestLinearSVR(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.y = self.X[:, 0] + 0.5 * self.X[:, 1] + 0.1 * np.random.randn(100)

        # Small dataset for quick tests
        self.X_small = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
        self.y_small = np.array([3.0, 5.0, 7.0, 9.0])

    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        model = ingenuityml.svm.LinearSVR(C=1.0, epsilon=0.1, max_iter=100)
        model.fit(self.X_small, self.y_small)

        self.assertTrue(model.is_fitted())

        predictions = model.predict(self.X_small)
        self.assertEqual(len(predictions), len(self.y_small))

        # Check that predictions are reasonable
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_parameters(self):
        """Test parameter getter and setter"""
        model = ingenuityml.svm.LinearSVR(C=2.0, epsilon=0.2)

        # Test parameter access
        params = model.get_params()
        self.assertIsInstance(params, dict)

        # Test coefficient and intercept access
        model.fit(self.X_small, self.y_small)
        coef = model.coef()
        intercept = model.intercept()

        self.assertEqual(len(coef), self.X_small.shape[1])
        self.assertIsInstance(intercept, float)

    def test_different_parameters(self):
        """Test with different parameter values"""
        parameters = [
            {"C": 0.1, "epsilon": 0.01},
            {"C": 1.0, "epsilon": 0.1},
            {"C": 10.0, "epsilon": 1.0},
        ]

        for params in parameters:
            with self.subTest(params=params):
                model = ingenuityml.svm.LinearSVR(**params, max_iter=50)
                model.fit(self.X_small, self.y_small)

                self.assertTrue(model.is_fitted())
                predictions = model.predict(self.X_small)
                self.assertEqual(len(predictions), len(self.y_small))

    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        model = ingenuityml.svm.LinearSVR()

        with self.assertRaises(RuntimeError):
            model.predict(self.X_small)

    def test_performance(self):
        """Test model performance on a simple regression task"""
        # Linear relationship
        model = ingenuityml.svm.LinearSVR(C=1.0, epsilon=0.1, max_iter=200)
        model.fit(self.X, self.y)

        predictions = model.predict(self.X)
        mse = np.mean((predictions - self.y) ** 2)

        # Should achieve reasonable accuracy
        self.assertLess(mse, 1.0)


class TestNuSVC(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)

        # Small dataset for quick tests
        self.X_small = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
        self.y_small = np.array([0, 0, 1, 1])

    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        model = ingenuityml.svm.NuSVC(nu=0.5, max_iter=100)
        model.fit(self.X_small, self.y_small)

        self.assertTrue(model.is_fitted())

        predictions = model.predict(self.X_small)
        self.assertEqual(len(predictions), len(self.y_small))

        # Check that predictions are valid class labels
        unique_predictions = set(predictions)
        unique_true = set(self.y_small)
        self.assertTrue(unique_predictions.issubset(unique_true))

    def test_predict_proba(self):
        """Test predict_proba method"""
        model = ingenuityml.svm.NuSVC(nu=0.5, max_iter=100)
        model.fit(self.X_small, self.y_small)

        proba = model.predict_proba(self.X_small)

        # Check shape
        self.assertEqual(proba.shape[0], len(self.y_small))
        self.assertEqual(proba.shape[1], 2)  # Binary classification

        # Check that probabilities sum to 1
        for i in range(proba.shape[0]):
            prob_sum = np.sum(proba[i, :])
            self.assertAlmostEqual(prob_sum, 1.0, places=5)

        # Check that all probabilities are non-negative
        self.assertTrue(np.all(proba >= 0))

    def test_decision_function(self):
        """Test decision_function method"""
        model = ingenuityml.svm.NuSVC(nu=0.5, max_iter=100)
        model.fit(self.X_small, self.y_small)

        decision_scores = model.decision_function(self.X_small)

        # Check shape
        self.assertEqual(len(decision_scores), len(self.y_small))

    def test_classes_property(self):
        """Test classes property"""
        model = ingenuityml.svm.NuSVC()
        model.fit(self.X_small, self.y_small)

        classes = model.classes()
        expected_classes = sorted(set(self.y_small))
        self.assertEqual(sorted(classes), expected_classes)

    def test_different_nu_values(self):
        """Test with different nu parameter values"""
        nu_values = [0.1, 0.5, 0.9]

        for nu in nu_values:
            with self.subTest(nu=nu):
                model = ingenuityml.svm.NuSVC(nu=nu, max_iter=50)
                model.fit(self.X_small, self.y_small)

                self.assertTrue(model.is_fitted())
                predictions = model.predict(self.X_small)
                self.assertEqual(len(predictions), len(self.y_small))

    def test_invalid_nu(self):
        """Test with invalid nu values - should raise error"""
        invalid_nu_values = [0.0, -0.1, 1.1, 2.0]

        for nu in invalid_nu_values:
            with self.subTest(nu=nu):
                model = ingenuityml.svm.NuSVC(nu=nu)
                with self.assertRaises(ValueError):
                    model.fit(self.X_small, self.y_small)

    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        model = ingenuityml.svm.NuSVC()

        with self.assertRaises(RuntimeError):
            model.predict(self.X_small)

        with self.assertRaises(RuntimeError):
            model.predict_proba(self.X_small)

        with self.assertRaises(RuntimeError):
            model.decision_function(self.X_small)

class TestSVC(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(60, 2)
        self.y = (self.X[:, 0] * self.X[:, 0] + self.X[:, 1] > 0.5).astype(int)

    def test_rbf_kernel(self):
        model = ingenuityml.svm.SVC(kernel="rbf", C=1.0, gamma=0.5)
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())

        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape[0], len(self.y))
        self.assertEqual(proba.shape[1], len(model.classes()))

    def test_poly_kernel(self):
        model = ingenuityml.svm.SVC(kernel="poly", C=1.0, gamma=0.5, degree=2.0, coef0=1.0)
        model.fit(self.X, self.y)
        scores = model.decision_function(self.X)
        self.assertEqual(len(scores), len(self.y))


class TestNuSVR(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.y = self.X[:, 0] + 0.5 * self.X[:, 1] + 0.1 * np.random.randn(100)

        # Small dataset for quick tests
        self.X_small = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
        self.y_small = np.array([3.0, 5.0, 7.0, 9.0])

    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        model = ingenuityml.svm.NuSVR(nu=0.5, C=1.0, max_iter=100)
        model.fit(self.X_small, self.y_small)

        self.assertTrue(model.is_fitted())

        predictions = model.predict(self.X_small)
        self.assertEqual(len(predictions), len(self.y_small))

        # Check that predictions are reasonable
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_parameters(self):
        """Test parameter getter and setter"""
        model = ingenuityml.svm.NuSVR(nu=0.3, C=2.0)

        # Test parameter access
        params = model.get_params()
        self.assertIsInstance(params, dict)

        # Test coefficient and intercept access
        model.fit(self.X_small, self.y_small)
        coef = model.coef()
        intercept = model.intercept()

        self.assertEqual(len(coef), self.X_small.shape[1])
        self.assertIsInstance(intercept, float)

    def test_different_parameters(self):
        """Test with different parameter values"""
        parameters = [
            {"nu": 0.1, "C": 0.5},
            {"nu": 0.5, "C": 1.0},
            {"nu": 0.9, "C": 2.0},
        ]

        for params in parameters:
            with self.subTest(params=params):
                model = ingenuityml.svm.NuSVR(**params, max_iter=50)
                model.fit(self.X_small, self.y_small)

                self.assertTrue(model.is_fitted())
                predictions = model.predict(self.X_small)
                self.assertEqual(len(predictions), len(self.y_small))

    def test_invalid_parameters(self):
        """Test with invalid parameter values - should raise error"""
        # Invalid nu values
        invalid_nu_values = [0.0, -0.1, 1.1]
        for nu in invalid_nu_values:
            with self.subTest(nu=nu):
                model = ingenuityml.svm.NuSVR(nu=nu, C=1.0)
                with self.assertRaises(ValueError):
                    model.fit(self.X_small, self.y_small)

        # Invalid C values
        invalid_C_values = [0.0, -1.0]
        for C in invalid_C_values:
            with self.subTest(C=C):
                model = ingenuityml.svm.NuSVR(nu=0.5, C=C)
                with self.assertRaises(ValueError):
                    model.fit(self.X_small, self.y_small)

    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        model = ingenuityml.svm.NuSVR()

        with self.assertRaises(RuntimeError):
            model.predict(self.X_small)

    def test_performance(self):
        """Test model performance on a simple regression task"""
        # Linear relationship
        model = ingenuityml.svm.NuSVR(nu=0.5, C=1.0, max_iter=200)
        model.fit(self.X, self.y)

        predictions = model.predict(self.X)
        mse = np.mean((predictions - self.y) ** 2)

        # Should achieve reasonable accuracy
        self.assertLess(mse, 1.0)


class TestSVMIntegration(unittest.TestCase):
    def test_all_svm_variants_available(self):
        """Test that all SVM variants are available in the module"""
        expected_algorithms = ["LinearSVC", "LinearSVR", "NuSVC", "NuSVR", "SVC", "SVR"]

        available_algorithms = [x for x in dir(ingenuityml.svm) if not x.startswith("_")]

        for alg in expected_algorithms:
            self.assertIn(alg, available_algorithms, f"{alg} not found in ingenuityml.svm")

    def test_svm_vs_linear_performance(self):
        """Compare SVM performance with linear models"""
        np.random.seed(42)
        X = np.random.randn(200, 4)
        y = (X[:, 0] * X[:, 1] > 0).astype(int)  # Non-linear boundary

        # Train SVM
        svm_model = ingenuityml.svm.NuSVC(nu=0.5, max_iter=500)
        svm_model.fit(X, y)
        svm_pred = svm_model.predict(X)
        svm_acc = np.mean(svm_pred == y)

        # Train logistic regression
        lr_model = ingenuityml.linear_model.LogisticRegression(max_iter=500)
        lr_model.fit(X, y)
        lr_pred = lr_model.predict(X)
        lr_acc = np.mean(lr_pred == y)

        # SVM should perform at least as well as logistic regression on non-linear data
        self.assertGreaterEqual(svm_acc, lr_acc - 0.05)  # Allow small tolerance


if __name__ == "__main__":
    unittest.main()
