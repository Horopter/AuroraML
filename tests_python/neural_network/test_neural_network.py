#!/usr/bin/env python3
"""
Test neural network algorithms
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


class TestMLPClassifier(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Binary classification data
        self.X_binary = np.random.randn(100, 4)
        self.y_binary = (self.X_binary[:, 0] + self.X_binary[:, 1] > 0).astype(int)

        # Multiclass classification data
        self.X_multi = np.random.randn(150, 4)
        self.y_multi = np.random.randint(0, 3, 150)

        # Small dataset for quick tests
        self.X_small = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_small = np.array([0, 0, 1, 1])

    def test_basic_functionality_binary(self):
        """Test basic fit and predict functionality for binary classification"""
        model = ingenuityml.neural_network.MLPClassifier(
            hidden_layer_sizes=[10], max_iter=50, random_state=42, learning_rate=0.01
        )
        model.fit(self.X_small, self.y_small)

        self.assertTrue(model.is_fitted())

        predictions = model.predict(self.X_small)
        self.assertEqual(len(predictions), len(self.y_small))

        # Check that predictions are valid class labels
        unique_predictions = set(predictions)
        unique_true = set(self.y_small)
        self.assertTrue(unique_predictions.issubset(unique_true))

    def test_basic_functionality_multiclass(self):
        """Test basic fit and predict functionality for multiclass classification"""
        model = ingenuityml.neural_network.MLPClassifier(
            hidden_layer_sizes=[10, 5],
            max_iter=100,
            random_state=42,
            learning_rate=0.01,
        )
        model.fit(self.X_multi, self.y_multi)

        self.assertTrue(model.is_fitted())

        predictions = model.predict(self.X_multi)
        self.assertEqual(len(predictions), len(self.y_multi))

        # Check that predictions are valid class labels
        unique_predictions = set(predictions)
        unique_true = set(self.y_multi)
        self.assertTrue(unique_predictions.issubset(unique_true))

    def test_predict_proba(self):
        """Test predict_proba method"""
        model = ingenuityml.neural_network.MLPClassifier(
            hidden_layer_sizes=[5], max_iter=50, random_state=42
        )
        model.fit(self.X_small, self.y_small)

        proba = model.predict_proba(self.X_small)

        # Check shape
        self.assertEqual(proba.shape[0], len(self.y_small))
        self.assertEqual(proba.shape[1], model.n_classes())

        # Check that probabilities sum to 1
        for i in range(proba.shape[0]):
            prob_sum = np.sum(proba[i, :])
            self.assertAlmostEqual(prob_sum, 1.0, places=5)

        # Check that all probabilities are non-negative
        self.assertTrue(np.all(proba >= 0))

    def test_decision_function(self):
        """Test decision_function method"""
        model = ingenuityml.neural_network.MLPClassifier(
            hidden_layer_sizes=[5], max_iter=50, random_state=42
        )
        model.fit(self.X_small, self.y_small)

        decision_scores = model.decision_function(self.X_small)

        # Check shape
        self.assertEqual(len(decision_scores), len(self.y_small))

    def test_different_activation_functions(self):
        """Test different activation functions"""
        activations = [
            ingenuityml.neural_network.ActivationFunction.RELU,
            ingenuityml.neural_network.ActivationFunction.TANH,
            ingenuityml.neural_network.ActivationFunction.LOGISTIC,
        ]

        for activation in activations:
            with self.subTest(activation=activation):
                model = ingenuityml.neural_network.MLPClassifier(
                    hidden_layer_sizes=[5],
                    activation=activation,
                    max_iter=30,
                    random_state=42,
                )
                model.fit(self.X_small, self.y_small)

                self.assertTrue(model.is_fitted())
                predictions = model.predict(self.X_small)
                self.assertEqual(len(predictions), len(self.y_small))

    def test_different_solvers(self):
        """Test different solvers"""
        solvers = [
            ingenuityml.neural_network.Solver.ADAM,
            ingenuityml.neural_network.Solver.SGD,
        ]

        for solver in solvers:
            with self.subTest(solver=solver):
                model = ingenuityml.neural_network.MLPClassifier(
                    hidden_layer_sizes=[5], solver=solver, max_iter=30, random_state=42
                )
                model.fit(self.X_small, self.y_small)

                self.assertTrue(model.is_fitted())
                predictions = model.predict(self.X_small)
                self.assertEqual(len(predictions), len(self.y_small))

    def test_parameters(self):
        """Test parameter getter and setter"""
        model = ingenuityml.neural_network.MLPClassifier(alpha=0.01, learning_rate=0.001)

        self.assertAlmostEqual(model.alpha(), 0.01, places=6)
        self.assertAlmostEqual(model.learning_rate(), 0.001, places=6)

        # Test parameter setting
        params = model.get_params()
        self.assertIsInstance(params, dict)

    def test_loss_curve(self):
        """Test that loss curve is recorded during training"""
        model = ingenuityml.neural_network.MLPClassifier(
            hidden_layer_sizes=[5], max_iter=20, random_state=42
        )
        model.fit(self.X_small, self.y_small)

        loss_curve = model.loss_curve()
        self.assertGreater(len(loss_curve), 0)
        self.assertLessEqual(len(loss_curve), 20)  # Should not exceed max_iter

        # Loss should generally decrease (allowing for some noise)
        self.assertLess(loss_curve[-1], loss_curve[0] * 2)

    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        model = ingenuityml.neural_network.MLPClassifier()

        with self.assertRaises(RuntimeError):
            model.predict(self.X_small)

        with self.assertRaises(RuntimeError):
            model.predict_proba(self.X_small)

        with self.assertRaises(RuntimeError):
            model.decision_function(self.X_small)

    def test_empty_data(self):
        """Test with empty data - should raise error"""
        model = ingenuityml.neural_network.MLPClassifier()

        with self.assertRaises(Exception):  # Could be various exception types
            model.fit(np.array([]).reshape(0, 2), np.array([]))

    def test_dimension_mismatch(self):
        """Test with dimension mismatch - should raise error"""
        model = ingenuityml.neural_network.MLPClassifier()
        model.fit(self.X_small, self.y_small)

        with self.assertRaises(Exception):
            model.predict(np.array([[1, 2, 3]]))  # Wrong number of features

    def test_n_iter_property(self):
        """Test n_iter property"""
        model = ingenuityml.neural_network.MLPClassifier(
            hidden_layer_sizes=[5], max_iter=10, random_state=42
        )
        model.fit(self.X_small, self.y_small)

        n_iter = model.n_iter()
        self.assertGreater(n_iter, 0)
        self.assertLessEqual(n_iter, 10)

    def test_classes_property(self):
        """Test classes property"""
        model = ingenuityml.neural_network.MLPClassifier()
        model.fit(self.X_small, self.y_small)

        classes = model.classes()
        expected_classes = sorted(set(self.y_small))
        self.assertEqual(sorted(classes), expected_classes)

        n_classes = model.n_classes()
        self.assertEqual(n_classes, len(expected_classes))


class TestMLPRegressor(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Regression data
        self.X = np.random.randn(100, 4)
        self.y = self.X[:, 0] * 2 + self.X[:, 1] * 1.5 + np.random.randn(100) * 0.1

        # Small dataset for quick tests
        self.X_small = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_small = np.array([3.0, 5.0, 7.0, 9.0])

    def test_basic_functionality(self):
        """Test basic fit and predict functionality"""
        model = ingenuityml.neural_network.MLPRegressor(
            hidden_layer_sizes=[10], max_iter=50, random_state=42, learning_rate=0.01
        )
        model.fit(self.X_small, self.y_small)

        self.assertTrue(model.is_fitted())

        predictions = model.predict(self.X_small)
        self.assertEqual(len(predictions), len(self.y_small))

        # Check that predictions are reasonable (not NaN or infinite)
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_different_hidden_layer_sizes(self):
        """Test different hidden layer configurations"""
        configurations = [[5], [10, 5], [8, 4, 2]]

        for hidden_layers in configurations:
            with self.subTest(hidden_layers=hidden_layers):
                model = ingenuityml.neural_network.MLPRegressor(
                    hidden_layer_sizes=hidden_layers, max_iter=30, random_state=42
                )
                model.fit(self.X_small, self.y_small)

                self.assertTrue(model.is_fitted())
                predictions = model.predict(self.X_small)
                self.assertEqual(len(predictions), len(self.y_small))

    def test_regularization_effect(self):
        """Test that regularization affects the model"""
        # Train models with different regularization
        model_no_reg = ingenuityml.neural_network.MLPRegressor(
            alpha=0.0, hidden_layer_sizes=[10], max_iter=50, random_state=42
        )
        model_high_reg = ingenuityml.neural_network.MLPRegressor(
            alpha=1.0, hidden_layer_sizes=[10], max_iter=50, random_state=42
        )

        model_no_reg.fit(self.X_small, self.y_small)
        model_high_reg.fit(self.X_small, self.y_small)

        # Both should be fitted
        self.assertTrue(model_no_reg.is_fitted())
        self.assertTrue(model_high_reg.is_fitted())

        # Predictions should be different due to regularization
        pred_no_reg = model_no_reg.predict(self.X_small)
        pred_high_reg = model_high_reg.predict(self.X_small)

        # Should not be identical (allowing for very small differences)
        max_diff = np.max(np.abs(pred_no_reg - pred_high_reg))
        self.assertGreater(max_diff, 1e-6)

    def test_loss_curve(self):
        """Test that loss curve is recorded during training"""
        model = ingenuityml.neural_network.MLPRegressor(
            hidden_layer_sizes=[5], max_iter=20, random_state=42
        )
        model.fit(self.X_small, self.y_small)

        loss_curve = model.loss_curve()
        self.assertGreater(len(loss_curve), 0)
        self.assertLessEqual(len(loss_curve), 20)  # Should not exceed max_iter

    def test_not_fitted_predict(self):
        """Test predict without fitting - should raise error"""
        model = ingenuityml.neural_network.MLPRegressor()

        with self.assertRaises(RuntimeError):
            model.predict(self.X_small)

    def test_parameters(self):
        """Test parameter getter and setter"""
        model = ingenuityml.neural_network.MLPRegressor(
            alpha=0.01, learning_rate=0.001, max_iter=100
        )

        self.assertAlmostEqual(model.alpha(), 0.01, places=6)
        self.assertAlmostEqual(model.learning_rate(), 0.001, places=6)
        self.assertEqual(model.max_iter(), 100)

        # Test hidden layer sizes
        self.assertEqual(model.hidden_layer_sizes(), [100])  # Default value

    def test_performance(self):
        """Test model performance on a simple regression task"""
        # Create a simple linear relationship
        X_test = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        y_test = X_test[:, 0] + X_test[:, 1]  # y = x1 + x2

        model = ingenuityml.neural_network.MLPRegressor(
            hidden_layer_sizes=[20, 10],
            max_iter=200,
            random_state=42,
            learning_rate=0.01,
        )
        model.fit(X_test, y_test)

        predictions = model.predict(X_test)

        # Should achieve reasonable accuracy on this simple task
        mse = np.mean((predictions - y_test) ** 2)
        self.assertLess(mse, 1.0)  # Should be reasonably accurate

    def test_dimension_mismatch(self):
        """Test with dimension mismatch - should raise error"""
        model = ingenuityml.neural_network.MLPRegressor()
        model.fit(self.X_small, self.y_small)

        with self.assertRaises(Exception):
            model.predict(np.array([[1, 2, 3]]))  # Wrong number of features


class TestNeuralNetworkUtilities(unittest.TestCase):
    def test_activation_functions_enum(self):
        """Test that activation function enums are accessible"""
        # Test that all activation functions are available
        relu = ingenuityml.neural_network.ActivationFunction.RELU
        tanh = ingenuityml.neural_network.ActivationFunction.TANH
        logistic = ingenuityml.neural_network.ActivationFunction.LOGISTIC
        identity = ingenuityml.neural_network.ActivationFunction.IDENTITY

        self.assertIsNotNone(relu)
        self.assertIsNotNone(tanh)
        self.assertIsNotNone(logistic)
        self.assertIsNotNone(identity)

    def test_solver_enum(self):
        """Test that solver enums are accessible"""
        adam = ingenuityml.neural_network.Solver.ADAM
        sgd = ingenuityml.neural_network.Solver.SGD
        lbfgs = ingenuityml.neural_network.Solver.LBFGS

        self.assertIsNotNone(adam)
        self.assertIsNotNone(sgd)
        self.assertIsNotNone(lbfgs)


if __name__ == "__main__":
    unittest.main()
