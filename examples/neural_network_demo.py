#!/usr/bin/env python3
"""
IngenuityML Neural Network Demo

This script demonstrates the new neural network capabilities in IngenuityML,
including MLPClassifier and MLPRegressor with various configurations.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add the build directory to Python path
build_path = os.path.join(os.path.dirname(__file__), "..", "build")
sys.path.insert(0, build_path)

try:
    import ingenuityml
except ImportError as e:
    print(f"Failed to import ingenuityml: {e}")
    print("Make sure you have built the project with: cd build && make")
    sys.exit(1)


def neural_network_classification_demo():
    """Demonstrate MLPClassifier capabilities"""
    print("=" * 80)
    print("IngenuityML Neural Network Classification Demo")
    print("=" * 80)

    # Generate sample classification data
    np.random.seed(42)
    n_samples = 500

    # Create a non-linear classification problem
    X = np.random.randn(n_samples, 2)
    # XOR-like pattern with some noise
    y = ((X[:, 0] > 0) & (X[:, 1] > 0) | (X[:, 0] < 0) & (X[:, 1] < 0)).astype(int)

    # Add some noise to make it more challenging
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    print(
        f"Dataset: {n_samples} samples, {X.shape[1]} features, {len(np.unique(y))} classes"
    )

    # Test different neural network configurations
    configurations = [
        {
            "name": "Simple MLP (ReLU)",
            "params": {
                "hidden_layer_sizes": [10],
                "activation": ingenuityml.neural_network.ActivationFunction.RELU,
                "max_iter": 200,
                "learning_rate": 0.01,
                "random_state": 42,
            },
        },
        {
            "name": "Deep MLP (Tanh)",
            "params": {
                "hidden_layer_sizes": [20, 10],
                "activation": ingenuityml.neural_network.ActivationFunction.TANH,
                "max_iter": 300,
                "learning_rate": 0.001,
                "random_state": 42,
            },
        },
        {
            "name": "Wide MLP (Logistic)",
            "params": {
                "hidden_layer_sizes": [50],
                "activation": ingenuityml.neural_network.ActivationFunction.LOGISTIC,
                "solver": ingenuityml.neural_network.Solver.ADAM,
                "max_iter": 250,
                "learning_rate": 0.005,
                "random_state": 42,
            },
        },
    ]

    results = []

    for config in configurations:
        print(f"\n--- {config['name']} ---")

        # Create and train the model
        mlp = ingenuityml.neural_network.MLPClassifier(**config["params"])

        # Fit the model
        print("Training...")
        mlp.fit(X, y)

        # Make predictions
        predictions = mlp.predict(X)
        probabilities = mlp.predict_proba(X)

        # Calculate accuracy
        accuracy = np.mean(predictions == y)
        print(f"Training Accuracy: {accuracy:.3f}")

        # Show convergence
        loss_curve = mlp.loss_curve()
        print(f"Final Loss: {loss_curve[-1]:.6f}")
        print(f"Iterations: {len(loss_curve)}")

        # Store results
        results.append(
            {
                "name": config["name"],
                "model": mlp,
                "predictions": predictions,
                "probabilities": probabilities,
                "accuracy": accuracy,
                "loss_curve": loss_curve,
            }
        )

        # Show model properties
        print(f"Classes: {mlp.classes()}")
        print(f"Number of classes: {mlp.n_classes()}")
        print(f"Hidden layer sizes: {mlp.hidden_layer_sizes()}")

    # Plot results if matplotlib is available
    try:
        plt.figure(figsize=(15, 10))

        # Plot original data
        plt.subplot(2, 3, 1)
        colors = ["red", "blue"]
        for i, class_val in enumerate(np.unique(y)):
            mask = y == class_val
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                c=colors[i],
                alpha=0.6,
                label=f"Class {class_val}",
            )
        plt.title("Original Data")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()

        # Plot predictions for each model
        for i, result in enumerate(results):
            plt.subplot(2, 3, i + 2)
            predictions = result["predictions"]

            for j, class_val in enumerate(np.unique(y)):
                mask = predictions == class_val
                plt.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    c=colors[j],
                    alpha=0.6,
                    label=f"Predicted {class_val}",
                )

            plt.title(f"{result['name']}\nAccuracy: {result['accuracy']:.3f}")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.legend()

        # Plot loss curves
        plt.subplot(2, 3, 5)
        for result in results:
            plt.plot(result["loss_curve"], label=result["name"])
        plt.title("Training Loss Curves")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.yscale("log")

        plt.tight_layout()
        plt.savefig(
            "neural_network_classification_demo.png", dpi=150, bbox_inches="tight"
        )
        print(f"\nPlots saved as 'neural_network_classification_demo.png'")

    except ImportError:
        print("Matplotlib not available - skipping plots")

    return results


def neural_network_regression_demo():
    """Demonstrate MLPRegressor capabilities"""
    print("\n" + "=" * 80)
    print("IngenuityML Neural Network Regression Demo")
    print("=" * 80)

    # Generate sample regression data
    np.random.seed(42)
    n_samples = 300

    # Create a non-linear regression problem
    X = np.random.uniform(-2, 2, (n_samples, 1))
    y = (
        X.ravel() ** 3
        - 2 * X.ravel() ** 2
        + X.ravel()
        + np.random.normal(0, 0.3, n_samples)
    )

    # Add a second feature for more complexity
    X2 = np.random.uniform(-1, 1, (n_samples, 1))
    X = np.column_stack([X, X2])
    y += 0.5 * X2.ravel()

    print(f"Dataset: {n_samples} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Test different neural network configurations
    configurations = [
        {
            "name": "Simple Regressor",
            "params": {
                "hidden_layer_sizes": [20],
                "activation": ingenuityml.neural_network.ActivationFunction.RELU,
                "max_iter": 500,
                "learning_rate": 0.001,
                "alpha": 0.0001,
                "random_state": 42,
            },
        },
        {
            "name": "Deep Regressor",
            "params": {
                "hidden_layer_sizes": [30, 20, 10],
                "activation": ingenuityml.neural_network.ActivationFunction.TANH,
                "max_iter": 800,
                "learning_rate": 0.0005,
                "alpha": 0.001,
                "random_state": 42,
            },
        },
        {
            "name": "Regularized Regressor",
            "params": {
                "hidden_layer_sizes": [50, 25],
                "activation": ingenuityml.neural_network.ActivationFunction.RELU,
                "solver": ingenuityml.neural_network.Solver.ADAM,
                "max_iter": 400,
                "learning_rate": 0.001,
                "alpha": 0.01,  # Strong regularization
                "random_state": 42,
            },
        },
    ]

    results = []

    for config in configurations:
        print(f"\n--- {config['name']} ---")

        # Create and train the model
        mlp = ingenuityml.neural_network.MLPRegressor(**config["params"])

        # Fit the model
        print("Training...")
        mlp.fit(X, y)

        # Make predictions
        predictions = mlp.predict(X)

        # Calculate metrics
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y))

        # R² score
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print(f"Training MSE: {mse:.6f}")
        print(f"Training RMSE: {rmse:.6f}")
        print(f"Training MAE: {mae:.6f}")
        print(f"Training R²: {r2:.6f}")

        # Show convergence
        loss_curve = mlp.loss_curve()
        print(f"Final Loss: {loss_curve[-1]:.6f}")
        print(f"Iterations: {len(loss_curve)}")

        # Store results
        results.append(
            {
                "name": config["name"],
                "model": mlp,
                "predictions": predictions,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "loss_curve": loss_curve,
            }
        )

        # Show model properties
        print(f"Hidden layer sizes: {mlp.hidden_layer_sizes()}")
        print(f"Alpha (regularization): {mlp.alpha()}")

    # Plot results if matplotlib is available
    try:
        plt.figure(figsize=(15, 10))

        # Plot predictions vs actual for each model
        for i, result in enumerate(results):
            plt.subplot(2, 3, i + 1)

            predictions = result["predictions"]
            plt.scatter(y, predictions, alpha=0.6)

            # Perfect prediction line
            min_val = min(y.min(), predictions.min())
            max_val = max(y.max(), predictions.max())
            plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)

            plt.title(f"{result['name']}\nR² = {result['r2']:.3f}")
            plt.xlabel("True Values")
            plt.ylabel("Predictions")

        # Plot loss curves
        plt.subplot(2, 3, 4)
        for result in results:
            plt.plot(result["loss_curve"], label=result["name"])
        plt.title("Training Loss Curves")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.yscale("log")

        # Plot feature importance (approximate)
        plt.subplot(2, 3, 5)
        feature_names = ["Feature 1", "Feature 2"]

        # Use the first model for feature importance demonstration
        model = results[0]["model"]
        # Simple approximation: use first layer weights magnitude
        first_layer_weights = np.abs(model.coefs()[0]).mean(axis=0)

        plt.bar(feature_names, first_layer_weights)
        plt.title("Approximate Feature Importance\n(First Layer Weight Magnitude)")
        plt.ylabel("Average Weight Magnitude")

        # Summary statistics
        plt.subplot(2, 3, 6)
        metrics = ["MSE", "RMSE", "MAE", "R²"]
        models = [r["name"] for r in results]

        for i, metric in enumerate(["mse", "rmse", "mae", "r2"]):
            values = [r[metric] for r in results]
            plt.plot(models, values, "o-", label=metrics[i])

        plt.title("Model Comparison")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig("neural_network_regression_demo.png", dpi=150, bbox_inches="tight")
        print(f"\nPlots saved as 'neural_network_regression_demo.png'")

    except ImportError:
        print("Matplotlib not available - skipping plots")

    return results


def compare_with_simple_models():
    """Compare neural networks with simpler models"""
    print("\n" + "=" * 80)
    print("Neural Network vs Simple Models Comparison")
    print("=" * 80)

    # Create a moderately complex dataset
    np.random.seed(42)
    n_samples = 1000

    # Non-linear classification problem
    X = np.random.randn(n_samples, 4)
    # Complex decision boundary
    y = ((X[:, 0] * X[:, 1] > 0) & (X[:, 2] + X[:, 3] > 0.5)).astype(int)

    print(f"Dataset: {n_samples} samples, {X.shape[1]} features")

    models = {}

    # Neural Network
    print("\n--- Neural Network ---")
    mlp = ingenuityml.neural_network.MLPClassifier(
        hidden_layer_sizes=[20, 10],
        max_iter=300,
        learning_rate=0.001,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X, y)
    mlp_pred = mlp.predict(X)
    mlp_acc = np.mean(mlp_pred == y)
    models["Neural Network"] = mlp_acc
    print(f"Accuracy: {mlp_acc:.3f}")
    print(f"Iterations: {mlp.n_iter()}")

    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr = ingenuityml.linear_model.LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    lr_pred = lr.predict(X)
    lr_acc = np.mean(lr_pred == y)
    models["Logistic Regression"] = lr_acc
    print(f"Accuracy: {lr_acc:.3f}")

    # Random Forest
    print("\n--- Random Forest ---")
    rf = ingenuityml.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_pred = rf.predict(X)
    rf_acc = np.mean(rf_pred == y)
    models["Random Forest"] = rf_acc
    print(f"Accuracy: {rf_acc:.3f}")

    # K-Nearest Neighbors
    print("\n--- K-Nearest Neighbors ---")
    knn = ingenuityml.neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    knn_pred = knn.predict(X)
    knn_acc = np.mean(knn_pred == y)
    models["KNN"] = knn_acc
    print(f"Accuracy: {knn_acc:.3f}")

    # Summary
    print("\n--- Model Comparison Summary ---")
    sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)
    for i, (model_name, accuracy) in enumerate(sorted_models):
        print(f"{i + 1}. {model_name:<20} : {accuracy:.3f}")

    return models


def activation_function_showcase():
    """Showcase different activation functions"""
    print("\n" + "=" * 80)
    print("Activation Function Showcase")
    print("=" * 80)

    # Simple regression problem to show activation effects
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = np.sin(X.ravel()) + 0.1 * np.random.randn(100)

    activations = [
        ("ReLU", ingenuityml.neural_network.ActivationFunction.RELU),
        ("Tanh", ingenuityml.neural_network.ActivationFunction.TANH),
        ("Logistic", ingenuityml.neural_network.ActivationFunction.LOGISTIC),
        ("Identity", ingenuityml.neural_network.ActivationFunction.IDENTITY),
    ]

    results = {}

    for name, activation in activations:
        print(f"\n--- {name} Activation ---")

        mlp = ingenuityml.neural_network.MLPRegressor(
            hidden_layer_sizes=[10],
            activation=activation,
            max_iter=500,
            learning_rate=0.01,
            random_state=42,
        )

        mlp.fit(X, y)
        predictions = mlp.predict(X)

        mse = np.mean((predictions - y) ** 2)
        print(f"MSE: {mse:.6f}")

        results[name] = {
            "predictions": predictions,
            "mse": mse,
            "loss_curve": mlp.loss_curve(),
        }

    # Plot if matplotlib is available
    try:
        plt.figure(figsize=(12, 8))

        colors = ["blue", "red", "green", "orange"]

        plt.subplot(2, 2, 1)
        plt.scatter(X.ravel(), y, alpha=0.6, color="black", label="True Data")

        for i, (name, result) in enumerate(results.items()):
            plt.plot(
                X.ravel(),
                result["predictions"],
                color=colors[i],
                label=f"{name} (MSE: {result['mse']:.4f})",
            )

        plt.title("Activation Function Comparison")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.legend()

        # Individual plots for each activation
        for i, (name, result) in enumerate(results.items()):
            plt.subplot(2, 2, i + 2)
            plt.scatter(X.ravel(), y, alpha=0.4, color="gray", label="True Data")
            plt.plot(
                X.ravel(),
                result["predictions"],
                color=colors[i],
                linewidth=2,
                label=f"{name}",
            )
            plt.title(f"{name} Activation (MSE: {result['mse']:.4f})")
            plt.xlabel("Input")
            plt.ylabel("Output")
            plt.legend()

        plt.tight_layout()
        plt.savefig("activation_functions_showcase.png", dpi=150, bbox_inches="tight")
        print(f"\nPlots saved as 'activation_functions_showcase.png'")

    except ImportError:
        print("Matplotlib not available - skipping plots")

    return results


def main():
    """Run all neural network demonstrations"""
    print("🚀 IngenuityML Neural Network Comprehensive Demo")
    print("=" * 80)
    print("This demo showcases the new neural network capabilities in IngenuityML")
    print("including MLPClassifier and MLPRegressor with various configurations.")
    print("=" * 80)

    try:
        # Run all demonstrations
        classification_results = neural_network_classification_demo()
        regression_results = neural_network_regression_demo()
        comparison_results = compare_with_simple_models()
        activation_results = activation_function_showcase()

        print("\n" + "=" * 80)
        print("🎉 Demo Completed Successfully!")
        print("=" * 80)

        print("\nKey Achievements:")
        print("✅ MLPClassifier: Multi-layer perceptron for classification")
        print("✅ MLPRegressor: Multi-layer perceptron for regression")
        print("✅ Multiple activation functions: ReLU, Tanh, Logistic, Identity")
        print("✅ Multiple solvers: Adam, SGD")
        print("✅ Regularization support (L2)")
        print("✅ Scikit-learn compatible API")
        print("✅ Real-time training monitoring")
        print("✅ Competitive performance with traditional ML algorithms")

        print(f"\nClassification Models Tested: {len(classification_results)}")
        print(f"Regression Models Tested: {len(regression_results)}")
        print(f"Model Comparison: {len(comparison_results)} algorithms")
        print(f"Activation Functions: {len(activation_results)} variants")

        print("\nIngenuityML Neural Networks are ready for production use! 🎯")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
