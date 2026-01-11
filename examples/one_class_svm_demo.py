#!/usr/bin/env python3
"""
OneClassSVM Demonstration Script for IngenuityML

This script demonstrates the capabilities of IngenuityML's OneClassSVM implementation,
showcasing both RBF and linear kernels for anomaly detection tasks.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import classification_report, confusion_matrix

# Add IngenuityML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))

try:
    import ingenuityml

    print("✅ Successfully imported IngenuityML")
except ImportError as e:
    print(f"❌ Failed to import IngenuityML: {e}")
    print("Make sure you've built the project: cd build && make")
    sys.exit(1)


def generate_sample_data():
    """Generate sample data for anomaly detection."""
    print("\n📊 Generating sample datasets...")

    # Dataset 1: Simple 2D Gaussian blob with outliers
    np.random.seed(42)
    X_inliers, _ = make_blobs(
        n_samples=150,
        centers=1,
        cluster_std=0.8,
        center_box=(0.0, 0.0),
        random_state=42,
    )

    # Add some clear outliers
    X_outliers = np.array(
        [
            [4, 4],
            [-4, -4],
            [4, -4],
            [-4, 4],  # Corner outliers
            [6, 0],
            [0, 6],
            [-6, 0],
            [0, -6],  # Edge outliers
            [3, 3],
            [-3, -3],  # Moderate outliers
        ]
    )

    X_combined = np.vstack([X_inliers, X_outliers])
    y_true = np.hstack([np.ones(len(X_inliers)), -np.ones(len(X_outliers))])

    print(f"  Dataset 1: {len(X_inliers)} inliers, {len(X_outliers)} outliers")

    # Dataset 2: Concentric circles (more complex)
    X_circles, _ = make_circles(n_samples=100, factor=0.3, noise=0.05, random_state=42)
    # Take only inner circle as inliers
    inner_mask = np.linalg.norm(X_circles, axis=1) < 0.6
    X_circles_inliers = X_circles[inner_mask]

    print(f"  Dataset 2: {len(X_circles_inliers)} inliers (inner circle)")

    return {
        "blob": (X_combined, y_true, X_inliers),
        "circles": (X_circles, X_circles_inliers),
    }


def demo_basic_functionality():
    """Demonstrate basic OneClassSVM functionality."""
    print("\n🔍 OneClassSVM Basic Functionality Demo")
    print("=" * 50)

    # Generate simple data
    np.random.seed(42)
    X_train = np.random.randn(100, 2) * 0.5  # Training data (inliers)
    X_test_inliers = np.random.randn(30, 2) * 0.5  # More inliers
    X_test_outliers = np.array([[3, 3], [-3, -3], [3, -3], [-3, 3], [4, 0], [0, 4]])

    # Create OneClassSVM with RBF kernel
    clf = ingenuityml.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1.0, random_state=42)

    print(f"Training on {len(X_train)} samples...")
    clf.fit(X_train)

    print("✅ Model fitted successfully")
    print(f"   Threshold: {clf.get_threshold():.4f}")
    print(f"   Parameters: {clf.get_params()}")

    # Test on inliers
    inlier_preds = clf.predict(X_test_inliers)
    inlier_scores = clf.decision_function(X_test_inliers)
    inlier_rate = np.sum(inlier_preds == 1) / len(inlier_preds)

    # Test on outliers
    outlier_preds = clf.predict(X_test_outliers)
    outlier_scores = clf.decision_function(X_test_outliers)
    outlier_detection_rate = np.sum(outlier_preds == -1) / len(outlier_preds)

    print(f"\n📈 Results:")
    print(f"   Inlier detection rate: {inlier_rate:.1%}")
    print(f"   Outlier detection rate: {outlier_detection_rate:.1%}")
    print(
        f"   Inlier scores range: [{np.min(inlier_scores):.3f}, {np.max(inlier_scores):.3f}]"
    )
    print(
        f"   Outlier scores range: [{np.min(outlier_scores):.3f}, {np.max(outlier_scores):.3f}]"
    )


def compare_kernels(data):
    """Compare RBF and linear kernels."""
    print("\n🆚 Kernel Comparison Demo")
    print("=" * 30)

    X_combined, y_true, X_train = data["blob"]

    kernels = {"RBF": {"kernel": "rbf", "gamma": 0.5}, "Linear": {"kernel": "linear"}}

    results = {}

    for name, params in kernels.items():
        print(f"\n🔧 Testing {name} kernel...")

        # Create and fit model
        clf = ingenuityml.svm.OneClassSVM(nu=0.1, random_state=42, **params)
        clf.fit(X_train)

        # Predict on combined data
        predictions = clf.predict(X_combined)
        scores = clf.decision_function(X_combined)

        # Calculate metrics
        tp = np.sum(
            (predictions == 1) & (y_true == 1)
        )  # True positives (inliers correctly identified)
        fp = np.sum(
            (predictions == 1) & (y_true == -1)
        )  # False positives (outliers as inliers)
        tn = np.sum(
            (predictions == -1) & (y_true == -1)
        )  # True negatives (outliers correctly identified)
        fn = np.sum(
            (predictions == -1) & (y_true == 1)
        )  # False negatives (inliers as outliers)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results[name] = {
            "predictions": predictions,
            "scores": scores,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": clf.get_threshold(),
        }

        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        print(f"   Threshold: {clf.get_threshold():.4f}")

    # Compare performance
    print(f"\n🏆 Best performing kernel:")
    best_kernel = max(results.keys(), key=lambda k: results[k]["f1"])
    print(f"   {best_kernel} kernel with F1-Score: {results[best_kernel]['f1']:.3f}")

    return results


def demo_parameter_sensitivity():
    """Demonstrate the effect of different nu values."""
    print("\n📊 Parameter Sensitivity Analysis")
    print("=" * 40)

    # Generate training data
    np.random.seed(42)
    X_train = np.random.randn(80, 2) * 0.6

    # Generate test data with known outliers
    X_test_inliers = np.random.randn(50, 2) * 0.6
    X_test_outliers = np.array(
        [[2.5, 2.5], [-2.5, -2.5], [2.5, -2.5], [-2.5, 2.5], [3, 0], [0, 3]]
    )
    X_test = np.vstack([X_test_inliers, X_test_outliers])
    y_test_true = np.hstack(
        [np.ones(len(X_test_inliers)), -np.ones(len(X_test_outliers))]
    )

    nu_values = [0.05, 0.1, 0.2, 0.3, 0.5]

    print("Testing different nu values (expected outlier fraction):")
    for nu in nu_values:
        clf = ingenuityml.svm.OneClassSVM(nu=nu, kernel="rbf", gamma=1.0, random_state=42)
        clf.fit(X_train)

        predictions = clf.predict(X_test)
        outlier_fraction = np.sum(predictions == -1) / len(predictions)

        # Calculate accuracy
        accuracy = np.sum(predictions == y_test_true) / len(y_test_true)

        print(
            f"   nu={nu:.2f}: {outlier_fraction:.1%} outliers detected, accuracy={accuracy:.1%}"
        )


def demo_real_world_scenario():
    """Simulate a real-world anomaly detection scenario."""
    print("\n🌍 Real-World Scenario: Server Performance Monitoring")
    print("=" * 60)

    # Simulate server metrics (CPU usage, Memory usage)
    np.random.seed(42)

    # Normal operation data
    normal_cpu = np.random.normal(30, 8, 200)  # 30% CPU ± 8%
    normal_memory = np.random.normal(45, 10, 200)  # 45% Memory ± 10%
    X_normal = np.column_stack([normal_cpu, normal_memory])

    # Anomalous data (high resource usage)
    anomaly_cpu = np.random.normal(85, 5, 15)  # 85% CPU (high)
    anomaly_memory = np.random.normal(90, 5, 15)  # 90% Memory (high)
    X_anomaly = np.column_stack([anomaly_cpu, anomaly_memory])

    # Some edge cases (moderate anomalies)
    edge_cpu = np.array([70, 75, 65, 80, 72])
    edge_memory = np.array([75, 70, 80, 85, 78])
    X_edge = np.column_stack([edge_cpu, edge_memory])

    # Train on normal data only
    clf = ingenuityml.svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1, random_state=42)
    print(f"Training on {len(X_normal)} normal server performance samples...")
    clf.fit(X_normal)

    # Test on all data
    X_test = np.vstack(
        [X_normal[:50], X_edge, X_anomaly]
    )  # Take subset of normal for testing
    test_labels = ["Normal"] * 50 + ["Edge Case"] * 5 + ["Anomaly"] * 15

    predictions = clf.predict(X_test)
    scores = clf.decision_function(X_test)

    print(f"\n📋 Detection Results:")
    print(f"   Total samples tested: {len(X_test)}")

    # Analyze by category
    categories = {"Normal": (0, 50), "Edge Case": (50, 55), "Anomaly": (55, 70)}

    for category, (start, end) in categories.items():
        cat_preds = predictions[start:end]
        cat_scores = scores[start:end]
        outlier_rate = np.sum(cat_preds == -1) / len(cat_preds)
        avg_score = np.mean(cat_scores)

        print(
            f"   {category:>12}: {outlier_rate:>6.1%} flagged as outliers (avg score: {avg_score:>6.3f})"
        )

    # Find most suspicious samples
    suspicious_indices = np.argsort(scores)[:5]  # 5 lowest scores
    print(f"\n🚨 Top 5 Most Suspicious Samples:")
    for i, idx in enumerate(suspicious_indices, 1):
        cpu, memory = X_test[idx]
        score = scores[idx]
        label = test_labels[idx]
        print(
            f"   {i}. CPU: {cpu:5.1f}%, Memory: {memory:5.1f}%, Score: {score:6.3f} [{label}]"
        )


def create_visualization_data(data):
    """Create data for visualization (returns info for plotting)."""
    print("\n📈 Preparing Visualization Data")
    print("=" * 35)

    X_combined, y_true, X_train = data["blob"]

    # Fit OneClassSVM
    clf = ingenuityml.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5, random_state=42)
    clf.fit(X_train)

    # Create a grid for decision boundary
    h = 0.1
    x_min, x_max = X_combined[:, 0].min() - 1, X_combined[:, 0].max() + 1
    y_min, y_max = X_combined[:, 1].min() - 1, X_combined[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    print(f"Computing decision function for {len(grid_points)} grid points...")
    try:
        Z = clf.decision_function(grid_points)
        Z = Z.reshape(xx.shape)

        predictions = clf.predict(X_combined)

        print("✅ Visualization data prepared")
        print("   Note: To see the plot, uncomment the matplotlib visualization code")

        # Return data for potential plotting
        return {
            "xx": xx,
            "yy": yy,
            "Z": Z,
            "X_combined": X_combined,
            "y_true": y_true,
            "predictions": predictions,
            "X_train": X_train,
        }

    except Exception as e:
        print(f"⚠️  Visualization preparation failed: {e}")
        return None


def run_comprehensive_demo():
    """Run the complete demonstration."""
    print("🚀 IngenuityML OneClassSVM Comprehensive Demo")
    print("=" * 50)
    print("This demo showcases anomaly detection capabilities using OneClassSVM")
    print("with both RBF and linear kernels.\n")

    try:
        # Generate data
        datasets = generate_sample_data()

        # Run demos
        demo_basic_functionality()
        compare_kernels(datasets)
        demo_parameter_sensitivity()
        demo_real_world_scenario()

        # Visualization data
        viz_data = create_visualization_data(datasets)

        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\n📝 Summary of OneClassSVM capabilities demonstrated:")
        print("   ✅ Basic anomaly detection with RBF and linear kernels")
        print("   ✅ Parameter sensitivity analysis (nu values)")
        print("   ✅ Kernel comparison and performance evaluation")
        print("   ✅ Real-world server monitoring scenario")
        print("   ✅ Decision boundary computation for visualization")

        print("\n💡 Key takeaways:")
        print("   • RBF kernel generally performs better on complex, non-linear data")
        print("   • Linear kernel is faster and works well for linearly separable data")
        print(
            "   • nu parameter controls the expected fraction of outliers (0 < nu ≤ 1)"
        )
        print("   • Lower nu values are more conservative (fewer outliers detected)")
        print("   • OneClassSVM is effective for unsupervised anomaly detection")

        print(f"\n🔧 IngenuityML OneClassSVM is ready for production use!")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_demo()
    sys.exit(0 if success else 1)
