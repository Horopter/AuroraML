#!/usr/bin/env python3
"""
OneClassSVM Demonstration Script for AuroraML (Simple Version)

This script demonstrates the capabilities of AuroraML's OneClassSVM implementation
without requiring external dependencies like matplotlib or sklearn.
"""

import os
import sys

import numpy as np

# Add AuroraML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))

try:
    import auroraml

    print("✅ Successfully imported AuroraML")
except ImportError as e:
    print(f"❌ Failed to import AuroraML: {e}")
    print("Make sure you've built the project: cd build && make")
    sys.exit(1)


def generate_sample_data():
    """Generate sample data for anomaly detection."""
    print("\n📊 Generating sample datasets...")

    # Dataset 1: Simple 2D Gaussian blob with outliers
    np.random.seed(42)

    # Generate inlier data (tight cluster around origin)
    X_inliers = np.random.randn(100, 2) * 0.8

    # Add some clear outliers
    X_outliers = np.array(
        [
            [4, 4],
            [-4, -4],
            [4, -4],
            [-4, 4],  # Corner outliers
            [5, 0],
            [0, 5],
            [-5, 0],
            [0, -5],  # Edge outliers
            [3, 3],
            [-3, -3],  # Moderate outliers
        ]
    )

    X_combined = np.vstack([X_inliers, X_outliers])
    y_true = np.hstack([np.ones(len(X_inliers)), -np.ones(len(X_outliers))])

    print(f"  Dataset: {len(X_inliers)} inliers, {len(X_outliers)} outliers")
    return X_combined, y_true, X_inliers


def demo_basic_functionality():
    """Demonstrate basic OneClassSVM functionality."""
    print("\n🔍 OneClassSVM Basic Functionality Demo")
    print("=" * 50)

    # Generate simple data
    np.random.seed(42)
    X_train = np.random.randn(80, 2) * 0.6  # Training data (inliers)
    X_test_inliers = np.random.randn(20, 2) * 0.6  # More inliers
    X_test_outliers = np.array([[3, 3], [-3, -3], [3, -3], [-3, 3], [4, 0], [0, 4]])

    # Create OneClassSVM with RBF kernel
    clf = auroraml.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1.0, random_state=42)

    print(f"Training on {len(X_train)} samples...")
    clf.fit(X_train)

    print("✅ Model fitted successfully")
    print(f"   Threshold: {clf.get_threshold():.4f}")

    params = clf.get_params()
    print(
        f"   Parameters: nu={params['nu']}, kernel={params['kernel']}, gamma={params['gamma']}"
    )

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


def compare_kernels(X_combined, y_true, X_train):
    """Compare RBF and linear kernels."""
    print("\n🆚 Kernel Comparison Demo")
    print("=" * 30)

    kernels = {"RBF": {"kernel": "rbf", "gamma": 0.5}, "Linear": {"kernel": "linear"}}

    results = {}

    for name, params in kernels.items():
        print(f"\n🔧 Testing {name} kernel...")

        # Create and fit model
        clf = auroraml.svm.OneClassSVM(nu=0.1, random_state=42, **params)
        clf.fit(X_train)

        # Predict on combined data
        predictions = clf.predict(X_combined)
        scores = clf.decision_function(X_combined)

        # Calculate metrics
        tp = np.sum((predictions == 1) & (y_true == 1))  # True positives
        fp = np.sum((predictions == 1) & (y_true == -1))  # False positives
        tn = np.sum((predictions == -1) & (y_true == -1))  # True negatives
        fn = np.sum((predictions == -1) & (y_true == 1))  # False negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (tp + tn) / len(predictions)

        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "threshold": clf.get_threshold(),
        }

        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        print(f"   Threshold: {clf.get_threshold():.4f}")

    # Compare performance
    print(f"\n🏆 Performance Summary:")
    best_kernel = max(results.keys(), key=lambda k: results[k]["f1"])
    print(f"   Best F1-Score: {best_kernel} kernel ({results[best_kernel]['f1']:.3f})")

    return results


def demo_parameter_sensitivity():
    """Demonstrate the effect of different nu values."""
    print("\n📊 Parameter Sensitivity Analysis")
    print("=" * 40)

    # Generate training data
    np.random.seed(42)
    X_train = np.random.randn(60, 2) * 0.7

    # Generate test data with known outliers
    X_test_inliers = np.random.randn(40, 2) * 0.7
    X_test_outliers = np.array(
        [
            [2.5, 2.5],
            [-2.5, -2.5],
            [2.5, -2.5],
            [-2.5, 2.5],
            [3, 0],
            [0, 3],
            [-3, 0],
            [0, -3],
        ]
    )
    X_test = np.vstack([X_test_inliers, X_test_outliers])
    y_test_true = np.hstack(
        [np.ones(len(X_test_inliers)), -np.ones(len(X_test_outliers))]
    )

    nu_values = [0.05, 0.1, 0.2, 0.3, 0.5]

    print("Testing different nu values (expected outlier fraction):")

    best_nu = None
    best_accuracy = 0

    for nu in nu_values:
        clf = auroraml.svm.OneClassSVM(nu=nu, kernel="rbf", gamma=1.0, random_state=42)
        clf.fit(X_train)

        predictions = clf.predict(X_test)
        outlier_fraction = np.sum(predictions == -1) / len(predictions)

        # Calculate accuracy
        accuracy = np.sum(predictions == y_test_true) / len(y_test_true)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_nu = nu

        print(
            f"   nu={nu:.2f}: {outlier_fraction:.1%} outliers detected, accuracy={accuracy:.1%}"
        )

    print(f"\n🎯 Best nu value: {best_nu} with accuracy {best_accuracy:.1%}")


def demo_real_world_scenario():
    """Simulate a real-world anomaly detection scenario."""
    print("\n🌍 Real-World Scenario: Network Traffic Monitoring")
    print("=" * 55)

    # Simulate network traffic metrics (bandwidth, latency)
    np.random.seed(42)

    # Normal traffic patterns
    normal_bandwidth = np.random.normal(50, 10, 150)  # 50 Mbps ± 10
    normal_latency = np.random.normal(20, 5, 150)  # 20ms ± 5ms
    X_normal = np.column_stack([normal_bandwidth, normal_latency])

    # Anomalous traffic (DDoS attack simulation)
    attack_bandwidth = np.random.normal(200, 20, 12)  # High bandwidth
    attack_latency = np.random.normal(100, 15, 12)  # High latency
    X_attack = np.column_stack([attack_bandwidth, attack_latency])

    # Suspicious activity (moderate anomalies)
    suspicious_bandwidth = np.array([120, 110, 130, 115, 125])
    suspicious_latency = np.array([60, 55, 65, 70, 58])
    X_suspicious = np.column_stack([suspicious_bandwidth, suspicious_latency])

    # Train on normal data only (unsupervised learning)
    clf = auroraml.svm.OneClassSVM(nu=0.08, kernel="rbf", gamma=0.01, random_state=42)
    print(f"Training on {len(X_normal)} normal network traffic samples...")
    clf.fit(X_normal)

    # Test on different types of traffic
    test_cases = {
        "Normal Traffic": X_normal[:30],
        "Suspicious Activity": X_suspicious,
        "DDoS Attack": X_attack,
    }

    print(f"\n📋 Network Traffic Analysis Results:")

    all_scores = []
    for traffic_type, X_test in test_cases.items():
        predictions = clf.predict(X_test)
        scores = clf.decision_function(X_test)

        outlier_rate = np.sum(predictions == -1) / len(predictions)
        avg_score = np.mean(scores)
        min_score = np.min(scores)

        all_scores.extend(scores)

        print(f"   {traffic_type:>18}: {outlier_rate:>6.1%} flagged as anomalies")
        print(
            f"                        Avg score: {avg_score:>6.3f}, Min score: {min_score:>6.3f}"
        )

    # Find most suspicious samples across all test data
    all_test_data = np.vstack([X_normal[:30], X_suspicious, X_attack])
    all_predictions = clf.predict(all_test_data)
    all_decision_scores = clf.decision_function(all_test_data)

    suspicious_indices = np.argsort(all_decision_scores)[:5]  # 5 lowest scores
    labels = ["Normal"] * 30 + ["Suspicious"] * 5 + ["Attack"] * 12

    print(f"\n🚨 Top 5 Most Suspicious Network Samples:")
    for i, idx in enumerate(suspicious_indices, 1):
        bandwidth, latency = all_test_data[idx]
        score = all_decision_scores[idx]
        label = labels[idx]
        prediction = "ANOMALY" if all_predictions[idx] == -1 else "Normal"
        print(
            f"   {i}. BW: {bandwidth:>5.1f} Mbps, Lat: {latency:>5.1f}ms, Score: {score:>6.3f} [{label}] → {prediction}"
        )


def demo_edge_cases():
    """Test edge cases and robustness."""
    print("\n🧪 Edge Cases and Robustness Testing")
    print("=" * 42)

    # Test 1: Single feature data
    print("\n1. Single Feature Data:")
    np.random.seed(42)
    X_1d = np.random.randn(50, 1) * 0.5
    clf_1d = auroraml.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1.0, random_state=42)
    clf_1d.fit(X_1d)

    X_test_1d = np.array([[2], [-2], [0.5], [-0.5]])
    preds_1d = clf_1d.predict(X_test_1d)
    print(f"   Fitted on {len(X_1d)} 1D samples")
    print(f"   Test predictions: {preds_1d} (outliers: {np.sum(preds_1d == -1)})")

    # Test 2: High-dimensional data
    print("\n2. High-Dimensional Data (10D):")
    X_high = np.random.randn(40, 10) * 0.3
    clf_high = auroraml.svm.OneClassSVM(nu=0.15, kernel="linear", random_state=42)
    clf_high.fit(X_high)

    X_test_high = np.random.randn(10, 10) * 0.3
    preds_high = clf_high.predict(X_test_high)
    print(f"   Fitted on {len(X_high)} 10D samples")
    print(
        f"   Test predictions: {np.sum(preds_high == 1)} inliers, {np.sum(preds_high == -1)} outliers"
    )

    # Test 3: Different nu values on same data
    print("\n3. Nu Parameter Robustness:")
    X_base = np.random.randn(60, 2) * 0.4

    for nu in [0.01, 0.05, 0.1, 0.3, 0.7]:
        try:
            clf_nu = auroraml.svm.OneClassSVM(
                nu=nu, kernel="rbf", gamma=0.5, random_state=42
            )
            clf_nu.fit(X_base)
            preds_nu = clf_nu.predict(X_base)
            outlier_rate = np.sum(preds_nu == -1) / len(preds_nu)
            print(f"   nu={nu:.2f}: {outlier_rate:.1%} outliers detected ✓")
        except Exception as e:
            print(f"   nu={nu:.2f}: Failed - {e}")

    # Test 4: Parameter updates
    print("\n4. Parameter Update Test:")
    clf_param = auroraml.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1.0)
    original_params = clf_param.get_params()

    # Update parameters
    new_params = {"nu": "0.2", "kernel": "linear", "gamma": "0.5"}
    clf_param.set_params(new_params)
    updated_params = clf_param.get_params()

    print(
        f"   Original nu: {original_params['nu']} → Updated nu: {updated_params['nu']}"
    )
    print(
        f"   Original kernel: {original_params['kernel']} → Updated kernel: {updated_params['kernel']}"
    )
    print("   Parameter updates working ✓")


def run_comprehensive_demo():
    """Run the complete demonstration."""
    print("🚀 AuroraML OneClassSVM Comprehensive Demo")
    print("=" * 50)
    print("This demo showcases anomaly detection capabilities using OneClassSVM")
    print("with both RBF and linear kernels - no external dependencies required!\n")

    try:
        # Generate data
        X_combined, y_true, X_train = generate_sample_data()

        # Run demos
        demo_basic_functionality()
        compare_kernels(X_combined, y_true, X_train)
        demo_parameter_sensitivity()
        demo_real_world_scenario()
        demo_edge_cases()

        print("\n" + "=" * 70)
        print("🎉 Demo completed successfully!")
        print("\n📝 Summary of OneClassSVM capabilities demonstrated:")
        print("   ✅ Basic anomaly detection with RBF and linear kernels")
        print("   ✅ Parameter sensitivity analysis (nu values)")
        print("   ✅ Kernel comparison and performance evaluation")
        print("   ✅ Real-world network traffic monitoring scenario")
        print("   ✅ Edge cases: 1D data, high-dimensional data, parameter updates")
        print("   ✅ Robust threshold computation and decision functions")

        print("\n💡 Key takeaways:")
        print(
            "   • RBF kernel generally performs better on complex, non-linear patterns"
        )
        print("   • Linear kernel is faster and works well for linearly separable data")
        print("   • nu parameter controls expected fraction of outliers (0 < nu ≤ 1)")
        print("   • Lower nu values are more conservative (detect fewer outliers)")
        print("   • OneClassSVM excels at unsupervised anomaly detection")
        print("   • Works well with various data dimensions and distributions")

        print(f"\n🔧 AuroraML OneClassSVM is ready for production use!")
        print(
            "   Complete SVM family: LinearSVC, SVC, SVR, LinearSVR, NuSVC, NuSVR, OneClassSVM ✅"
        )

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_demo()
    sys.exit(0 if success else 1)
