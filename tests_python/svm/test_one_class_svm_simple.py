import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import numpy as np

try:
    import auroraml

    print("Successfully imported auroraml")
except ImportError as e:
    print(f"Failed to import auroraml: {e}")
    sys.exit(1)


def test_one_class_svm_basic():
    """Test basic OneClassSVM functionality."""
    print("Testing OneClassSVM basic functionality...")

    # Generate simple inlier data
    np.random.seed(42)
    X_inliers = np.random.randn(50, 2) * 0.5  # Tight cluster around origin

    # Create OneClassSVM
    clf = auroraml.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1.0, random_state=42)

    # Fit on inliers
    clf.fit(X_inliers)

    assert clf.is_fitted(), "Model should be fitted"
    print("✓ Model fitted successfully")

    # Test prediction
    predictions = clf.predict(X_inliers)
    assert len(predictions) == len(X_inliers), "Prediction length mismatch"
    assert all(p in [1, -1] for p in predictions), "Invalid prediction values"
    print("✓ Predictions have correct format")

    # Most inliers should be predicted as inliers (+1)
    inlier_rate = np.sum(predictions == 1) / len(predictions)
    print(f"  Inlier detection rate: {inlier_rate:.3f}")
    assert inlier_rate > 0.7, f"Inlier rate too low: {inlier_rate}"
    print("✓ Reasonable inlier detection rate")


def test_one_class_svm_outliers():
    """Test outlier detection."""
    print("\nTesting outlier detection...")

    # Generate inlier data (tight cluster)
    np.random.seed(42)
    X_inliers = np.random.randn(30, 2) * 0.3

    # Generate clear outliers (far from cluster)
    X_outliers = np.array([[3, 3], [-3, -3], [3, -3], [-3, 3]])

    # Create and fit OneClassSVM
    clf = auroraml.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=2.0, random_state=42)
    clf.fit(X_inliers)

    # Test on inliers
    inlier_predictions = clf.predict(X_inliers)
    inlier_rate = np.sum(inlier_predictions == 1) / len(inlier_predictions)
    print(f"  Inlier detection rate: {inlier_rate:.3f}")

    # Test on outliers
    outlier_predictions = clf.predict(X_outliers)
    outlier_rate = np.sum(outlier_predictions == -1) / len(outlier_predictions)
    print(f"  Outlier detection rate: {outlier_rate:.3f}")

    assert inlier_rate > 0.6, f"Inlier detection rate too low: {inlier_rate}"
    assert outlier_rate > 0.25, f"Outlier detection rate too low: {outlier_rate}"
    print("✓ Reasonable outlier detection")


def test_decision_function():
    """Test decision function."""
    print("\nTesting decision function...")

    np.random.seed(42)
    X = np.random.randn(25, 2) * 0.8

    clf = auroraml.svm.OneClassSVM(nu=0.2, kernel="rbf", gamma=0.5, random_state=42)
    clf.fit(X)

    # Test decision function
    scores = clf.decision_function(X)
    assert len(scores) == len(X), "Score length mismatch"
    print(f"  Decision scores range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")

    # Test score_samples
    score_samples = clf.score_samples(X)
    assert len(score_samples) == len(X), "Score samples length mismatch"

    # For OneClassSVM, these should be the same
    assert np.allclose(scores, score_samples, rtol=1e-10), "Scores should match"
    print("✓ Decision function works correctly")


def test_linear_kernel():
    """Test linear kernel."""
    print("\nTesting linear kernel...")

    np.random.seed(42)
    X = np.random.randn(40, 2) * 0.5

    clf = auroraml.svm.OneClassSVM(nu=0.15, kernel="linear", random_state=42)
    clf.fit(X)

    assert clf.is_fitted(), "Linear kernel model should be fitted"

    predictions = clf.predict(X)
    scores = clf.decision_function(X)

    assert len(predictions) == len(X), "Prediction length mismatch"
    assert len(scores) == len(X), "Score length mismatch"

    inlier_rate = np.sum(predictions == 1) / len(predictions)
    print(f"  Linear kernel inlier rate: {inlier_rate:.3f}")
    assert inlier_rate > 0.6, f"Linear kernel inlier rate too low: {inlier_rate}"
    print("✓ Linear kernel works correctly")


def test_parameters():
    """Test parameter handling."""
    print("\nTesting parameter handling...")

    clf = auroraml.svm.OneClassSVM(
        nu=0.3, gamma=0.7, kernel="rbf", max_iter=500, random_state=123
    )

    # Test get_params
    params = clf.get_params()
    assert "nu" in params, "nu parameter missing"
    assert "gamma" in params, "gamma parameter missing"
    assert "kernel" in params, "kernel parameter missing"
    print("✓ get_params works")

    # Test set_params
    new_params = {"nu": "0.2", "gamma": "1.0", "kernel": "linear", "max_iter": "800"}
    clf.set_params(new_params)

    updated_params = clf.get_params()
    print(f"  Updated params: {updated_params}")
    assert float(updated_params["nu"]) == 0.2, (
        f"nu parameter not updated: {updated_params['nu']}"
    )
    assert float(updated_params["gamma"]) == 1.0, (
        f"gamma parameter not updated: {updated_params['gamma']}"
    )
    assert updated_params["kernel"] == "linear", (
        f"kernel parameter not updated: {updated_params['kernel']}"
    )
    assert int(updated_params["max_iter"]) == 800, (
        f"max_iter parameter not updated: {updated_params['max_iter']}"
    )
    print("✓ set_params works")


def test_threshold():
    """Test threshold functionality."""
    print("\nTesting threshold...")

    np.random.seed(42)
    X = np.random.randn(30, 2) * 0.6

    clf = auroraml.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.8, random_state=42)
    clf.fit(X)

    threshold = clf.get_threshold()
    assert isinstance(threshold, (int, float)), "Threshold should be numeric"
    print(f"  Threshold value: {threshold:.3f}")
    print("✓ Threshold functionality works")


def test_not_fitted_errors():
    """Test error handling for unfitted model."""
    print("\nTesting error handling...")

    clf = auroraml.svm.OneClassSVM(nu=0.1, random_state=42)
    X_test = np.array([[1, 2], [3, 4]])

    try:
        clf.predict(X_test)
        assert False, "Should have raised error"
    except RuntimeError:
        print("✓ Predict raises error when not fitted")

    try:
        clf.decision_function(X_test)
        assert False, "Should have raised error"
    except RuntimeError:
        print("✓ Decision function raises error when not fitted")


if __name__ == "__main__":
    print("Running OneClassSVM validation tests...\n")

    try:
        test_one_class_svm_basic()
        test_one_class_svm_outliers()
        test_decision_function()
        test_linear_kernel()
        test_parameters()
        test_threshold()
        test_not_fitted_errors()

        print("\n🎉 All OneClassSVM tests passed successfully!")
        print("OneClassSVM implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
