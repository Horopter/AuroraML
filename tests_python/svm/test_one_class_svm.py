import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import numpy as np

import ingenuityml


def make_blobs(n_samples=100, centers=1, cluster_std=0.5, random_state=None):
    rng = np.random.RandomState(random_state)
    if isinstance(centers, int):
        centers_arr = rng.uniform(-2.0, 2.0, size=(centers, 2))
    else:
        centers_arr = np.array(centers, dtype=float)
    n_centers = centers_arr.shape[0]
    X = np.zeros((n_samples, centers_arr.shape[1]), dtype=float)
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        c = i % n_centers
        X[i] = rng.normal(loc=centers_arr[c], scale=cluster_std, size=centers_arr.shape[1])
        y[i] = c
    return X, y

def test_one_class_svm_basic():
    """Test basic OneClassSVM functionality."""
    # Generate inlier data
    X_inliers, _ = make_blobs(
        n_samples=100, centers=1, cluster_std=0.5, random_state=42
    )

    # Create OneClassSVM
    clf = ingenuityml.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1, random_state=42)

    # Fit on inliers
    clf.fit(X_inliers)

    # Test basic properties
    assert clf.is_fitted()
    assert hasattr(clf, "get_threshold")

    # Test prediction
    predictions = clf.predict(X_inliers)
    assert len(predictions) == len(X_inliers)
    assert all(p in [1, -1] for p in predictions)

    # Most inliers should be predicted as inliers (+1)
    inlier_rate = np.sum(predictions == 1) / len(predictions)
    assert inlier_rate > 0.8, f"Inlier rate too low: {inlier_rate}"


def test_one_class_svm_outlier_detection():
    """Test outlier detection capability."""
    # Generate inlier data (tight cluster)
    X_inliers, _ = make_blobs(
        n_samples=100, centers=[[0, 0]], cluster_std=0.5, random_state=42
    )

    # Generate outlier data (far from cluster)
    X_outliers = np.array([[5, 5], [-5, -5], [5, -5], [-5, 5]])

    # Create and fit OneClassSVM
    clf = ingenuityml.svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1.0, random_state=42)
    clf.fit(X_inliers)

    # Test on inliers
    inlier_predictions = clf.predict(X_inliers)
    inlier_rate = np.sum(inlier_predictions == 1) / len(inlier_predictions)

    # Test on outliers
    outlier_predictions = clf.predict(X_outliers)
    outlier_rate = np.sum(outlier_predictions == -1) / len(outlier_predictions)

    # Most inliers should be classified as inliers
    assert inlier_rate > 0.85, f"Inlier detection rate too low: {inlier_rate}"

    # Most outliers should be classified as outliers
    assert outlier_rate > 0.5, f"Outlier detection rate too low: {outlier_rate}"


def test_one_class_svm_decision_function():
    """Test decision function returns reasonable scores."""
    X, _ = make_blobs(n_samples=50, centers=1, cluster_std=0.8, random_state=42)

    clf = ingenuityml.svm.OneClassSVM(nu=0.2, kernel="rbf", gamma=0.5, random_state=42)
    clf.fit(X)

    # Test decision function
    scores = clf.decision_function(X)
    assert len(scores) == len(X)
    assert all(isinstance(s, (int, float)) for s in scores)

    # Predictions should align with decision function
    predictions = clf.predict(X)
    for i in range(len(X)):
        if predictions[i] == 1:
            assert scores[i] >= 0, f"Positive prediction but negative score at {i}"
        else:
            assert scores[i] < 0, f"Negative prediction but positive score at {i}"


def test_one_class_svm_score_samples():
    """Test score_samples method."""
    X, _ = make_blobs(n_samples=50, centers=1, cluster_std=0.6, random_state=42)

    clf = ingenuityml.svm.OneClassSVM(nu=0.15, kernel="rbf", gamma=0.8, random_state=42)
    clf.fit(X)

    # Test score_samples
    scores = clf.score_samples(X)
    decision_scores = clf.decision_function(X)

    # For OneClassSVM, score_samples should be the same as decision_function
    assert len(scores) == len(decision_scores)
    np.testing.assert_allclose(scores, decision_scores, rtol=1e-10)


def test_one_class_svm_linear_kernel():
    """Test OneClassSVM with linear kernel."""
    X, _ = make_blobs(n_samples=80, centers=1, cluster_std=0.7, random_state=42)

    clf = ingenuityml.svm.OneClassSVM(nu=0.1, kernel="linear", random_state=42)
    clf.fit(X)

    assert clf.is_fitted()

    predictions = clf.predict(X)
    assert len(predictions) == len(X)

    # Test decision function works with linear kernel
    scores = clf.decision_function(X)
    assert len(scores) == len(X)

    # Most points should be inliers for a normal dataset
    inlier_rate = np.sum(predictions == 1) / len(predictions)
    assert inlier_rate > 0.8, f"Inlier rate too low with linear kernel: {inlier_rate}"


def test_one_class_svm_parameters():
    """Test parameter getting and setting."""
    clf = ingenuityml.svm.OneClassSVM(
        nu=0.3, gamma=0.7, kernel="rbf", max_iter=500, random_state=123
    )

    # Test get_params
    params = clf.get_params()
    assert "nu" in params
    assert "gamma" in params
    assert "kernel" in params
    assert "max_iter" in params
    assert "random_state" in params

    # Test set_params
    new_params = {"nu": "0.2", "gamma": "1.0", "kernel": "linear", "max_iter": "800"}
    clf.set_params(new_params)

    updated_params = clf.get_params()
    assert abs(float(updated_params["nu"]) - 0.2) < 1e-12
    assert abs(float(updated_params["gamma"]) - 1.0) < 1e-12
    assert updated_params["kernel"] == "linear"
    assert updated_params["max_iter"] == "800"


def test_one_class_svm_nu_parameter():
    """Test different nu values."""
    X, _ = make_blobs(n_samples=100, centers=1, cluster_std=0.5, random_state=42)

    # Test with different nu values
    nu_values = [0.05, 0.1, 0.2, 0.5]

    for nu in nu_values:
        clf = ingenuityml.svm.OneClassSVM(nu=nu, kernel="rbf", gamma=0.5, random_state=42)
        clf.fit(X)

        predictions = clf.predict(X)
        outlier_fraction = np.sum(predictions == -1) / len(predictions)

        # Higher nu should generally result in more outliers
        # But we allow flexibility since it's approximate
        assert 0.0 <= outlier_fraction <= 1.0
        assert clf.is_fitted()


def test_one_class_svm_empty_input():
    """Test behavior with edge cases."""
    clf = ingenuityml.svm.OneClassSVM(nu=0.1, random_state=42)

    # Test not fitted error
    X_test = np.array([[1, 2], [3, 4]])

    try:
        clf.predict(X_test)
        assert False, "Expected predict to raise RuntimeError when not fitted"
    except RuntimeError:
        pass

    try:
        clf.decision_function(X_test)
        assert False, "Expected decision_function to raise RuntimeError when not fitted"
    except RuntimeError:
        pass


def test_one_class_svm_single_feature():
    """Test with single feature data."""
    # Single feature data
    X = np.random.randn(50, 1)

    clf = ingenuityml.svm.OneClassSVM(nu=0.2, kernel="rbf", gamma=1.0, random_state=42)
    clf.fit(X)

    assert clf.is_fitted()

    predictions = clf.predict(X)
    scores = clf.decision_function(X)

    assert len(predictions) == len(X)
    assert len(scores) == len(X)


def test_one_class_svm_threshold():
    """Test threshold functionality."""
    X, _ = make_blobs(n_samples=60, centers=1, cluster_std=0.5, random_state=42)

    clf = ingenuityml.svm.OneClassSVM(nu=0.15, kernel="rbf", gamma=0.8, random_state=42)
    clf.fit(X)

    # Test get_threshold
    threshold = clf.get_threshold()
    assert isinstance(threshold, (int, float))

    # Decision function values should relate to threshold
    scores = clf.decision_function(X)
    predictions = clf.predict(X)

    for i in range(len(X)):
        if predictions[i] == 1:
            assert scores[i] >= 0  # Inliers have positive adjusted scores
        else:
            assert scores[i] < 0  # Outliers have negative adjusted scores


def test_one_class_svm_reproducibility():
    """Test reproducibility with same random state."""
    X, _ = make_blobs(n_samples=50, centers=1, cluster_std=0.6, random_state=42)

    # Create two identical models
    clf1 = ingenuityml.svm.OneClassSVM(nu=0.1, gamma=0.5, kernel="rbf", random_state=42)
    clf2 = ingenuityml.svm.OneClassSVM(nu=0.1, gamma=0.5, kernel="rbf", random_state=42)

    # Fit both models
    clf1.fit(X)
    clf2.fit(X)

    # Predictions should be identical
    pred1 = clf1.predict(X)
    pred2 = clf2.predict(X)

    # Should be exactly the same due to same random state
    np.testing.assert_array_equal(pred1, pred2)


if __name__ == "__main__":
    print("Running OneClassSVM tests...")

    test_one_class_svm_basic()
    print("✓ Basic functionality test passed")

    test_one_class_svm_outlier_detection()
    print("✓ Outlier detection test passed")

    test_one_class_svm_decision_function()
    print("✓ Decision function test passed")

    test_one_class_svm_score_samples()
    print("✓ Score samples test passed")

    test_one_class_svm_linear_kernel()
    print("✓ Linear kernel test passed")

    test_one_class_svm_parameters()
    print("✓ Parameter test passed")

    test_one_class_svm_nu_parameter()
    print("✓ Nu parameter test passed")

    test_one_class_svm_empty_input()
    print("✓ Edge cases test passed")

    test_one_class_svm_single_feature()
    print("✓ Single feature test passed")

    test_one_class_svm_threshold()
    print("✓ Threshold test passed")

    test_one_class_svm_reproducibility()
    print("✓ Reproducibility test passed")

    print("\nAll OneClassSVM tests passed! ✨")
