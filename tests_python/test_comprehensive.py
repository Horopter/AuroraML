#!/usr/bin/env python3
"""
Comprehensive Test Suite for AuroraML
Tests all major algorithms with real data to verify functionality
"""

import sys
import os
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

def test_linear_models():
    """Test linear models with regression data"""
    print("📦 Testing Linear Models...")
    
    try:
        import auroraml.linear_model as aml_lm
        import auroraml.metrics as aml_metrics
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float64)
        y = X @ np.array([1.0, -2.0, 0.5, 3.0, -1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        
        models = [
            ('LinearRegression', aml_lm.LinearRegression()),
            ('Ridge', aml_lm.Ridge(alpha=0.1)),
            ('Lasso', aml_lm.Lasso(alpha=0.1))
        ]
        
        for name, model in models:
            model.fit(X, y)
            predictions = model.predict(X)
            mse = aml_metrics.mean_squared_error(y, predictions)
            print(f"  ✅ {name}: MSE = {mse:.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Linear models failed: {e}")
        return False

def test_classification():
    """Test classification algorithms"""
    print("📦 Testing Classification...")
    
    try:
        import auroraml.neighbors as aml_neighbors
        import auroraml.tree as aml_tree
        import auroraml.naive_bayes as aml_nb
        import auroraml.metrics as aml_metrics
        
        # Generate classification data
        np.random.seed(42)
        X = np.random.randn(100, 4).astype(np.float64)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
        
        models = [
            ('KNeighborsClassifier', aml_neighbors.KNeighborsClassifier(n_neighbors=5)),
            ('DecisionTreeClassifier', aml_tree.DecisionTreeClassifier(max_depth=5)),
            ('GaussianNB', aml_nb.GaussianNB())
        ]
        
        for name, model in models:
            model.fit(X, y)
            predictions = model.predict(X)
            accuracy = aml_metrics.accuracy_score(y, predictions)
            print(f"  ✅ {name}: Accuracy = {accuracy:.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Classification failed: {e}")
        return False

def test_regression():
    """Test regression algorithms"""
    print("📦 Testing Regression...")
    
    try:
        import auroraml.neighbors as aml_neighbors
        import auroraml.tree as aml_tree
        import auroraml.metrics as aml_metrics
        
        # Generate regression data
        np.random.seed(42)
        X = np.random.randn(100, 4).astype(np.float64)
        y = X @ np.array([1.0, -2.0, 0.5, 1.5]) + 0.1 * np.random.randn(100).astype(np.float64)
        
        models = [
            ('KNeighborsRegressor', aml_neighbors.KNeighborsRegressor(n_neighbors=5)),
            ('DecisionTreeRegressor', aml_tree.DecisionTreeRegressor(max_depth=5))
        ]
        
        for name, model in models:
            model.fit(X, y)
            predictions = model.predict(X)
            mse = aml_metrics.mean_squared_error(y, predictions)
            print(f"  ✅ {name}: MSE = {mse:.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Regression failed: {e}")
        return False

def test_clustering():
    """Test clustering algorithms"""
    print("📦 Testing Clustering...")
    
    try:
        import auroraml.cluster as aml_cluster
        
        # Generate clustering data
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + np.array([2, 2]),
            np.random.randn(50, 2) + np.array([-2, -2])
        ]).astype(np.float64)
        y_dummy = np.zeros((X.shape[0], 1)).astype(np.float64)
        
        models = [
            ('KMeans', aml_cluster.KMeans(n_clusters=2, random_state=42)),
            ('DBSCAN', aml_cluster.DBSCAN(eps=1.0, min_samples=5))
        ]
        
        for name, model in models:
            model.fit(X, y_dummy)
            if hasattr(model, 'predict_labels'):
                labels = model.predict_labels(X)
            else:
                labels = model.labels()
            n_clusters = len(np.unique(labels))
            print(f"  ✅ {name}: Found {n_clusters} clusters")
        
        return True
    except Exception as e:
        print(f"  ❌ Clustering failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing algorithms"""
    print("📦 Testing Preprocessing...")
    
    try:
        import auroraml.preprocessing as aml_pp
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float64)
        y_dummy = np.zeros((X.shape[0], 1)).astype(np.float64)
        
        # Test StandardScaler
        scaler = aml_pp.StandardScaler()
        X_scaled = scaler.fit_transform(X, y_dummy)
        mean_scaled = np.mean(X_scaled, axis=0)
        std_scaled = np.std(X_scaled, axis=0)
        print(f"  ✅ StandardScaler: Mean ≈ {mean_scaled[0]:.4f}, Std ≈ {std_scaled[0]:.4f}")
        
        # Test MinMaxScaler
        minmax = aml_pp.MinMaxScaler()
        X_minmax = minmax.fit_transform(X, y_dummy)
        min_val = np.min(X_minmax)
        max_val = np.max(X_minmax)
        print(f"  ✅ MinMaxScaler: Range = [{min_val:.4f}, {max_val:.4f}]")
        
        return True
    except Exception as e:
        print(f"  ❌ Preprocessing failed: {e}")
        return False

def test_metrics():
    """Test evaluation metrics"""
    print("📦 Testing Metrics...")
    
    try:
        import auroraml.metrics as aml_metrics
        
        # Test classification metrics
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        accuracy = aml_metrics.accuracy_score(y_true, y_pred)
        print(f"  ✅ Accuracy Score: {accuracy:.4f}")
        
        # Test regression metrics
        y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_reg = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        mse = aml_metrics.mean_squared_error(y_true_reg, y_pred_reg)
        print(f"  ✅ Mean Squared Error: {mse:.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Metrics failed: {e}")
        return False

def test_model_selection():
    """Test model selection utilities"""
    print("📦 Testing Model Selection...")
    
    try:
        import auroraml.model_selection as aml_ms
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 4).astype(np.float64)
        y = np.random.randint(0, 2, 100).astype(np.int32)
        
        # Test train_test_split
        X_train, X_test, y_train, y_test = aml_ms.train_test_split(X, y, test_size=0.25, random_state=42)
        print(f"  ✅ Train/Test Split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Test KFold
        kfold = aml_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kfold.split(X))
        print(f"  ✅ KFold: {len(splits)} splits")
        
        return True
    except Exception as e:
        print(f"  ❌ Model Selection failed: {e}")
        return False

def test_random():
    """Test random number generation"""
    print("📦 Testing Random Number Generation...")
    
    try:
        import auroraml.random as aml_random
        
        # Test PCG64
        rng = aml_random.PCG64(seed=42)
        
        # Test uniform random numbers
        uniform_nums = [rng.uniform() for _ in range(1000)]
        uniform_mean = np.mean(uniform_nums)
        print(f"  ✅ PCG64 Uniform: Mean = {uniform_mean:.4f}")
        
        # Test normal random numbers
        normal_nums = [rng.normal() for _ in range(1000)]
        normal_mean = np.mean(normal_nums)
        normal_std = np.std(normal_nums)
        print(f"  ✅ PCG64 Normal: Mean = {normal_mean:.4f}, Std = {normal_std:.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Random failed: {e}")
        return False

def main():
    """Run all comprehensive tests"""
    print("🚀 Starting AuroraML Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Linear Models", test_linear_models),
        ("Classification", test_classification),
        ("Regression", test_regression),
        ("Clustering", test_clustering),
        ("Preprocessing", test_preprocessing),
        ("Metrics", test_metrics),
        ("Model Selection", test_model_selection),
        ("Random Number Generation", test_random)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            print(f"✅ {test_name} passed")
            passed += 1
        else:
            print(f"❌ {test_name} failed")
        print()
    
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All comprehensive tests passed!")
        return True
    else:
        print(f"💥 {total - passed} test(s) failed!")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
