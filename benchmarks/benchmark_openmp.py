#!/usr/bin/env python3
"""
Focused Performance Benchmark for CxML Parallel Processing
Tests the performance improvements from OpenMP parallelization
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

# Import CxML
try:
    import cxml
    CXML_AVAILABLE = True
    print("✅ CxML successfully imported")
except ImportError as e:
    print(f"❌ CxML import failed: {e}")
    CXML_AVAILABLE = False

def benchmark_knn_performance():
    """Benchmark KNN performance with different dataset sizes"""
    print("\n🔍 Benchmarking K-Nearest Neighbors Performance")
    print("-" * 50)
    
    results = []
    
    # Test different dataset sizes
    sizes = [
        (500, 10, "Small"),
        (2000, 20, "Medium"), 
        (5000, 50, "Large"),
        (10000, 100, "Very Large")
    ]
    
    for n_samples, n_features, size_name in sizes:
        print(f"\n📏 Testing {size_name} dataset: {n_samples} samples, {n_features} features")
        
        # Generate classification data
        X_clf, y_clf = make_classification(
            n_samples=n_samples, n_features=n_features, n_classes=3, 
            n_informative=max(3, n_features//2), random_state=42
        )
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42
        )
        
        # Generate regression data
        X_reg, y_reg = make_regression(
            n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42
        )
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Benchmark KNN Classifier
        print(f"  🎯 KNN Classifier...")
        
        # sklearn
        sklearn_knn_clf = KNeighborsClassifier(n_neighbors=5)
        start_time = time.time()
        sklearn_knn_clf.fit(X_train_clf, y_train_clf)
        sklearn_fit_time = time.time() - start_time
        
        start_time = time.time()
        sklearn_pred = sklearn_knn_clf.predict(X_test_clf)
        sklearn_pred_time = time.time() - start_time
        sklearn_score = accuracy_score(y_test_clf, sklearn_pred)
        
        results.append({
            'algorithm': 'KNN_Classifier',
            'library': 'scikit-learn',
            'dataset_size': size_name,
            'n_samples': n_samples,
            'n_features': n_features,
            'fit_time': sklearn_fit_time,
            'predict_time': sklearn_pred_time,
            'score': sklearn_score
        })
        
        # CxML
        if CXML_AVAILABLE:
            cxml_knn_clf = cxml.neighbors.KNeighborsClassifier(5)
            start_time = time.time()
            cxml_knn_clf.fit(X_train_clf, y_train_clf)
            cxml_fit_time = time.time() - start_time
            
            start_time = time.time()
            cxml_pred = cxml_knn_clf.predict(X_test_clf)
            cxml_pred_time = time.time() - start_time
            cxml_score = accuracy_score(y_test_clf, cxml_pred)
            
            results.append({
                'algorithm': 'KNN_Classifier',
                'library': 'CxML',
                'dataset_size': size_name,
                'n_samples': n_samples,
                'n_features': n_features,
                'fit_time': cxml_fit_time,
                'predict_time': cxml_pred_time,
                'score': cxml_score
            })
            
            # Calculate speedup
            fit_speedup = sklearn_fit_time / cxml_fit_time if cxml_fit_time > 0 else 0
            pred_speedup = sklearn_pred_time / cxml_pred_time if cxml_pred_time > 0 else 0
            
            print(f"    📊 Fit speedup: {fit_speedup:.2f}x")
            print(f"    📊 Predict speedup: {pred_speedup:.2f}x")
            print(f"    📊 Accuracy: sklearn={sklearn_score:.4f}, CxML={cxml_score:.4f}")
        
        # Benchmark KNN Regressor
        print(f"  📈 KNN Regressor...")
        
        # sklearn
        sklearn_knn_reg = KNeighborsRegressor(n_neighbors=5)
        start_time = time.time()
        sklearn_knn_reg.fit(X_train_reg, y_train_reg)
        sklearn_fit_time = time.time() - start_time
        
        start_time = time.time()
        sklearn_pred = sklearn_knn_reg.predict(X_test_reg)
        sklearn_pred_time = time.time() - start_time
        sklearn_score = -mean_squared_error(y_test_reg, sklearn_pred)
        
        results.append({
            'algorithm': 'KNN_Regressor',
            'library': 'scikit-learn',
            'dataset_size': size_name,
            'n_samples': n_samples,
            'n_features': n_features,
            'fit_time': sklearn_fit_time,
            'predict_time': sklearn_pred_time,
            'score': sklearn_score
        })
        
        # CxML
        if CXML_AVAILABLE:
            cxml_knn_reg = cxml.neighbors.KNeighborsRegressor(5)
            start_time = time.time()
            cxml_knn_reg.fit(X_train_reg, y_train_reg)
            cxml_fit_time = time.time() - start_time
            
            start_time = time.time()
            cxml_pred = cxml_knn_reg.predict(X_test_reg)
            cxml_pred_time = time.time() - start_time
            cxml_score = -mean_squared_error(y_test_reg, cxml_pred)
            
            results.append({
                'algorithm': 'KNN_Regressor',
                'library': 'CxML',
                'dataset_size': size_name,
                'n_samples': n_samples,
                'n_features': n_features,
                'fit_time': cxml_fit_time,
                'predict_time': cxml_pred_time,
                'score': cxml_score
            })
            
            # Calculate speedup
            fit_speedup = sklearn_fit_time / cxml_fit_time if cxml_fit_time > 0 else 0
            pred_speedup = sklearn_pred_time / cxml_pred_time if cxml_pred_time > 0 else 0
            
            print(f"    📊 Fit speedup: {fit_speedup:.2f}x")
            print(f"    📊 Predict speedup: {pred_speedup:.2f}x")
            print(f"    📊 MSE: sklearn={-sklearn_score:.4f}, CxML={-cxml_score:.4f}")
    
    return results

def benchmark_kmeans_performance():
    """Benchmark K-Means performance with different dataset sizes"""
    print("\n🎯 Benchmarking K-Means Performance")
    print("-" * 50)
    
    results = []
    
    # Test different dataset sizes
    sizes = [
        (1000, 10, "Small"),
        (5000, 20, "Medium"),
        (10000, 50, "Large"),
        (20000, 100, "Very Large")
    ]
    
    for n_samples, n_features, size_name in sizes:
        print(f"\n📏 Testing {size_name} dataset: {n_samples} samples, {n_features} features")
        
        # Generate clustering data
        X, y_true = make_blobs(
            n_samples=n_samples, centers=8, n_features=n_features, random_state=42
        )
        
        # sklearn
        sklearn_kmeans = KMeans(n_clusters=8, random_state=42)
        start_time = time.time()
        sklearn_kmeans.fit(X)
        sklearn_fit_time = time.time() - start_time
        
        start_time = time.time()
        sklearn_pred = sklearn_kmeans.predict(X)
        sklearn_pred_time = time.time() - start_time
        sklearn_score = adjusted_rand_score(y_true, sklearn_pred)
        
        results.append({
            'algorithm': 'KMeans',
            'library': 'scikit-learn',
            'dataset_size': size_name,
            'n_samples': n_samples,
            'n_features': n_features,
            'fit_time': sklearn_fit_time,
            'predict_time': sklearn_pred_time,
            'score': sklearn_score
        })
        
        # CxML
        if CXML_AVAILABLE:
            cxml_kmeans = cxml.cluster.KMeans(8, 300, 1e-4, "k-means++", 42)
            start_time = time.time()
            cxml_kmeans.fit(X, np.zeros(X.shape[0]))  # Dummy y for unsupervised
            cxml_fit_time = time.time() - start_time
            
            start_time = time.time()
            cxml_pred = cxml_kmeans.predict_labels(X)
            cxml_pred_time = time.time() - start_time
            cxml_score = adjusted_rand_score(y_true, cxml_pred)
            
            results.append({
                'algorithm': 'KMeans',
                'library': 'CxML',
                'dataset_size': size_name,
                'n_samples': n_samples,
                'n_features': n_features,
                'fit_time': cxml_fit_time,
                'predict_time': cxml_pred_time,
                'score': cxml_score
            })
            
            # Calculate speedup
            fit_speedup = sklearn_fit_time / cxml_fit_time if cxml_fit_time > 0 else 0
            pred_speedup = sklearn_pred_time / cxml_pred_time if cxml_pred_time > 0 else 0
            
            print(f"    📊 Fit speedup: {fit_speedup:.2f}x")
            print(f"    📊 Predict speedup: {pred_speedup:.2f}x")
            print(f"    📊 ARI: sklearn={sklearn_score:.4f}, CxML={cxml_score:.4f}")
    
    return results

def benchmark_random_forest_performance():
    """Benchmark Random Forest performance with different dataset sizes"""
    print("\n🌲 Benchmarking Random Forest Performance")
    print("-" * 50)
    
    results = []
    
    # Test different dataset sizes
    sizes = [
        (1000, 10, "Small"),
        (3000, 20, "Medium"),
        (5000, 50, "Large")
    ]
    
    for n_samples, n_features, size_name in sizes:
        print(f"\n📏 Testing {size_name} dataset: {n_samples} samples, {n_features} features")
        
        # Generate classification data
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_classes=3,
            n_informative=max(3, n_features//2), random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # sklearn
        sklearn_rf = RandomForestClassifier(n_estimators=50, random_state=42)
        start_time = time.time()
        sklearn_rf.fit(X_train, y_train)
        sklearn_fit_time = time.time() - start_time
        
        start_time = time.time()
        sklearn_pred = sklearn_rf.predict(X_test)
        sklearn_pred_time = time.time() - start_time
        sklearn_score = accuracy_score(y_test, sklearn_pred)
        
        results.append({
            'algorithm': 'RandomForest',
            'library': 'scikit-learn',
            'dataset_size': size_name,
            'n_samples': n_samples,
            'n_features': n_features,
            'fit_time': sklearn_fit_time,
            'predict_time': sklearn_pred_time,
            'score': sklearn_score
        })
        
        # CxML
        if CXML_AVAILABLE:
            cxml_rf = cxml.ensemble.RandomForestClassifier(50, -1, -1, 42)
            start_time = time.time()
            cxml_rf.fit(X_train, y_train)
            cxml_fit_time = time.time() - start_time
            
            start_time = time.time()
            cxml_pred = cxml_rf.predict(X_test)
            cxml_pred_time = time.time() - start_time
            cxml_score = accuracy_score(y_test, cxml_pred)
            
            results.append({
                'algorithm': 'RandomForest',
                'library': 'CxML',
                'dataset_size': size_name,
                'n_samples': n_samples,
                'n_features': n_features,
                'fit_time': cxml_fit_time,
                'predict_time': cxml_pred_time,
                'score': cxml_score
            })
            
            # Calculate speedup
            fit_speedup = sklearn_fit_time / cxml_fit_time if cxml_fit_time > 0 else 0
            pred_speedup = sklearn_pred_time / cxml_pred_time if cxml_pred_time > 0 else 0
            
            print(f"    📊 Fit speedup: {fit_speedup:.2f}x")
            print(f"    📊 Predict speedup: {pred_speedup:.2f}x")
            print(f"    📊 Accuracy: sklearn={sklearn_score:.4f}, CxML={cxml_score:.4f}")
    
    return results

def create_performance_visualization(results):
    """Create performance visualization"""
    if not results:
        print("❌ No results to visualize")
        return
    
    df = pd.DataFrame(results)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CxML OpenMP Parallel Processing Performance', fontsize=16)
    
    # 1. Fit time comparison by algorithm
    fit_time_pivot = df.pivot_table(
        index='algorithm', columns='library', values='fit_time', aggfunc='mean'
    )
    fit_time_pivot.plot(kind='bar', ax=axes[0,0], title='Training Time Comparison')
    axes[0,0].set_ylabel('Time (seconds)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Prediction time comparison by algorithm
    pred_time_pivot = df.pivot_table(
        index='algorithm', columns='library', values='predict_time', aggfunc='mean'
    )
    pred_time_pivot.plot(kind='bar', ax=axes[0,1], title='Prediction Time Comparison')
    axes[0,1].set_ylabel('Time (seconds)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Performance scaling with dataset size
    for algorithm in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == algorithm]
        for library in alg_data['library'].unique():
            lib_data = alg_data[alg_data['library'] == library]
            axes[1,0].plot(lib_data['n_samples'], lib_data['fit_time'], 
                          marker='o', label=f'{algorithm} ({library})')
    
    axes[1,0].set_xlabel('Dataset Size (samples)')
    axes[1,0].set_ylabel('Training Time (seconds)')
    axes[1,0].set_title('Training Time Scaling')
    axes[1,0].legend()
    axes[1,0].set_xscale('log')
    axes[1,0].set_yscale('log')
    
    # 4. Speedup analysis
    speedup_data = []
    for algorithm in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == algorithm]
        if len(alg_data) >= 2:  # Both libraries present
            sklearn_fit = alg_data[alg_data['library'] == 'scikit-learn']['fit_time'].mean()
            cxml_fit = alg_data[alg_data['library'] == 'CxML']['fit_time'].mean()
            sklearn_pred = alg_data[alg_data['library'] == 'scikit-learn']['predict_time'].mean()
            cxml_pred = alg_data[alg_data['library'] == 'CxML']['predict_time'].mean()
            
            speedup_data.append({
                'Algorithm': algorithm,
                'Fit Speedup': sklearn_fit / cxml_fit if cxml_fit > 0 else 0,
                'Predict Speedup': sklearn_pred / cxml_pred if cxml_pred > 0 else 0
            })
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data).set_index('Algorithm')
        speedup_df.plot(kind='bar', ax=axes[1,1], title='Speedup Factor (Higher = Better)')
        axes[1,1].set_ylabel('Speedup Factor')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('cxml_openmp_performance.png', dpi=300, bbox_inches='tight')
    print("📊 Performance visualization saved as 'cxml_openmp_performance.png'")
    
    # Save results
    df.to_csv('cxml_openmp_benchmark.csv', index=False)
    print("📄 Results saved as 'cxml_openmp_benchmark.csv'")
    
    # Print summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    summary = df.groupby(['algorithm', 'library']).agg({
        'fit_time': 'mean',
        'predict_time': 'mean',
        'score': 'mean'
    }).round(4)
    
    print(summary)

def main():
    """Main benchmarking function"""
    print("🚀 CxML OpenMP Parallel Processing Benchmark")
    print("=" * 60)
    
    all_results = []
    
    # Run benchmarks
    all_results.extend(benchmark_knn_performance())
    all_results.extend(benchmark_kmeans_performance())
    all_results.extend(benchmark_random_forest_performance())
    
    # Create visualizations and summary
    create_performance_visualization(all_results)
    
    print("\n🎉 OpenMP Performance Benchmarking completed!")
    print("📊 Check 'cxml_openmp_performance.png' for visualizations")
    print("📄 Check 'cxml_openmp_benchmark.csv' for detailed results")

if __name__ == "__main__":
    main()
