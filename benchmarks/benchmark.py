#!/usr/bin/env python3
"""
CxML Performance Benchmarks
Compare CxML performance with scikit-learn
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

import cxml
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.neighbors import KNeighborsRegressor as SKKNeighborsRegressor
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.cluster import KMeans as SKKMeans
from sklearn.decomposition import PCA as SKPCA

def benchmark_linear_regression():
    """Benchmark Linear Regression"""
    print("Linear Regression Benchmark")
    print("=" * 40)
    
    sizes = [100, 500, 1000, 5000, 10000]
    features = [10, 50, 100]
    
    results = {
        'sizes': sizes,
        'features': features,
        'cxml_times': [],
        'sk_times': [],
        'speedups': []
    }
    
    for n_features in features:
        print(f"\nFeatures: {n_features}")
        print("-" * 20)
        
        for n_samples in sizes:
            # Generate data
            X = np.random.randn(n_samples, n_features).astype(np.float64)
            y = np.random.randn(n_samples).astype(np.float64)
            
            # CxML
            start = time.time()
            lr_cxml = cxml.linear_model.LinearRegression()
            lr_cxml.fit(X, y)
            y_pred_cxml = lr_cxml.predict(X)
            cxml_time = time.time() - start
            
            # scikit-learn
            start = time.time()
            lr_sk = SKLinearRegression()
            lr_sk.fit(X, y)
            y_pred_sk = lr_sk.predict(X)
            sk_time = time.time() - start
            
            speedup = sk_time / cxml_time if cxml_time > 0 else 0
            
            print(f"  Samples: {n_samples:5d} | CxML: {cxml_time:.4f}s | scikit-learn: {sk_time:.4f}s | Speedup: {speedup:.2f}x")
            
            results['cxml_times'].append(cxml_time)
            results['sk_times'].append(sk_time)
            results['speedups'].append(speedup)
    
    return results

def benchmark_knn():
    """Benchmark K-Nearest Neighbors"""
    print("\nK-Nearest Neighbors Benchmark")
    print("=" * 40)
    
    sizes = [100, 500, 1000, 2000]
    features = [5, 10, 20]
    
    results = {
        'sizes': sizes,
        'features': features,
        'cxml_times': [],
        'sk_times': [],
        'speedups': []
    }
    
    for n_features in features:
        print(f"\nFeatures: {n_features}")
        print("-" * 20)
        
        for n_samples in sizes:
            # Generate data
            X = np.random.randn(n_samples, n_features).astype(np.float64)
            y = np.random.randn(n_samples).astype(np.float64)
            
            # CxML
            start = time.time()
            knn_cxml = cxml.neighbors.KNeighborsRegressor(n_neighbors=5)
            knn_cxml.fit(X, y)
            y_pred_cxml = knn_cxml.predict(X)
            cxml_time = time.time() - start
            
            # scikit-learn
            start = time.time()
            knn_sk = SKKNeighborsRegressor(n_neighbors=5)
            knn_sk.fit(X, y)
            y_pred_sk = knn_sk.predict(X)
            sk_time = time.time() - start
            
            speedup = sk_time / cxml_time if cxml_time > 0 else 0
            
            print(f"  Samples: {n_samples:5d} | CxML: {cxml_time:.4f}s | scikit-learn: {sk_time:.4f}s | Speedup: {speedup:.2f}x")
            
            results['cxml_times'].append(cxml_time)
            results['sk_times'].append(sk_time)
            results['speedups'].append(speedup)
    
    return results

def benchmark_preprocessing():
    """Benchmark Preprocessing"""
    print("\nPreprocessing Benchmark")
    print("=" * 40)
    
    sizes = [1000, 5000, 10000, 50000]
    features = [10, 50, 100]
    
    results = {
        'sizes': sizes,
        'features': features,
        'cxml_times': [],
        'sk_times': [],
        'speedups': []
    }
    
    for n_features in features:
        print(f"\nFeatures: {n_features}")
        print("-" * 20)
        
        for n_samples in sizes:
            # Generate data
            X = np.random.randn(n_samples, n_features).astype(np.float64)
            
            # CxML
            start = time.time()
            scaler_cxml = cxml.preprocessing.StandardScaler()
            scaler_cxml.fit(X, np.zeros(n_samples))
            X_scaled_cxml = scaler_cxml.transform(X)
            cxml_time = time.time() - start
            
            # scikit-learn
            start = time.time()
            scaler_sk = SKStandardScaler()
            scaler_sk.fit(X)
            X_scaled_sk = scaler_sk.transform(X)
            sk_time = time.time() - start
            
            speedup = sk_time / cxml_time if cxml_time > 0 else 0
            
            print(f"  Samples: {n_samples:5d} | CxML: {cxml_time:.4f}s | scikit-learn: {sk_time:.4f}s | Speedup: {speedup:.2f}x")
            
            results['cxml_times'].append(cxml_time)
            results['sk_times'].append(sk_time)
            results['speedups'].append(speedup)
    
    return results

def benchmark_clustering():
    """Benchmark K-Means Clustering"""
    print("\nK-Means Clustering Benchmark")
    print("=" * 40)
    
    sizes = [1000, 5000, 10000, 20000]
    features = [5, 10, 20]
    n_clusters = 5
    
    results = {
        'sizes': sizes,
        'features': features,
        'cxml_times': [],
        'sk_times': [],
        'speedups': []
    }
    
    for n_features in features:
        print(f"\nFeatures: {n_features}")
        print("-" * 20)
        
        for n_samples in sizes:
            # Generate data
            X = np.random.randn(n_samples, n_features).astype(np.float64)
            
            # CxML
            start = time.time()
            kmeans_cxml = cxml.cluster.KMeans(n_clusters=n_clusters)
            kmeans_cxml.fit(X, np.zeros(n_samples))
            labels_cxml = kmeans_cxml.predict_labels(X)
            cxml_time = time.time() - start
            
            # scikit-learn
            start = time.time()
            kmeans_sk = SKKMeans(n_clusters=n_clusters, random_state=42)
            kmeans_sk.fit(X)
            labels_sk = kmeans_sk.predict(X)
            sk_time = time.time() - start
            
            speedup = sk_time / cxml_time if cxml_time > 0 else 0
            
            print(f"  Samples: {n_samples:5d} | CxML: {cxml_time:.4f}s | scikit-learn: {sk_time:.4f}s | Speedup: {speedup:.2f}x")
            
            results['cxml_times'].append(cxml_time)
            results['sk_times'].append(sk_time)
            results['speedups'].append(speedup)
    
    return results

def benchmark_pca():
    """Benchmark PCA"""
    print("\nPCA Benchmark")
    print("=" * 40)
    
    sizes = [1000, 5000, 10000]
    features = [20, 50, 100]
    n_components = 10
    
    results = {
        'sizes': sizes,
        'features': features,
        'cxml_times': [],
        'sk_times': [],
        'speedups': []
    }
    
    for n_features in features:
        print(f"\nFeatures: {n_features}")
        print("-" * 20)
        
        for n_samples in sizes:
            # Generate data
            X = np.random.randn(n_samples, n_features).astype(np.float64)
            
            # CxML
            start = time.time()
            pca_cxml = cxml.decomposition.PCA(n_components=n_components)
            pca_cxml.fit(X, np.zeros(n_samples))
            X_transformed_cxml = pca_cxml.transform(X)
            cxml_time = time.time() - start
            
            # scikit-learn
            start = time.time()
            pca_sk = SKPCA(n_components=n_components)
            pca_sk.fit(X)
            X_transformed_sk = pca_sk.transform(X)
            sk_time = time.time() - start
            
            speedup = sk_time / cxml_time if cxml_time > 0 else 0
            
            print(f"  Samples: {n_samples:5d} | CxML: {cxml_time:.4f}s | scikit-learn: {sk_time:.4f}s | Speedup: {speedup:.2f}x")
            
            results['cxml_times'].append(cxml_time)
            results['sk_times'].append(sk_time)
            results['speedups'].append(speedup)
    
    return results

def plot_results(results_dict, title):
    """Plot benchmark results"""
    try:
        plt.figure(figsize=(12, 8))
        
        for i, (algorithm, results) in enumerate(results_dict.items()):
            plt.subplot(2, 2, i + 1)
            
            sizes = results['sizes']
            speedups = results['speedups']
            
            plt.plot(sizes, speedups, 'o-', label=algorithm)
            plt.xlabel('Number of Samples')
            plt.ylabel('Speedup (scikit-learn / CxML)')
            plt.title(f'{algorithm} Speedup')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        print(f"\nBenchmark results saved to benchmark_results.png")
        
    except ImportError:
        print("Matplotlib not available, skipping plot generation")

def main():
    """Run all benchmarks"""
    print("CxML Performance Benchmarks")
    print("=" * 50)
    print("Comparing CxML with scikit-learn")
    print("Higher speedup values indicate CxML is faster")
    print()
    
    # Run benchmarks
    lr_results = benchmark_linear_regression()
    knn_results = benchmark_knn()
    preproc_results = benchmark_preprocessing()
    cluster_results = benchmark_clustering()
    pca_results = benchmark_pca()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    algorithms = {
        'Linear Regression': lr_results,
        'K-Nearest Neighbors': knn_results,
        'Preprocessing': preproc_results,
        'K-Means Clustering': cluster_results,
        'PCA': pca_results
    }
    
    for algorithm, results in algorithms.items():
        avg_speedup = np.mean(results['speedups'])
        max_speedup = np.max(results['speedups'])
        print(f"{algorithm:20s}: Avg Speedup: {avg_speedup:.2f}x, Max Speedup: {max_speedup:.2f}x")
    
    # Plot results
    plot_results(algorithms, "CxML vs scikit-learn Performance")

if __name__ == "__main__":
    main()
