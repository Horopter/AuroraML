#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Script for CxML
Compares CxML performance with scikit-learn across multiple algorithms
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
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

class PerformanceBenchmark:
    def __init__(self):
        self.results = []
        self.datasets = {}
        
    def generate_datasets(self):
        """Generate various datasets for benchmarking"""
        print("📊 Generating benchmark datasets...")
        
        # Classification datasets
        X_clf_small, y_clf_small = make_classification(
            n_samples=1000, n_features=10, n_classes=3, random_state=42
        )
        X_clf_medium, y_clf_medium = make_classification(
            n_samples=5000, n_features=20, n_classes=3, random_state=42
        )
        X_clf_large, y_clf_large = make_classification(
            n_samples=10000, n_features=50, n_classes=5, random_state=42
        )
        
        # Regression datasets
        X_reg_small, y_reg_small = make_regression(
            n_samples=1000, n_features=10, noise=0.1, random_state=42
        )
        X_reg_medium, y_reg_medium = make_regression(
            n_samples=5000, n_features=20, noise=0.1, random_state=42
        )
        X_reg_large, y_reg_large = make_regression(
            n_samples=10000, n_features=50, noise=0.1, random_state=42
        )
        
        # Clustering datasets
        X_cluster_small, y_cluster_small = make_blobs(
            n_samples=1000, centers=5, n_features=10, random_state=42
        )
        X_cluster_medium, y_cluster_medium = make_blobs(
            n_samples=5000, centers=8, n_features=20, random_state=42
        )
        X_cluster_large, y_cluster_large = make_blobs(
            n_samples=10000, centers=10, n_features=50, random_state=42
        )
        
        self.datasets = {
            'classification': {
                'small': (X_clf_small, y_clf_small),
                'medium': (X_clf_medium, y_clf_medium),
                'large': (X_clf_large, y_clf_large)
            },
            'regression': {
                'small': (X_reg_small, y_reg_small),
                'medium': (X_reg_medium, y_reg_medium),
                'large': (X_reg_large, y_reg_large)
            },
            'clustering': {
                'small': (X_cluster_small, y_cluster_small),
                'medium': (X_cluster_medium, y_cluster_medium),
                'large': (X_cluster_large, y_cluster_large)
            }
        }
        print("✅ Datasets generated successfully")
    
    def benchmark_algorithm(self, algorithm_name, sklearn_model, cxml_model, 
                          X_train, X_test, y_train, y_test, task_type):
        """Benchmark a single algorithm"""
        results = []
        
        # Benchmark sklearn
        try:
            start_time = time.time()
            sklearn_model.fit(X_train, y_train)
            sklearn_fit_time = time.time() - start_time
            
            start_time = time.time()
            sklearn_pred = sklearn_model.predict(X_test)
            sklearn_pred_time = time.time() - start_time
            
            # Calculate metrics
            if task_type == 'classification':
                sklearn_score = accuracy_score(y_test, sklearn_pred)
            elif task_type == 'regression':
                sklearn_score = -mean_squared_error(y_test, sklearn_pred)  # Negative MSE for consistency
            else:  # clustering
                sklearn_score = adjusted_rand_score(y_test, sklearn_pred)
            
            results.append({
                'algorithm': algorithm_name,
                'library': 'scikit-learn',
                'fit_time': sklearn_fit_time,
                'predict_time': sklearn_pred_time,
                'score': sklearn_score,
                'task': task_type
            })
            
        except Exception as e:
            print(f"❌ sklearn {algorithm_name} failed: {e}")
        
        # Benchmark CxML
        if CXML_AVAILABLE:
            try:
                start_time = time.time()
                cxml_model.fit(X_train, y_train)
                cxml_fit_time = time.time() - start_time
                
                start_time = time.time()
                cxml_pred = cxml_model.predict(X_test)
                cxml_pred_time = time.time() - start_time
                
                # Calculate metrics
                if task_type == 'classification':
                    cxml_score = accuracy_score(y_test, cxml_pred)
                elif task_type == 'regression':
                    cxml_score = -mean_squared_error(y_test, cxml_pred)
                else:  # clustering
                    cxml_score = adjusted_rand_score(y_test, cxml_pred)
                
                results.append({
                    'algorithm': algorithm_name,
                    'library': 'CxML',
                    'fit_time': cxml_fit_time,
                    'predict_time': cxml_pred_time,
                    'score': cxml_score,
                    'task': task_type
                })
                
            except Exception as e:
                print(f"❌ CxML {algorithm_name} failed: {e}")
        
        return results
    
    def run_classification_benchmarks(self):
        """Run classification algorithm benchmarks"""
        print("\n🔍 Running Classification Benchmarks...")
        
        algorithms = [
            ('KNN', KNeighborsClassifier(n_neighbors=5), 
             cxml.neighbors.KNeighborsClassifier(5) if CXML_AVAILABLE else None),
            ('Random Forest', RandomForestClassifier(n_estimators=50, random_state=42),
             cxml.ensemble.RandomForestClassifier(50, -1, -1, 42) if CXML_AVAILABLE else None),
            ('Naive Bayes', GaussianNB(),
             cxml.naive_bayes.GaussianNB() if CXML_AVAILABLE else None),
            ('Linear SVM', LinearSVC(random_state=42),
             cxml.svm.LinearSVC(1.0, 1000, 0.01, 42) if CXML_AVAILABLE else None),
            ('Decision Tree', DecisionTreeClassifier(random_state=42),
             cxml.tree.DecisionTreeClassifier("gini", -1, 2, 1, 0.0, 42) if CXML_AVAILABLE else None)
        ]
        
        for size_name, (X, y) in self.datasets['classification'].items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"  📏 Dataset size: {size_name} ({X.shape[0]} samples, {X.shape[1]} features)")
            
            for alg_name, sklearn_model, cxml_model in algorithms:
                if cxml_model is not None:
                    results = self.benchmark_algorithm(
                        f"{alg_name}_{size_name}", sklearn_model, cxml_model,
                        X_train, X_test, y_train, y_test, 'classification'
                    )
                    self.results.extend(results)
    
    def run_regression_benchmarks(self):
        """Run regression algorithm benchmarks"""
        print("\n📈 Running Regression Benchmarks...")
        
        algorithms = [
            ('Linear Regression', LinearRegression(),
             cxml.linear_model.LinearRegression() if CXML_AVAILABLE else None),
            ('Ridge', Ridge(random_state=42),
             cxml.linear_model.Ridge(1.0, 42) if CXML_AVAILABLE else None),
            ('KNN', KNeighborsRegressor(n_neighbors=5),
             cxml.neighbors.KNeighborsRegressor(5) if CXML_AVAILABLE else None),
            ('Random Forest', RandomForestRegressor(n_estimators=50, random_state=42),
             cxml.ensemble.RandomForestRegressor(50, -1, -1, 42) if CXML_AVAILABLE else None),
            ('Decision Tree', DecisionTreeRegressor(random_state=42),
             cxml.tree.DecisionTreeRegressor("mse", -1, 2, 1, 0.0, 42) if CXML_AVAILABLE else None)
        ]
        
        for size_name, (X, y) in self.datasets['regression'].items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"  📏 Dataset size: {size_name} ({X.shape[0]} samples, {X.shape[1]} features)")
            
            for alg_name, sklearn_model, cxml_model in algorithms:
                if cxml_model is not None:
                    results = self.benchmark_algorithm(
                        f"{alg_name}_{size_name}", sklearn_model, cxml_model,
                        X_train, X_test, y_train, y_test, 'regression'
                    )
                    self.results.extend(results)
    
    def run_clustering_benchmarks(self):
        """Run clustering algorithm benchmarks"""
        print("\n🎯 Running Clustering Benchmarks...")
        
        algorithms = [
            ('K-Means', KMeans(n_clusters=5, random_state=42),
             cxml.cluster.KMeans(5, 300, 1e-4, "k-means++", 42) if CXML_AVAILABLE else None)
        ]
        
        for size_name, (X, y_true) in self.datasets['clustering'].items():
            print(f"  📏 Dataset size: {size_name} ({X.shape[0]} samples, {X.shape[1]} features)")
            
            for alg_name, sklearn_model, cxml_model in algorithms:
                if cxml_model is not None:
                    results = self.benchmark_algorithm(
                        f"{alg_name}_{size_name}", sklearn_model, cxml_model,
                        X, X, y_true, y_true, 'clustering'  # For clustering, we use the same data
                    )
                    self.results.extend(results)
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("🚀 Starting Comprehensive Performance Benchmarking")
        print("=" * 60)
        
        self.generate_datasets()
        
        if CXML_AVAILABLE:
            self.run_classification_benchmarks()
            self.run_regression_benchmarks()
            self.run_clustering_benchmarks()
        else:
            print("❌ CxML not available, skipping benchmarks")
        
        print("\n✅ All benchmarks completed!")
    
    def analyze_results(self):
        """Analyze and visualize benchmark results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n📊 BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        # Performance comparison
        comparison = df.groupby(['algorithm', 'library']).agg({
            'fit_time': 'mean',
            'predict_time': 'mean',
            'score': 'mean'
        }).round(4)
        
        print("\n🏆 Performance Comparison (Average across all datasets):")
        print(comparison)
        
        # Speedup analysis
        print("\n⚡ Speedup Analysis (CxML vs scikit-learn):")
        speedup_df = df.pivot_table(
            index='algorithm', 
            columns='library', 
            values=['fit_time', 'predict_time'], 
            aggfunc='mean'
        )
        
        if 'CxML' in speedup_df.columns.get_level_values(1):
            fit_speedup = speedup_df[('fit_time', 'scikit-learn')] / speedup_df[('fit_time', 'CxML')]
            pred_speedup = speedup_df[('predict_time', 'scikit-learn')] / speedup_df[('predict_time', 'CxML')]
            
            speedup_summary = pd.DataFrame({
                'Fit Speedup': fit_speedup.round(2),
                'Predict Speedup': pred_speedup.round(2)
            })
            print(speedup_summary)
        
        # Create visualizations
        self.create_visualizations(df)
    
    def create_visualizations(self, df):
        """Create performance visualization plots"""
        print("\n📈 Creating performance visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CxML vs scikit-learn Performance Comparison', fontsize=16)
        
        # 1. Fit time comparison
        fit_time_pivot = df.pivot_table(
            index='algorithm', columns='library', values='fit_time', aggfunc='mean'
        )
        fit_time_pivot.plot(kind='bar', ax=axes[0,0], title='Training Time Comparison')
        axes[0,0].set_ylabel('Time (seconds)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Prediction time comparison
        pred_time_pivot = df.pivot_table(
            index='algorithm', columns='library', values='predict_time', aggfunc='mean'
        )
        pred_time_pivot.plot(kind='bar', ax=axes[0,1], title='Prediction Time Comparison')
        axes[0,1].set_ylabel('Time (seconds)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Score comparison
        score_pivot = df.pivot_table(
            index='algorithm', columns='library', values='score', aggfunc='mean'
        )
        score_pivot.plot(kind='bar', ax=axes[1,0], title='Model Performance Comparison')
        axes[1,0].set_ylabel('Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Speedup heatmap
        if 'CxML' in df['library'].values:
            speedup_data = []
            for alg in df['algorithm'].unique():
                alg_data = df[df['algorithm'] == alg]
                if len(alg_data) >= 2:  # Both libraries present
                    sklearn_fit = alg_data[alg_data['library'] == 'scikit-learn']['fit_time'].mean()
                    cxml_fit = alg_data[alg_data['library'] == 'CxML']['fit_time'].mean()
                    sklearn_pred = alg_data[alg_data['library'] == 'scikit-learn']['predict_time'].mean()
                    cxml_pred = alg_data[alg_data['library'] == 'CxML']['predict_time'].mean()
                    
                    speedup_data.append({
                        'Algorithm': alg,
                        'Fit Speedup': sklearn_fit / cxml_fit if cxml_fit > 0 else 0,
                        'Predict Speedup': sklearn_pred / cxml_pred if cxml_pred > 0 else 0
                    })
            
            if speedup_data:
                speedup_df = pd.DataFrame(speedup_data).set_index('Algorithm')
                sns.heatmap(speedup_df, annot=True, fmt='.2f', ax=axes[1,1], 
                           cmap='RdYlGn', center=1.0, title='Speedup Factor (Higher = Better)')
        
        plt.tight_layout()
        plt.savefig('cxml_benchmark_results.png', dpi=300, bbox_inches='tight')
        print("📊 Visualization saved as 'cxml_benchmark_results.png'")
        
        # Save detailed results to CSV
        df.to_csv('cxml_benchmark_detailed.csv', index=False)
        print("📄 Detailed results saved as 'cxml_benchmark_detailed.csv'")

def main():
    """Main benchmarking function"""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
    benchmark.analyze_results()
    
    print("\n🎉 Benchmarking completed successfully!")
    print("📊 Check 'cxml_benchmark_results.png' for visualizations")
    print("📄 Check 'cxml_benchmark_detailed.csv' for detailed results")

if __name__ == "__main__":
    main()
