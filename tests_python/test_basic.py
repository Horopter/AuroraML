#!/usr/bin/env python3
"""
Basic Test Suite for AuroraML
Tests basic import and instantiation of AuroraML modules and classes
"""

import sys
import os
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

def test_auroraml_import():
    """Test that AuroraML can be imported successfully"""
    try:
        import auroraml
        print("✅ AuroraML successfully imported")
        return True
    except Exception as e:
        print(f"❌ AuroraML import failed: {e}")
        return False

def test_auroraml_modules():
    """Test that AuroraML modules can be imported"""
    modules_to_test = [
        'auroraml.linear_model',
        'auroraml.neighbors', 
        'auroraml.tree',
        'auroraml.metrics',
        'auroraml.preprocessing',
        'auroraml.model_selection',
        'auroraml.naive_bayes',
        'auroraml.cluster',
        'auroraml.decomposition',
        'auroraml.svm',
        'auroraml.ensemble',
        'auroraml.gradient_boosting',
        'auroraml.random'
    ]
    
    success_count = 0
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name} module imported successfully")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name} import failed: {e}")
    
    print(f"\n📊 Module Import Summary: {success_count}/{len(modules_to_test)} modules imported successfully")
    return success_count == len(modules_to_test)

def test_auroraml_classes():
    """Test that AuroraML classes can be instantiated"""
    try:
        import auroraml.linear_model as aml_lm
        import auroraml.neighbors as aml_neighbors
        import auroraml.tree as aml_tree
        import auroraml.preprocessing as aml_pp
        import auroraml.cluster as aml_cluster
        import auroraml.decomposition as aml_decomp
        import auroraml.naive_bayes as aml_nb
        import auroraml.svm as aml_svm
        import auroraml.ensemble as aml_ensemble
        import auroraml.random as aml_random
        
        classes_to_test = [
            ('LinearRegression', aml_lm.LinearRegression),
            ('Ridge', aml_lm.Ridge),
            ('Lasso', aml_lm.Lasso),
            ('KNeighborsClassifier', aml_neighbors.KNeighborsClassifier),
            ('KNeighborsRegressor', aml_neighbors.KNeighborsRegressor),
            ('DecisionTreeClassifier', aml_tree.DecisionTreeClassifier),
            ('DecisionTreeRegressor', aml_tree.DecisionTreeRegressor),
            ('StandardScaler', aml_pp.StandardScaler),
            ('MinMaxScaler', aml_pp.MinMaxScaler),
            ('KMeans', aml_cluster.KMeans),
            ('PCA', aml_decomp.PCA),
            ('GaussianNB', aml_nb.GaussianNB),
            ('LinearSVC', aml_svm.LinearSVC),
            ('RandomForestClassifier', aml_ensemble.RandomForestClassifier),
            ('RandomForestRegressor', aml_ensemble.RandomForestRegressor),
            ('PCG64', aml_random.PCG64)
        ]
        
        success_count = 0
        for class_name, class_obj in classes_to_test:
            try:
                instance = class_obj()
                print(f"✅ {class_name} instantiated successfully")
                success_count += 1
            except Exception as e:
                print(f"❌ {class_name} instantiation failed: {e}")
        
        print(f"\n📊 Class Instantiation Summary: {success_count}/{len(classes_to_test)} classes instantiated successfully")
        return success_count == len(classes_to_test)
        
    except Exception as e:
        print(f"❌ Class testing failed: {e}")
        return False

def test_auroraml_functions():
    """Test that AuroraML functions are callable"""
    try:
        import auroraml.metrics as aml_metrics
        import auroraml.model_selection as aml_ms
        
        functions_to_test = [
            ('accuracy_score', aml_metrics.accuracy_score),
            ('mean_squared_error', aml_metrics.mean_squared_error),
            ('train_test_split', aml_ms.train_test_split)
        ]
        
        success_count = 0
        for func_name, func_obj in functions_to_test:
            try:
                if callable(func_obj):
                    print(f"✅ {func_name} function is callable")
                    success_count += 1
                else:
                    print(f"❌ {func_name} is not callable")
            except Exception as e:
                print(f"❌ {func_name} testing failed: {e}")
        
        print(f"\n📊 Function Test Summary: {success_count}/{len(functions_to_test)} functions are callable")
        return success_count == len(functions_to_test)
        
    except Exception as e:
        print(f"❌ Function testing failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🚀 Starting AuroraML Basic Test Suite")
    print("=" * 60)
    
    tests = [
        ("AuroraML Import", test_auroraml_import),
        ("AuroraML Modules", test_auroraml_modules),
        ("AuroraML Classes", test_auroraml_classes),
        ("AuroraML Functions", test_auroraml_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📦 Testing {test_name}...")
        if test_func():
            print(f"✅ {test_name} passed")
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return True
    else:
        print(f"💥 {total - passed} test(s) failed!")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
