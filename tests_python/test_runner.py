#!/usr/bin/env python3
"""
Comprehensive Test Runner for AuroraML Python Tests
Runs all test modules and provides detailed reporting
"""

import sys
import os
import unittest
import time
import random

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

def run_all_tests():
    """Run all AuroraML Python tests"""
    
    print("🚀 Starting AuroraML Comprehensive Python Test Suite")
    print("=" * 80)
    
    # List of all test modules
    test_modules = [
        'test_basic',
        'test_comprehensive', 
        'test_linear_models',
        'test_neighbors',
        'test_tree',
        'test_metrics',
        'test_preprocessing',
        'test_model_selection',
        'test_naive_bayes',
        'test_clustering',
        'test_decomposition',
        'test_svm',
        'test_ensemble',
        'test_random'
    ]
    
    print(f"Running {len(test_modules)} test modules...")
    print("=" * 80)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    results = {}
    
    for i, module_name in enumerate(test_modules, 1):
        print(f"\n[{i}/{len(test_modules)}] Running {module_name}...")
        print("-" * 60)
        
        try:
            # Import and run the test module
            spec = __import__(module_name)
            
            # Create a test suite from the module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(spec)
            
            # Run the tests
            runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
            result = runner.run(suite)
            
            # Collect results
            tests_run = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)
            
            total_tests += tests_run
            total_failures += failures
            total_errors += errors
            
            results[module_name] = {
                'tests_run': tests_run,
                'failures': failures,
                'errors': errors,
                'success': failures == 0 and errors == 0
            }
            
            status = "✅ PASSED" if failures == 0 and errors == 0 else "❌ FAILED"
            print(f"{status} - {tests_run} tests, {failures} failures, {errors} errors")
            
        except Exception as e:
            print(f"❌ ERROR running {module_name}: {e}")
            results[module_name] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'success': False
            }
            total_errors += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    successful_modules = sum(1 for r in results.values() if r['success'])
    
    print(f"Modules tested: {len(test_modules)}")
    print(f"Successful modules: {successful_modules}")
    print(f"Failed modules: {len(test_modules) - successful_modules}")
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    
    success_rate = (total_tests - total_failures - total_errors) / total_tests * 100 if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 60)
    for module_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {module_name}: {result['tests_run']} tests, {result['failures']} failures, {result['errors']} errors")
    
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n💥 {total_failures + total_errors} test(s) failed!")
        return False

def run_tests_with_shuffling():
    """Run tests with shuffling to verify robustness"""
    
    print("🚀 Starting AuroraML Python Test Suite with Shuffling")
    print("=" * 80)
    
    # List of all test modules
    test_modules = [
        'test_basic',
        'test_comprehensive', 
        'test_linear_models',
        'test_neighbors',
        'test_tree',
        'test_metrics',
        'test_preprocessing',
        'test_model_selection',
        'test_naive_bayes',
        'test_clustering',
        'test_decomposition',
        'test_svm',
        'test_ensemble',
        'test_random'
    ]
    
    # Shuffle the test modules
    random.seed(int(time.time()))
    shuffled_modules = test_modules.copy()
    random.shuffle(shuffled_modules)
    
    print(f"Running {len(test_modules)} test modules in random order...")
    print(f"Test execution order: {shuffled_modules}")
    print("=" * 80)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    results = {}
    
    for i, module_name in enumerate(shuffled_modules, 1):
        print(f"\n[{i}/{len(shuffled_modules)}] Running {module_name}...")
        print("-" * 60)
        
        try:
            # Import and run the test module
            spec = __import__(module_name)
            
            # Create a test suite from the module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(spec)
            
            # Run the tests
            runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
            result = runner.run(suite)
            
            # Collect results
            tests_run = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)
            
            total_tests += tests_run
            total_failures += failures
            total_errors += errors
            
            results[module_name] = {
                'tests_run': tests_run,
                'failures': failures,
                'errors': errors,
                'success': failures == 0 and errors == 0
            }
            
            status = "✅ PASSED" if failures == 0 and errors == 0 else "❌ FAILED"
            print(f"{status} - {tests_run} tests, {failures} failures, {errors} errors")
            
        except Exception as e:
            print(f"❌ ERROR running {module_name}: {e}")
            results[module_name] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'success': False
            }
            total_errors += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SHUFFLED TEST SUMMARY")
    print("=" * 80)
    
    successful_modules = sum(1 for r in results.values() if r['success'])
    
    print(f"Modules tested: {len(test_modules)}")
    print(f"Successful modules: {successful_modules}")
    print(f"Failed modules: {len(test_modules) - successful_modules}")
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    
    success_rate = (total_tests - total_failures - total_errors) / total_tests * 100 if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 60)
    for module_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {module_name}: {result['tests_run']} tests, {result['failures']} failures, {result['errors']} errors")
    
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n💥 {total_failures + total_errors} test(s) failed!")
        return False

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--shuffle':
        success = run_tests_with_shuffling()
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
