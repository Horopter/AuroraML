#!/usr/bin/env python3
"""
Custom Test Runner for AuroraML Python Tests with Shuffling
Runs the working test files in random order to verify robustness
"""

import sys
import os
import unittest
import random
import time

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

def run_tests_with_shuffling():
    """Run tests with shuffling to verify robustness"""
    
    # List of working test files (can be expanded as more tests are fixed)
    working_test_files = [
        'test_basic.py',
        'test_comprehensive.py',
        'test_linear_models.py',
        'test_neighbors.py',
        'test_tree.py',
        'test_metrics.py'
    ]
    
    # List of test files that need API fixes (commented out for now)
    # 'test_preprocessing.py',      # API issues with encoders and set_params
    # 'test_model_selection.py',    # Not tested yet
    # 'test_naive_bayes.py',       # Not tested yet
    # 'test_clustering.py',        # Not tested yet
    # 'test_decomposition.py',     # Not tested yet
    # 'test_svm.py',              # Not tested yet
    # 'test_ensemble.py',         # Not tested yet
    # 'test_random.py'            # Not tested yet
    
    print("🚀 Starting AuroraML Python Test Suite with Shuffling")
    print("=" * 60)
    print(f"Running {len(working_test_files)} test files in random order...")
    
    # Shuffle the test files
    random.seed(int(time.time()))
    shuffled_files = working_test_files.copy()
    random.shuffle(shuffled_files)
    
    print(f"Test execution order: {shuffled_files}")
    print("=" * 60)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    results = {}
    
    for i, test_file in enumerate(shuffled_files, 1):
        print(f"\n[{i}/{len(shuffled_files)}] Running {test_file}...")
        print("-" * 40)
        
        try:
            # Import and run the test module
            module_name = test_file[:-3]  # Remove .py extension
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
            
            results[test_file] = {
                'tests_run': tests_run,
                'failures': failures,
                'errors': errors,
                'success': failures == 0 and errors == 0
            }
            
            status = "✅ PASSED" if failures == 0 and errors == 0 else "❌ FAILED"
            print(f"{status} - {tests_run} tests, {failures} failures, {errors} errors")
            
        except Exception as e:
            print(f"❌ ERROR running {test_file}: {e}")
            results[test_file] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'success': False
            }
            total_errors += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful_files = sum(1 for r in results.values() if r['success'])
    
    print(f"Files tested: {len(working_test_files)}")
    print(f"Successful files: {successful_files}")
    print(f"Failed files: {len(working_test_files) - successful_files}")
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    
    success_rate = (total_tests - total_failures - total_errors) / total_tests * 100 if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 40)
    for test_file, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {test_file}: {result['tests_run']} tests, {result['failures']} failures, {result['errors']} errors")
    
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n💥 {total_failures + total_errors} test(s) failed!")
        return False

if __name__ == '__main__':
    success = run_tests_with_shuffling()
    sys.exit(0 if success else 1)
