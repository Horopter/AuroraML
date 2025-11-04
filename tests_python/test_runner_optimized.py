#!/usr/bin/env python3
"""
Optimized Test Runner for AuroraML Python Tests
Runs fast tests first, then slow tests at the end for better progress visibility
"""

import sys
import os
import unittest
import time

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

# Fast tests - typically complete in < 1 second
FAST_TESTS = [
    'test_basic',
    'test_utils',
    'test_feature_selection',
    'test_preprocessing_extended',
    'test_compose',
    'test_pipeline',
    'test_impute',
    'test_calibration',
    'test_discriminant_analysis',
    'test_naive_bayes_variants',
    'test_extratree',
    'test_outlier_detection',
    'test_cluster_extended',
    'test_metrics',
    'test_preprocessing',
    'test_tree',
    'test_neighbors',
    'test_linear_models',
    'test_svm',
    'test_decomposition',
    'test_clustering',
    'test_naive_bayes',
    'test_model_selection',
    'test_random',
    'test_inspection',
    'test_ensemble_wrappers',
]

# Slow tests - algorithms with iterations, boosting, EM, etc.
SLOW_TESTS = [
    'test_isotonic',        # Known bottleneck - PAVA algorithm
    'test_mixture',         # GaussianMixture with EM (max_iter=50)
    'test_semi_supervised', # LabelPropagation/Spreading (iterative)
    'test_adaboost',        # n_estimators=50-100
    'test_xgboost',         # n_estimators=50-100
    'test_catboost',        # n_estimators=50-100
    'test_ensemble',        # May contain slow ensemble methods
    'test_comprehensive',   # May contain various slow tests
]

def run_test_module(module_name, test_file):
    """Run a single test module and return results with 5-minute timeout"""
    start_time = time.time()
    
    try:
        # Use subprocess to isolate crashes
        import subprocess
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute (300 second) timeout per test file
            cwd=os.path.dirname(test_file)
        )
        
        elapsed = time.time() - start_time
        
        # Parse output to count tests
        output = result.stdout + result.stderr
        tests_run = output.count('test_')
        if 'Ran' in output:
            # Try to extract test count from unittest output
            import re
            match = re.search(r'Ran (\d+) test', output)
            if match:
                tests_run = int(match.group(1))
        
        # Check if it passed
        success = result.returncode == 0 and ('OK' in output or 'FAILED' not in output)
        has_failures = 'FAILED' in output or 'ERROR' in output
        
        return {
            'tests_run': tests_run if tests_run > 0 else 1,
            'failures': 1 if has_failures and not success else 0,
            'errors': 1 if result.returncode != 0 and not success else 0,
            'success': success,
            'elapsed': elapsed,
            'output': output[:500]  # First 500 chars
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'success': False,
            'elapsed': elapsed,
            'timeout': True,
            'error_msg': f'Test exceeded 5-minute ({300}s) timeout. Should be moved to {module_name}_large_time.py'
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'success': False,
            'elapsed': elapsed,
            'error_msg': str(e)
        }

def run_optimized_tests():
    """Run tests with fast tests first, slow tests last"""
    
    print("🚀 AuroraML Optimized Test Runner")
    print("=" * 80)
    print("⚡ Fast tests first → 🐢 Slow tests last")
    print("=" * 80)
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Collect all available test files
    fast_test_files = []
    slow_test_files = []
    
    for test_name in FAST_TESTS:
        test_file = os.path.join(test_dir, f"{test_name}.py")
        # Exclude _large_time files from regular test runs
        if os.path.exists(test_file) and not test_name.endswith('_large_time'):
            fast_test_files.append((test_name, test_file))
    
    for test_name in SLOW_TESTS:
        test_file = os.path.join(test_dir, f"{test_name}.py")
        # Exclude _large_time files from regular test runs
        if os.path.exists(test_file) and not test_name.endswith('_large_time'):
            slow_test_files.append((test_name, test_file))
    
    total_files = len(fast_test_files) + len(slow_test_files)
    print(f"\n📋 Found {len(fast_test_files)} fast tests, {len(slow_test_files)} slow tests")
    print("=" * 80)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    results = {}
    overall_start = time.time()
    
    # Run fast tests first
    print(f"\n⚡ PHASE 1: Fast Tests ({len(fast_test_files)} files)")
    print("-" * 80)
    
    for i, (test_name, test_file) in enumerate(fast_test_files, 1):
        print(f"[{i}/{len(fast_test_files)}] {test_name}... ", end='', flush=True)
        result = run_test_module(test_name, test_file)
        results[test_name] = result
        
        total_tests += result['tests_run']
        total_failures += result['failures']
        total_errors += result['errors']
        
        if result.get('timeout', False):
            status = "⏱️"
            time_str = f"{result['elapsed']:.2f}s"
            print(f"{status} TIMEOUT ({time_str}) - Move to test_{test_name}_large_time.py")
            if 'error_msg' in result:
                print(f"    {result['error_msg']}")
        else:
            status = "✅" if result['success'] else "❌"
            time_str = f"{result['elapsed']:.2f}s"
            print(f"{status} ({result['tests_run']} tests, {time_str})")
            
            if not result['success']:
                if 'error_msg' in result:
                    print(f"    Error: {result['error_msg']}")
                elif result['failures'] > 0 or result['errors'] > 0:
                    print(f"    ⚠️  {result['failures']} failures, {result['errors']} errors")
    
    # Run slow tests last
    print(f"\n🐢 PHASE 2: Slow Tests ({len(slow_test_files)} files)")
    print("-" * 80)
    
    for i, (test_name, test_file) in enumerate(slow_test_files, 1):
        print(f"[{i}/{len(slow_test_files)}] {test_name}... ", end='', flush=True)
        result = run_test_module(test_name, test_file)
        results[test_name] = result
        
        total_tests += result['tests_run']
        total_failures += result['failures']
        total_errors += result['errors']
        
        if result.get('timeout', False):
            status = "⏱️"
            time_str = f"{result['elapsed']:.2f}s"
            print(f"{status} TIMEOUT ({time_str}) - Move to test_{test_name}_large_time.py")
            if 'error_msg' in result:
                print(f"    {result['error_msg']}")
        else:
            status = "✅" if result['success'] else "❌"
            time_str = f"{result['elapsed']:.2f}s"
            print(f"{status} ({result['tests_run']} tests, {time_str})")
            
            if not result['success']:
                if 'error_msg' in result:
                    print(f"    Error: {result['error_msg']}")
                elif result['failures'] > 0 or result['errors'] > 0:
                    print(f"    ⚠️  {result['failures']} failures, {result['errors']} errors")
    
    overall_elapsed = time.time() - overall_start
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    successful_files = sum(1 for r in results.values() if r['success'])
    
    print(f"Files tested: {total_files}")
    print(f"Successful: {successful_files} ✅")
    print(f"Failed: {total_files - successful_files} ❌")
    print(f"Total tests: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Total time: {overall_elapsed:.2f}s")
    
    success_rate = (total_tests - total_failures - total_errors) / total_tests * 100 if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    # Timing breakdown
    fast_time = sum(r['elapsed'] for name, _ in fast_test_files for r in [results.get(name, {})] if r)
    slow_time = sum(r['elapsed'] for name, _ in slow_test_files for r in [results.get(name, {})] if r)
    
    print(f"\n⏱️  Timing Breakdown:")
    print(f"   Fast tests: {fast_time:.2f}s")
    print(f"   Slow tests: {slow_time:.2f}s")
    print(f"   Total: {overall_elapsed:.2f}s")
    
    # Timeout tests
    timeout_tests = [name for name, result in results.items() if result.get('timeout', False)]
    if timeout_tests:
        print(f"\n⏱️  Tests that exceeded 5-minute timeout ({len(timeout_tests)}):")
        for name in timeout_tests:
            elapsed = results[name]['elapsed']
            print(f"   - {name}: {elapsed:.2f}s (should be moved to {name}_large_time.py)")
    
    # Slowest tests
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('elapsed', 0), reverse=True)
    if sorted_results and sorted_results[0][1].get('elapsed', 0) > 1.0:
        print(f"\n🐌 Slowest Tests (>1s, excluding timeouts):")
        for name, result in sorted_results[:5]:
            if result.get('elapsed', 0) > 1.0 and not result.get('timeout', False):
                print(f"   {name}: {result['elapsed']:.2f}s")
    
    print("\n" + "=" * 80)
    
    total_timeouts = len(timeout_tests)
    if total_failures == 0 and total_errors == 0 and total_timeouts == 0:
        print("🎉 All tests passed within timeout!")
        return True
    else:
        if total_timeouts > 0:
            print(f"⏱️  {total_timeouts} test(s) exceeded timeout!")
        if total_failures + total_errors > 0:
            print(f"💥 {total_failures + total_errors} test(s) failed!")
        return False

if __name__ == '__main__':
    success = run_optimized_tests()
    sys.exit(0 if success else 1)

