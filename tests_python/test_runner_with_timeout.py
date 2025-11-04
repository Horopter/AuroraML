#!/usr/bin/env python3
"""
Test Runner with 5-minute timeout per test
Tests exceeding timeout will be reported and can be moved to _large_time files
"""

import sys
import os
import unittest
import time
import signal
import subprocess

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

TIMEOUT_SECONDS = 300  # 5 minutes

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError(f"Test exceeded {TIMEOUT_SECONDS} second timeout")

def run_test_with_timeout(test_file):
    """Run a test file with timeout using subprocess"""
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=os.path.dirname(test_file)
        )
        
        elapsed = time.time() - start_time
        
        output = result.stdout + result.stderr
        
        # Parse test results
        import re
        match = re.search(r'Ran (\d+) test', output)
        tests_run = int(match.group(1)) if match else 0
        
        success = result.returncode == 0 and 'OK' in output
        has_failures = 'FAILED' in output or 'ERROR' in output
        
        return {
            'success': success,
            'tests_run': tests_run,
            'failures': 1 if has_failures and not success else 0,
            'errors': 1 if result.returncode != 0 and not success else 0,
            'elapsed': elapsed,
            'timeout': False,
            'output': output
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'elapsed': elapsed,
            'timeout': True,
            'output': f'Test exceeded {TIMEOUT_SECONDS} second timeout'
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'elapsed': elapsed,
            'timeout': False,
            'output': str(e)
        }

def find_all_test_files():
    """Find all test_*.py files"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = []
    
    for filename in os.listdir(test_dir):
        if filename.startswith('test_') and filename.endswith('.py') and not filename.endswith('_large_time.py'):
            # Exclude runner files
            if 'runner' not in filename and 'timeout' not in filename:
                test_files.append(os.path.join(test_dir, filename))
    
    return sorted(test_files)

def main():
    """Run all tests with timeout"""
    print(f"🚀 AuroraML Test Runner with {TIMEOUT_SECONDS}s timeout per test")
    print("=" * 80)
    
    test_files = find_all_test_files()
    print(f"Found {len(test_files)} test files\n")
    
    results = []
    timeout_tests = []
    
    for i, test_file in enumerate(test_files, 1):
        test_name = os.path.basename(test_file)
        print(f"[{i}/{len(test_files)}] {test_name}... ", end='', flush=True)
        
        result = run_test_with_timeout(test_file)
        result['name'] = test_name
        results.append(result)
        
        if result['timeout']:
            timeout_tests.append(test_name)
            print(f"⏱️  TIMEOUT ({result['elapsed']:.1f}s)")
        elif result['success']:
            print(f"✅ PASS ({result['tests_run']} tests, {result['elapsed']:.2f}s)")
        else:
            print(f"❌ FAIL ({result['tests_run']} tests, {result['elapsed']:.2f}s)")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    total_tests = sum(r['tests_run'] for r in results)
    total_failures = sum(r['failures'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    total_timeouts = len(timeout_tests)
    successful = sum(1 for r in results if r['success'] and not r['timeout'])
    
    print(f"Total test files: {len(test_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(test_files) - successful - total_timeouts}")
    print(f"⏱️  Timeouts: {total_timeouts}")
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    
    if timeout_tests:
        print(f"\n⚠️  Tests that exceeded {TIMEOUT_SECONDS}s timeout:")
        for test_name in timeout_tests:
            print(f"   - {test_name}")
        print(f"\n💡 These tests should be moved to test_<component>_large_time.py files")
    
    # Slow tests (>10s)
    slow_tests = [(r['name'], r['elapsed']) for r in results if r['elapsed'] > 10 and not r['timeout']]
    if slow_tests:
        print(f"\n🐌 Slow tests (>10s):")
        for name, elapsed in sorted(slow_tests, key=lambda x: x[1], reverse=True):
            print(f"   - {name}: {elapsed:.2f}s")
    
    return total_timeouts == 0 and total_failures == 0 and total_errors == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

