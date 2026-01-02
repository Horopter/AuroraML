#!/usr/bin/env python3
"""
Run all AuroraML Python tests in parallel, ordered by fastest completion
"""

import sys
import os
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Try to import psutil, but handle gracefully if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available - memory profiling disabled. Install with: pip install psutil", flush=True)

# Force immediate output flushing
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("=" * 80)
print("AuroraML Test Runner (Parallel)")
print("=" * 80)
print(f"Python: {sys.executable}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)

# Add the build directory to Python path
build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_path)

test_dir = os.path.dirname(__file__)
print(f"Test directory: {test_dir}", flush=True)

# Global timeouts (seconds)
PROFILE_TIMEOUT_SECONDS = 10
TEST_TIMEOUT_SECONDS = 15 * 60

# Find all test files recursively
test_files = []
excluded_dirs = {"__pycache__", "logs", "build", "auroraml"}
for root, dirs, files in os.walk(test_dir):
    dirs[:] = [d for d in dirs if d not in excluded_dirs]
    for filename in files:
        if filename.startswith('test_') and filename.endswith('.py') and filename not in ['run_all_tests.py', 'run_all_tests_simple.py']:
            rel_path = os.path.relpath(os.path.join(root, filename), test_dir)
            test_files.append(rel_path)
test_files = sorted(test_files)

print(f"\nFound {len(test_files)} test files", flush=True)

# Step 1: Quick profiling pass to estimate test times (with 10s timeout)
print("\n" + "=" * 80)
print("PROFILING: Measuring test execution times...")
print("=" * 80)

def profile_test(test_file, timeout=PROFILE_TIMEOUT_SECONDS):
    """Quick profile run to estimate test time and memory"""
    start = time.time()
    max_memory_mb = 0
    try:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(test_dir, test_file)],
            cwd=test_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor memory usage if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(proc.pid)
                proc.communicate(timeout=timeout)
                # Get peak memory after completion
                try:
                    mem_info = process.memory_info()
                    max_memory_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            except Exception:
                proc.communicate(timeout=timeout)
        else:
            proc.communicate(timeout=timeout)
        
        elapsed = time.time() - start
        return (elapsed if proc.returncode == 0 else timeout * 2, max_memory_mb)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return (timeout * 2, timeout * 2)  # Treat timeout as slow
    except Exception:
        return (timeout * 2, timeout * 2)

# Profile all tests (can be done in parallel too, but sequential is fine for profiling)
test_times = {}
test_memory = {}
print("Profiling tests (time and memory)...", flush=True)
for i, test_file in enumerate(test_files, 1):
    print(f"  [{i}/{len(test_files)}] Profiling {test_file}... ", end='', flush=True)
    elapsed, memory_mb = profile_test(test_file)
    test_times[test_file] = elapsed
    test_memory[test_file] = memory_mb
    print(f"{elapsed:.2f}s, {memory_mb:.1f}MB", flush=True)

# Identify memory-intensive tests (by name patterns and actual memory usage)
memory_intensive_patterns = ['large', 'catboost', 'xgboost', 'gradientboosting', 'randomforest', 'mixture', 'decomposition', 'svd']
memory_intensive_tests = []
regular_tests = []

# Calculate memory statistics
memory_values = list(test_memory.values())
if memory_values:
    avg_memory = sum(memory_values) / len(memory_values)
    max_memory = max(memory_values)
    # Flag tests that use more than 2x average or >100MB
    memory_threshold = max(avg_memory * 2, 100)
else:
    memory_threshold = 100

print(f"\nMemory statistics: avg={avg_memory:.1f}MB, max={max_memory:.1f}MB, threshold={memory_threshold:.1f}MB", flush=True)

for test_file in test_files:
    is_memory_intensive = any(pattern in test_file.lower() for pattern in memory_intensive_patterns)
    # Also flag by actual memory usage
    if test_memory.get(test_file, 0) > memory_threshold:
        is_memory_intensive = True
        print(f"  ⚠️  {test_file} flagged as memory-intensive: {test_memory.get(test_file, 0):.1f}MB", flush=True)
    
    if is_memory_intensive:
        memory_intensive_tests.append(test_file)
    else:
        regular_tests.append(test_file)

# Sort regular tests by estimated time (fastest first)
sorted_regular = sorted(regular_tests, key=lambda x: test_times.get(x, 999))
# Sort memory-intensive tests by estimated time (fastest first)
sorted_memory = sorted(memory_intensive_tests, key=lambda x: test_times.get(x, 999))

print(f"\nRegular tests: {len(sorted_regular)} (will run in parallel)", flush=True)
print(f"Memory-intensive tests: {len(sorted_memory)} (will run last, sequentially)", flush=True)

# Step 2: Run all tests in parallel
print("\n" + "=" * 80)
print("EXECUTING: Running all tests in parallel...")
print("=" * 80)

results = {
    'total': 0,
    'passed': 0,
    'failed': 0,
    'timeout': 0,
    'error': 0
}
failed_tests = []
test_results = {}  # Store results for each test

# Thread-safe tracking for live updates
status_lock = threading.Lock()
running_tests = {}  # test_file -> start_time
completed_tests = {}  # test_file -> result

def run_test(test_file, show_output=True):
    """Run a single test file with live output"""
    start = time.time()
    start_memory = 0
    
    # Mark as running
    with status_lock:
        running_tests[test_file] = start
        print(f"\n{'='*80}", flush=True)
        print(f"▶️  STARTED: {test_file}", flush=True)
        print(f"{'='*80}", flush=True)
    
    result = {
        'file': test_file,
        'status': 'unknown',
        'elapsed': 0,
        'memory_mb': 0,
        'stdout': '',
        'stderr': ''
    }
    
    stdout_lines = []
    stderr_lines = []
    max_memory_mb = 0
    
    try:
        proc = subprocess.Popen(
            [sys.executable, '-u', os.path.join(test_dir, test_file)],
            cwd=test_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Monitor memory usage if psutil is available
        if PSUTIL_AVAILABLE:
            def monitor_memory():
                nonlocal max_memory_mb
                try:
                    process = psutil.Process(proc.pid)
                    while proc.poll() is None:
                        try:
                            mem_info = process.memory_info()
                            current_mb = mem_info.rss / (1024 * 1024)
                            max_memory_mb = max(max_memory_mb, current_mb)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            break
                        time.sleep(0.1)  # Check every 100ms
                except Exception:
                    pass
            
            memory_monitor = threading.Thread(target=monitor_memory, daemon=True)
            memory_monitor.start()
        
        # Stream output in real-time
        def read_output(pipe, lines_list, label):
            for line in iter(pipe.readline, ''):
                if line:
                    lines_list.append(line)
                    if show_output:
                        with status_lock:
                            print(f"[{test_file}] {line.rstrip()}", flush=True)
            pipe.close()
        
        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(target=read_output, args=(proc.stdout, stdout_lines, 'OUT'))
        stderr_thread = threading.Thread(target=read_output, args=(proc.stderr, stderr_lines, 'ERR'))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        try:
            # Wait for process to complete
            proc.wait(timeout=TEST_TIMEOUT_SECONDS)
            elapsed = time.time() - start
            
            # Wait for output threads to finish
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            
            result['elapsed'] = elapsed
            result['memory_mb'] = max_memory_mb
            result['stdout'] = ''.join(stdout_lines)
            result['stderr'] = ''.join(stderr_lines)
            
            if proc.returncode == 0:
                result['status'] = 'passed'
            else:
                result['status'] = 'failed'
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            result['elapsed'] = elapsed
            result['status'] = 'timeout'
            result['stdout'] = ''.join(stdout_lines)
            result['stderr'] = ''.join(stderr_lines)
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                
    except Exception as e:
        elapsed = time.time() - start
        result['elapsed'] = elapsed
        result['status'] = 'error'
        result['error'] = str(e)
        result['stdout'] = ''.join(stdout_lines)
        result['stderr'] = ''.join(stderr_lines)
    
    # Mark as completed
    with status_lock:
        running_tests.pop(test_file, None)
        completed_tests[test_file] = result
        
        status_icon = {
            'passed': '✅',
            'failed': '❌',
            'timeout': '⏱️',
            'error': '⚠️'
        }.get(result['status'], '?')
        
        print(f"{'='*80}", flush=True)
        memory_str = f", {result.get('memory_mb', 0):.1f}MB" if result.get('memory_mb', 0) > 0 else ""
        print(f"✓ COMPLETED: {status_icon} {test_file} ({result['elapsed']:.2f}s{memory_str})", flush=True)
        print(f"{'='*80}\n", flush=True)
    
    return result

def status_monitor():
    """Background thread to show running tests status"""
    while True:
        time.sleep(5)  # Update every 5 seconds
        with status_lock:
            if running_tests:
                current_time = time.time()
                running_list = []
                for test_file, start_time in running_tests.items():
                    elapsed = current_time - start_time
                    running_list.append(f"{test_file} ({elapsed:.1f}s)")
                if running_list:
                    print(f"\n⏳ STATUS: {len(running_list)} test(s) still running: {', '.join(running_list[:3])}{'...' if len(running_list) > 3 else ''}", flush=True)

# Step 2a: Run regular tests in parallel
print("\n" + "=" * 80)
print("PHASE 1: Running regular tests in parallel...")
print("=" * 80)

# Use number of CPU cores, but cap at reasonable limit
max_workers = min(len(sorted_regular), os.cpu_count() or 4, 8)
print(f"Running with {max_workers} parallel workers\n", flush=True)

# Start status monitor thread
monitor_thread = threading.Thread(target=status_monitor, daemon=True)
monitor_thread.start()

if sorted_regular:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all regular tests
        future_to_test = {executor.submit(run_test, test_file): test_file for test_file in sorted_regular}
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_test):
            test_file = future_to_test[future]
            completed += 1
            
            try:
                result = future.result()
                test_results[test_file] = result
                results['total'] += 1
                
                if result['status'] == 'passed':
                    results['passed'] += 1
                elif result['status'] == 'failed':
                    results['failed'] += 1
                    failed_tests.append(test_file)
                elif result['status'] == 'timeout':
                    results['timeout'] += 1
                    failed_tests.append(test_file)
                elif result['status'] == 'error':
                    results['error'] += 1
                    failed_tests.append(test_file)
                    
            except Exception as e:
                print(f"⚠️ {test_file} - Exception: {e}", flush=True)
                results['error'] += 1
                results['total'] += 1
                failed_tests.append(test_file)
    
    # Wait a bit for final status updates
    time.sleep(0.5)

# Step 2b: Run memory-intensive tests sequentially (one at a time)
if sorted_memory:
    print("\n" + "=" * 80)
    print("PHASE 2: Running memory-intensive tests sequentially...")
    print("=" * 80)
    print("Running one at a time to avoid memory issues\n", flush=True)
    
    for idx, test_file in enumerate(sorted_memory, 1):
        print(f"[{idx}/{len(sorted_memory)}] Memory-intensive: {test_file}", flush=True)
        
        try:
            result = run_test(test_file)
            test_results[test_file] = result
            results['total'] += 1
            
            if result['status'] == 'passed':
                results['passed'] += 1
            elif result['status'] == 'failed':
                results['failed'] += 1
                failed_tests.append(test_file)
            elif result['status'] == 'timeout':
                results['timeout'] += 1
                failed_tests.append(test_file)
            elif result['status'] == 'error':
                results['error'] += 1
                failed_tests.append(test_file)
                
        except Exception as e:
            print(f"⚠️ {test_file} - Exception: {e}", flush=True)
            results['error'] += 1
            results['total'] += 1
            failed_tests.append(test_file)
        
        # Small delay between memory-intensive tests to allow cleanup
        time.sleep(0.5)

# Print detailed results for failed tests
if failed_tests:
    print("\n" + "=" * 80)
    print("FAILED TESTS DETAILS")
    print("=" * 80)
    for test_file in failed_tests:
        if test_file in test_results:
            result = test_results[test_file]
            print(f"\n{test_file} ({result['status']}, {result['elapsed']:.2f}s):")
            if result.get('stdout'):
                lines = result['stdout'].split('\n')
                for line in lines[-10:]:  # Last 10 lines
                    if line.strip():
                        print(f"  {line}")
            if result.get('stderr'):
                lines = result['stderr'].split('\n')
                for line in lines[-10:]:  # Last 10 lines
                    if line.strip():
                        print(f"  {line}")
            if result.get('error'):
                print(f"  Error: {result['error']}")

# Print summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Total test files: {results['total']}")
print(f"✅ Passed: {results['passed']}")
print(f"❌ Failed: {results['failed']}")
print(f"⏱️  Timeout: {results['timeout']}")
print(f"⚠️  Error: {results['error']}")

if failed_tests:
    print(f"\nFailed test files ({len(failed_tests)}):")
    for test_file in failed_tests:
        print(f"  - {test_file}")

sys.exit(0 if not failed_tests else 1)
