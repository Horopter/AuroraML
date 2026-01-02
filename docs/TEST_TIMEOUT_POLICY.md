# Test Timeout Policy

## Overview
Target runtime is **~5 minutes per test file**. C++ tests enforce timeouts via CTest; the Python runner currently profiles tests with a short timeout and then runs them without a hard cap.

## Policy

### Timeout Enforcement
- **C++ Tests**: All C++ tests registered in `tests/CMakeLists.txt` have a 300-second timeout set via `set_tests_properties(TIMEOUT 300)`
- **Python Tests**: `tests_python/run_all_tests.py` profiles each test with a short timeout and then runs tests (fast in parallel, slower ones sequentially)

### Long-Running Tests
If a test file consistently exceeds the 5-minute timeout, it should be moved to a separate file with the naming convention:
```
test_<component_name>_large_time.py  (for Python)
test_<component_name>_large_time.cpp (for C++)
```

### Test Runner

- `tests_python/run_all_tests.py` (recursive discovery, parallel execution, memory-intensive tests run last)

### Usage

```bash
# Run all Python tests
python3 tests_python/run_all_tests.py

# Run C++ tests (timeout enforced by CTest)
cd build && ctest --timeout 300
```

### Identifying Timeout Violations

Slow tests should be moved to `_large_time` files so the runner can schedule them last.

### Moving Tests to _large_time Files

When a test exceeds timeout:

1. Create `test_<component>_large_time.py` or `test_<component>_large_time.cpp`
2. Move the slow test cases to the new file
3. Update any test runners to exclude `_large_time` files from regular runs
4. Document why the test needs more time in a comment

### Example

If `test_isotonic.py` exceeds timeout:
```bash
# Create the large time version
cp tests_python/test_isotonic.py tests_python/test_isotonic_large_time.py

# Keep only fast tests in test_isotonic.py
# Move slow tests to test_isotonic_large_time.py
```

## Current Status

- ✅ All C++ tests have timeout configured in CMakeLists.txt
- ✅ Python test runners enforce 5-minute timeout
- ⚠️ Monitor for tests that consistently timeout and move them to `_large_time` files
