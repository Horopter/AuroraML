# Test Timeout Policy

## Overview
All tests (both C++ and Python) must complete within **5 minutes (300 seconds)** per test file.

## Policy

### Timeout Enforcement
- **C++ Tests**: All C++ tests registered in `tests/CMakeLists.txt` have a 300-second timeout set via `set_tests_properties(TIMEOUT 300)`
- **Python Tests**: All Python test runners use `subprocess.run(timeout=300)` to enforce 5-minute limits

### Long-Running Tests
If a test file consistently exceeds the 5-minute timeout, it should be moved to a separate file with the naming convention:
```
test_<component_name>_large_time.py  (for Python)
test_<component_name>_large_time.cpp (for C++)
```

### Test Runners

1. **test_runner_with_timeout.py**: Basic runner that identifies tests exceeding timeout
2. **test_runner_optimized.py**: Optimized runner that runs fast tests first, then slow tests, with timeout enforcement

### Usage

```bash
# Run all tests with timeout enforcement
python3 tests_python/test_runner_optimized.py

# Check for timeout violations
python3 tests_python/test_runner_with_timeout.py

# Run C++ tests (timeout enforced by CTest)
cd build && ctest --timeout 300
```

### Identifying Timeout Violations

Tests that timeout will:
- Show `⏱️ TIMEOUT` in the output
- Report the elapsed time
- Suggest moving to `_large_time` file

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

