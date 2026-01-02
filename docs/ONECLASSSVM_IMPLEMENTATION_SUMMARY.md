> Note: For current coverage, see `docs/ALGORITHM_MATRIX.md`.

# OneClassSVM Implementation Summary

Status: ✅ Implemented

Key locations:
- C++ core: `include/auroraml/svm.hpp`, `src/svm.cpp`
- Python bindings: `python/auroraml_bindings.cpp`
- Tests: `tests/svm/` and `tests_python/svm/`

Run tests with:
```bash
python3 tests_python/run_all_tests.py
ctest --test-dir build --output-on-failure
```
