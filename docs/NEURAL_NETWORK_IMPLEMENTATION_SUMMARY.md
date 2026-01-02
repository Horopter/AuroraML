> Note: For current coverage, see `docs/ALGORITHM_MATRIX.md`.

# Neural Network Implementation Summary

Status: ✅ Implemented (MLPClassifier, MLPRegressor)

Key locations:
- C++ core: `include/auroraml/neural_network.hpp`, `src/neural_network.cpp`
- Python bindings: `python/auroraml_bindings.cpp`
- Tests: `tests/neural_network/` and `tests_python/neural_network/`

Run tests with:
```bash
python3 tests_python/run_all_tests.py
ctest --test-dir build --output-on-failure
```
