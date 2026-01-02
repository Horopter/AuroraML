# AuroraML vs scikit-learn: Algorithm Comparison

The authoritative coverage table lives in `docs/ALGORITHM_MATRIX.md`.
For a flattened view (single table), see `docs/ALGORITHM_TABLE.md`.
This file provides context on how the matrix is derived and where to validate behavior.

## Ground truth sources

- Python bindings: `python/auroraml_bindings.cpp`
- C++ implementations: `include/auroraml/*.hpp` and `src/*.cpp`

## Compatibility notes

- `SVC` and `SVR` are linear-only in AuroraML.
- `ColumnTransformer` is dense-only (sparse output is not supported).
- `LDA` in AuroraML is a transformer; scikit-learn exposes LDA as a classifier (`LinearDiscriminantAnalysis`).
- `PermutationImportance` and `PartialDependence` are AuroraML classes; scikit-learn exposes similar functionality via functions/APIs.

## Testing

- C++ tests: `tests/` (run with CTest)
- Python tests: `tests_python/` (organized by module; `run_all_tests.py` discovers tests recursively)
- Timeout policy: `docs/TEST_TIMEOUT_POLICY.md`
