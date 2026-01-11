# IngenuityML Test Coverage

This document summarizes test coverage for IngenuityML and lists the full test catalog to aim for,
plus a matrix showing what is currently covered vs. not covered. Counts below are test-case counts
derived from the test definitions (not just file counts).

## Sources
- C++ tests: `tests/` (GTest macros).
- Python tests: `tests_python/` (unittest-style `test_*` functions).
- Large-time tests: `tests_python/*/*_large_time.py`.
- Build artifacts are excluded (`build/`, `tests_python/build/`).

## Test Count Summary (Exact)
- Total C++ tests: 673
- Total Python tests: 754
- Total Python large-time tests: 34
- Total tests (C++ + Python): 1427

## Target Baseline (for “tests that could be performed”)
- Baseline: 8 core test types per algorithm: API contract, Input validation, Correctness/sanity, Output quality, Determinism/randomness, Numerical stability, Performance/scaling, Serialization/persistence.
- Target counts below are exact given this baseline and the algorithm counts in `docs/STATUS_TODAY.md`.
- Remaining counts are shown per section and do not subtract surplus coverage from other sections.
- Non-algorithm sections (Utilities & Metrics) are excluded from target calculations.

## Test Catalog (All Test Types to Consider)
The list below is the comprehensive set of test types applicable to this project. Each section can
be evaluated against these categories.

1) Build & Packaging
- CMake configure/build, pip editable install, wheel build, importability, version metadata.
2) API Contract
- Constructor defaults, `fit`/`predict`/`transform`/`inverse_transform`, `predict_proba`, `decision_function`,
  `get_params`/`set_params`, `is_fitted`, attribute existence (e.g., `classes_`).
3) Input Validation & Error Handling
- Empty inputs, shape mismatch, invalid params, NaN/Inf handling, type casting, unsupported options.
4) Correctness & Sanity Checks
- Basic accuracy/regression sanity, expected invariants, simple toy datasets, identity transforms.
5) Probabilistic Output Quality
- Probabilities sum to 1, within [0,1], monotonicity where expected, calibration sanity.
6) Determinism & Randomness
- `random_state` reproducibility, consistent outputs across runs, seed handling.
7) Numerical Stability
- Extreme values, singular matrices, convergence edge cases, under/overflow safety.
8) Performance & Scaling
- Large datasets, timeouts, algorithmic complexity sanity, runtime budgets.
9) Memory Behavior
- Peak memory usage, leaks, large intermediate allocations.
10) Serialization & Persistence
- Save/load round trips, invalid file handling, backward compatibility.
11) Integration & Composition
- Pipelines, ColumnTransformer, CV splitters, grid/random/halving search, meta-estimators.
12) Cross-Language Consistency
- C++ vs Python API parity on the same inputs.
13) Compatibility / Parity
- Scikit-learn behavioral parity for known reference cases.
14) Concurrency & Threading
- OpenMP on/off, thread safety, deterministic behavior with multi-threading.
15) Documentation & Examples
- Examples run without errors and match documented outputs.

## Coverage Matrix by Algorithm Section (test-level)
Counts are test-case counts (GTest TEST/TEST_F/TEST_P macros and Python `test_*` functions).
Target counts use the baseline above.
Section mapping uses test folder names with filename-based overrides for:
- `*xgboost*`/`*catboost*` -> Extras (non-sklearn)
- `*dummy*` -> Dummy Models
- `tests/preprocessing/test_impute*` -> Imputation
- `tests/preprocessing/test_feature_selection*` -> Feature Selection
Supporting utilities (metrics/random/utils) are tracked under Utilities & Metrics (non-algorithm).

| section | algorithms | C++ tests | Python tests | Python large-time tests | current total | target total | remaining | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Models | 36 | 89 | 75 | 0 | 164 | 288 | 124 | c++ + python |
| Support Vector Machines | 7 | 34 | 64 | 0 | 98 | 56 | 0 | c++ + python |
| Neighbors | 8 | 58 | 58 | 0 | 116 | 64 | 0 | c++ + python |
| Decision Trees | 4 | 16 | 21 | 0 | 37 | 32 | 0 | c++ + python |
| Ensemble & Boosting | 17 | 47 | 97 | 0 | 144 | 136 | 0 | c++ + python |
| Naive Bayes | 5 | 15 | 18 | 0 | 33 | 40 | 7 | c++ + python |
| Discriminant Analysis | 2 | 27 | 12 | 0 | 39 | 16 | 0 | c++ + python |
| Neural Networks | 3 | 0 | 27 | 0 | 27 | 24 | 0 | python only |
| Clustering | 13 | 20 | 44 | 0 | 64 | 104 | 40 | c++ + python |
| Mixture Models | 2 | 6 | 2 | 0 | 8 | 16 | 8 | c++ + python |
| Decomposition & Dimensionality Reduction | 14 | 64 | 59 | 0 | 123 | 112 | 0 | c++ + python |
| Manifold Learning | 5 | 0 | 4 | 0 | 4 | 40 | 36 | python only |
| Cross-Decomposition | 4 | 0 | 4 | 0 | 4 | 32 | 28 | python only |
| Random Projection | 2 | 0 | 2 | 0 | 2 | 16 | 14 | python only |
| Gaussian Processes | 2 | 0 | 2 | 0 | 2 | 16 | 14 | python only |
| Covariance Estimation | 6 | 6 | 6 | 0 | 12 | 48 | 36 | c++ + python |
| Density Estimation | 1 | 0 | 1 | 0 | 1 | 8 | 7 | python only |
| Outlier Detection | 3 | 8 | 2 | 0 | 10 | 24 | 14 | c++ + python |
| Semi-Supervised Learning | 3 | 7 | 3 | 0 | 10 | 24 | 14 | c++ + python |
| Preprocessing & Feature Engineering | 17 | 20 | 31 | 0 | 51 | 136 | 85 | c++ + python |
| Imputation | 4 | 8 | 4 | 0 | 12 | 32 | 20 | c++ + python |
| Feature Selection | 11 | 17 | 12 | 0 | 29 | 88 | 59 | c++ + python |
| Model Selection (CV splitters & search) | 18 | 56 | 36 | 0 | 92 | 144 | 52 | c++ + python |
| Pipeline & Composition | 4 | 12 | 10 | 0 | 22 | 32 | 10 | c++ + python |
| Calibration & Isotonic | 2 | 12 | 5 | 2 | 17 | 16 | 0 | c++ + python |
| Meta-Estimators (Multiclass/Multioutput) | 7 | 7 | 7 | 0 | 14 | 56 | 42 | c++ + python |
| Dummy Models | 2 | 13 | 11 | 0 | 24 | 16 | 0 | c++ + python |
| Inspection / Explainability | 2 | 6 | 3 | 0 | 9 | 16 | 7 | c++ + python |
| Extras (non-sklearn) | 4 | 30 | 64 | 32 | 94 | 32 | 0 | c++ + python |
| Utilities & Metrics (non-algorithm) | n/a | 95 | 70 | 0 | 165 | n/a | n/a | c++ + python |

## Section Summary (Created vs Needed)
Created = C++ + Python tests; Needed = Remaining to baseline target (no credit for surplus in other sections).

| section | created tests | needed tests |
| --- | --- | --- |
| Linear Models | 164 | 124 |
| Support Vector Machines | 98 | 0 |
| Neighbors | 116 | 0 |
| Decision Trees | 37 | 0 |
| Ensemble & Boosting | 144 | 0 |
| Naive Bayes | 33 | 7 |
| Discriminant Analysis | 39 | 0 |
| Neural Networks | 27 | 0 |
| Clustering | 64 | 40 |
| Mixture Models | 8 | 8 |
| Decomposition & Dimensionality Reduction | 123 | 0 |
| Manifold Learning | 4 | 36 |
| Cross-Decomposition | 4 | 28 |
| Random Projection | 2 | 14 |
| Gaussian Processes | 2 | 14 |
| Covariance Estimation | 12 | 36 |
| Density Estimation | 1 | 7 |
| Outlier Detection | 10 | 14 |
| Semi-Supervised Learning | 10 | 14 |
| Preprocessing & Feature Engineering | 51 | 85 |
| Imputation | 12 | 20 |
| Feature Selection | 29 | 59 |
| Model Selection (CV splitters & search) | 92 | 52 |
| Pipeline & Composition | 22 | 10 |
| Calibration & Isotonic | 17 | 0 |
| Meta-Estimators (Multiclass/Multioutput) | 14 | 42 |
| Dummy Models | 24 | 0 |
| Inspection / Explainability | 9 | 7 |
| Extras (non-sklearn) | 94 | 0 |

## Overall Summary (Created vs Needed)
- Algorithms tracked (from `docs/STATUS_TODAY.md`): 208
- Baseline tests per algorithm: 8
- Target total tests (baseline): 1664
- Created tests (algorithm sections only): 1262
- Remaining tests (sum of per-section gaps, no surplus offset): 617
- Net remaining tests (target minus created): 402

## Coverage Matrix by Test Type
| test type | status | evidence (examples) | gaps |
| --- | --- | --- | --- |
| Build & Packaging | not covered | none | add CI or scripted build/install/import tests |
| API Contract | covered | `tests_python/**/test_*.py`, `tests/**/test_*.cpp` | none identified |
| Input Validation & Error Handling | partial | `tests/svm/test_svm.cpp`, `tests_python/**` with invalid cases | extend to all modules/options |
| Correctness & Sanity Checks | covered | broad coverage across modules | add more reference-case checks |
| Probabilistic Output Quality | partial | `tests_python/naive_bayes/test_naive_bayes.py`, `tests_python/svm/test_svm.py` | add coverage for all classifiers |
| Determinism & Randomness | partial | `tests_python/random/test_random.py`, `tests_python/ensemble/test_xgboost.py` | add explicit reproducibility checks everywhere |
| Numerical Stability | partial | scattered checks for finite outputs | add stress/edge-case suites |
| Performance & Scaling | partial | `tests_python/*/*_large_time.py` | add large-time tests per section |
| Memory Behavior | not covered | none | add memory profiling/limits tests |
| Serialization & Persistence | partial | `tests_python/svm/test_svm.py`, `tests_python/naive_bayes/test_naive_bayes.py`, `tests_python/preprocessing/test_preprocessing.py` | add save/load for all estimators |
| Integration & Composition | covered | `tests_python/pipeline/`, `tests_python/model_selection/` | expand cross-module pipelines |
| Cross-Language Consistency | not covered | none | add paired C++/Python parity checks |
| Compatibility / Parity | partial | functional parity implied in module tests | add explicit sklearn parity assertions |
| Concurrency & Threading | not covered | none | add OpenMP/threading tests |
| Documentation & Examples | not covered | none | add example execution tests |

## Yet-To-Be-Tested Highlights
- Build/packaging automation (CMake + pip install + import verification).
- Memory profiling and leak checks.
- Concurrency/OpenMP behavior and determinism under threading.
- Explicit C++ vs Python parity tests.
- Example script execution and output validation.
- Broad, explicit sklearn parity baselines for key estimators.
