# IngenuityML - High-Performance C++ Machine Learning Library

IngenuityML is a C++ machine learning library with Python bindings. It targets a scikit-learn-style API with strong performance and growing feature parity.

## Goal
- Deliver a fast, reliable C++ ML core with a Python-facing API that mirrors scikit-learn where practical.
- Track and close coverage gaps against scikit-learn with transparent, test-backed progress.
- Keep tests deterministic and fast enough for regular iteration.

## How to get there (Approach)
- Implement algorithms in C++ (`include/ingenuityml/*.hpp`, `src/*.cpp`) and expose them via pybind11 (`python/ingenuityml_bindings.cpp`).
- Add C++ tests under `tests/` and Python tests under `tests_python/`.
- Update `docs/STATUS_TODAY.md` with coverage and implementation status, and keep `docs/NEXT_STEPS.md` current as gaps close.
- Use Eigen for linear algebra and OpenMP where available for parallel sections.

## Documentation map
- `docs/STATUS_TODAY.md` - current coverage and implementation snapshot.
- `docs/NEXT_STEPS.md` - prioritized backlog to reach parity.

## Build and install

### Prerequisites
- C++17 compatible compiler
- CMake 3.12+
- Python 3.8+
- NumPy, pybind11, Eigen3

### Build from source
```bash
git clone https://github.com/your-repo/IngenuityML.git
cd IngenuityML
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Python installation
```bash
pip install -e .
```

## Quick start

### Classification
```python
import ingenuityml
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = ingenuityml.model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf = ingenuityml.ensemble.RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
acc = ingenuityml.metrics.accuracy_score(y_test, pred)
print(f"Random Forest accuracy: {acc:.3f}")
```

### Regression
```python
import ingenuityml
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = ingenuityml.model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42
)

ridge = ingenuityml.linear_model.Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

pred = ridge.predict(X_test)
r2 = ingenuityml.metrics.r2_score(y_test, pred)
print(f"Ridge R2: {r2:.3f}")
```

## Testing and timeout policy
C++ tests target ~5 minutes per test file; the Python runner currently enforces a 15-minute timeout per file.

- C++ tests: CTest enforces a 300-second timeout per test via `tests/CMakeLists.txt`.
- Python tests: `tests_python/run_all_tests.py` profiles tests with a 10-second timeout, then runs fast tests in parallel and memory-intensive ones sequentially; per-file timeout is 15 minutes.
- Slow tests should move to `_large_time` files:
  - `test_<component>_large_time.py`
  - `test_<component>_large_time.cpp`

```bash
# Run all Python tests
python3 tests_python/run_all_tests.py

# Run C++ tests (timeout enforced by CTest)
cd build && ctest --timeout 300
```

## License
Creative Commons Attribution-ShareAlike 4.0 International License.
