# AuroraML - High-Performance C++ Machine Learning Library

AuroraML is a C++ machine learning library with Python bindings. It provides a scikit-learn-style API for a growing subset of estimators and utilities.

## Coverage and compatibility

Authoritative coverage lives in `docs/ALGORITHM_MATRIX.md`, which lists each algorithm and whether it exists in scikit-learn and/or AuroraML. A flattened table is available in `docs/ALGORITHM_TABLE.md`.

Notes on current compatibility:
- `SVC` and `SVR` are linear-only in AuroraML.
- `ColumnTransformer` is dense-only (sparse output is not supported).
- `LDA` in AuroraML is a transformer; scikit-learn exposes LDA as a classifier (`LinearDiscriminantAnalysis`).

## Architecture

- C++ core built on Eigen for linear algebra.
- Python bindings via pybind11.
- Optional OpenMP parallelism where available.

## Installation

### Prerequisites
- C++17 compatible compiler
- CMake 3.12+
- Python 3.8+
- NumPy, pybind11, Eigen3

### Build from source
```bash
git clone https://github.com/your-repo/AuroraML.git
cd AuroraML
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Python installation
```bash
pip install -e .
```

## Quick start examples

### Classification
```python
import auroraml
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = auroraml.model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf = auroraml.ensemble.RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
acc = auroraml.metrics.accuracy_score(y_test, pred)
print(f"Random Forest accuracy: {acc:.3f}")
```

### Regression
```python
import auroraml
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = auroraml.model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42
)

ridge = auroraml.linear_model.Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

pred = ridge.predict(X_test)
r2 = auroraml.metrics.r2_score(y_test, pred)
print(f"Ridge R2: {r2:.3f}")
```

### Clustering
```python
import auroraml
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

kmeans = auroraml.cluster.KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
print(f"KMeans clusters: {len(set(labels))}")
```

## Testing

- Python tests live in `tests_python/`.
- C++ tests live in `tests/` and run via CTest.
- Test timeouts are documented in `docs/TEST_TIMEOUT_POLICY.md`.

```bash
# Run all Python tests
python tests_python/run_all_tests.py

# Run C++ tests
cd build && ctest
```

## License

Creative Commons Attribution-ShareAlike 4.0 International License.
