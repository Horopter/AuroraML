# IngenuityML Status (What Was Done Until Now)

## Snapshot
- Coverage tracked: 208 total algorithms; 208 implemented in IngenuityML (208 exposed in Python), 0 C++ only, 0 partial, 0 missing.
- Scikit-learn parity: 201 scikit-learn algorithms tracked; 201 exposed in Python; 0 remain for parity.
- Compatibility notes: ColumnTransformer is dense-only; LDA is also available as a transformer (scikit-learn exposes LinearDiscriminantAnalysis).
- Tests: C++ and Python suites are in place; see `README.md` for timeout policy and commands.

## Ground truth sources
- Python bindings: `python/ingenuityml_bindings.cpp`
- C++ implementations: `include/ingenuityml/*.hpp` and `src/*.cpp`

## Implemented highlights

### Neural Networks
Status: Implemented (MLPClassifier, MLPRegressor)
Key locations:
- C++ core: `include/ingenuityml/neural_network.hpp`, `src/neural_network.cpp`
- Python bindings: `python/ingenuityml_bindings.cpp`
- Tests: `tests/neural_network/`, `tests_python/neural_network/`

### OneClassSVM
Status: Implemented
Key locations:
- C++ core: `include/ingenuityml/svm.hpp`, `src/svm.cpp`
- Python bindings: `python/ingenuityml_bindings.cpp`
- Tests: `tests/svm/`, `tests_python/svm/`

## Coverage Matrix

Ground truth is based on the current IngenuityML codebase (Python bindings + C++ headers).
Use this matrix as the single source of truth for coverage; other docs link here.

Legend:
- `yes`: available in IngenuityML Python API
- `C++ only`: implemented in C++ but not exposed in Python
- `partial`: present but not fully equivalent to scikit-learn
- `no`: not implemented in IngenuityML

Notes:
- `ColumnTransformer` is dense-only (sparse output not supported).
- `LDA` in IngenuityML is a transformer; `LinearDiscriminantAnalysis` is provided for scikit-learn parity.
- `PermutationImportance` and `PartialDependence` are IngenuityML classes; scikit-learn exposes similar functionality via functions/APIs.

<!-- SUMMARY_STATS_START -->
## Summary Statistics

Overall (all rows in this matrix):
- Total algorithms tracked: 208
- Implemented in IngenuityML (any form): 208
- Implemented in IngenuityML Python API: 208
- Partially implemented: 0
- Implemented in C++ only: 0
- Not implemented: 0

Scikit-learn coverage (rows where `sklearn` = `yes`):
- Total scikit-learn algorithms tracked: 201
- Implemented in IngenuityML (any form): 201
- Implemented in IngenuityML Python API: 201
- Partially implemented: 0
- Implemented in C++ only: 0
- Missing in IngenuityML: 0
- Remaining for IngenuityML Python parity: 0

Per-section scikit-learn coverage:

| section | sklearn total | ingenuityml yes | partial | C++ only | missing |
| --- | --- | --- | --- | --- | --- |
| Linear Models | 36 | 36 | 0 | 0 | 0 |
| Support Vector Machines | 7 | 7 | 0 | 0 | 0 |
| Neighbors | 8 | 8 | 0 | 0 | 0 |
| Decision Trees | 4 | 4 | 0 | 0 | 0 |
| Ensemble & Boosting | 17 | 17 | 0 | 0 | 0 |
| Naive Bayes | 5 | 5 | 0 | 0 | 0 |
| Discriminant Analysis | 2 | 2 | 0 | 0 | 0 |
| Neural Networks | 3 | 3 | 0 | 0 | 0 |
| Clustering | 13 | 13 | 0 | 0 | 0 |
| Mixture Models | 2 | 2 | 0 | 0 | 0 |
| Decomposition & Dimensionality Reduction | 13 | 13 | 0 | 0 | 0 |
| Manifold Learning | 5 | 5 | 0 | 0 | 0 |
| Cross-Decomposition | 4 | 4 | 0 | 0 | 0 |
| Random Projection | 2 | 2 | 0 | 0 | 0 |
| Gaussian Processes | 2 | 2 | 0 | 0 | 0 |
| Covariance Estimation | 6 | 6 | 0 | 0 | 0 |
| Density Estimation | 1 | 1 | 0 | 0 | 0 |
| Outlier Detection | 3 | 3 | 0 | 0 | 0 |
| Semi-Supervised Learning | 3 | 3 | 0 | 0 | 0 |
| Preprocessing & Feature Engineering | 17 | 17 | 0 | 0 | 0 |
| Imputation | 4 | 4 | 0 | 0 | 0 |
| Feature Selection | 11 | 11 | 0 | 0 | 0 |
| Model Selection (CV splitters & search) | 18 | 18 | 0 | 0 | 0 |
| Pipeline & Composition | 4 | 4 | 0 | 0 | 0 |
| Calibration & Isotonic | 2 | 2 | 0 | 0 | 0 |
| Meta-Estimators (Multiclass/Multioutput) | 7 | 7 | 0 | 0 | 0 |
| Dummy Models | 2 | 2 | 0 | 0 | 0 |
<!-- SUMMARY_STATS_END -->

## Linear Models

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| LinearRegression | yes | yes |
| Ridge | yes | yes |
| RidgeCV | yes | yes |
| Lasso | yes | yes |
| LassoCV | yes | yes |
| ElasticNet | yes | yes |
| ElasticNetCV | yes | yes |
| Lars | yes | yes |
| LarsCV | yes | yes |
| LassoLars | yes | yes |
| LassoLarsCV | yes | yes |
| LassoLarsIC | yes | yes |
| OrthogonalMatchingPursuit | yes | yes |
| OrthogonalMatchingPursuitCV | yes | yes |
| BayesianRidge | yes | yes |
| ARDRegression | yes | yes |
| HuberRegressor | yes | yes |
| RANSACRegressor | yes | yes |
| TheilSenRegressor | yes | yes |
| SGDRegressor | yes | yes |
| SGDClassifier | yes | yes |
| PassiveAggressiveRegressor | yes | yes |
| PassiveAggressiveClassifier | yes | yes |
| Perceptron | yes | yes |
| LogisticRegression | yes | yes |
| LogisticRegressionCV | yes | yes |
| RidgeClassifier | yes | yes |
| RidgeClassifierCV | yes | yes |
| QuantileRegressor | yes | yes |
| PoissonRegressor | yes | yes |
| GammaRegressor | yes | yes |
| TweedieRegressor | yes | yes |
| MultiTaskLasso | yes | yes |
| MultiTaskLassoCV | yes | yes |
| MultiTaskElasticNet | yes | yes |
| MultiTaskElasticNetCV | yes | yes |

## Support Vector Machines

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| SVC | yes | yes |
| NuSVC | yes | yes |
| LinearSVC | yes | yes |
| SVR | yes | yes |
| NuSVR | yes | yes |
| LinearSVR | yes | yes |
| OneClassSVM | yes | yes |

## Neighbors

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| KNeighborsClassifier | yes | yes |
| KNeighborsRegressor | yes | yes |
| RadiusNeighborsClassifier | yes | yes |
| RadiusNeighborsRegressor | yes | yes |
| NearestNeighbors | yes | yes |
| NearestCentroid | yes | yes |
| KNeighborsTransformer | yes | yes |
| RadiusNeighborsTransformer | yes | yes |

## Decision Trees

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| DecisionTreeClassifier | yes | yes |
| DecisionTreeRegressor | yes | yes |
| ExtraTreeClassifier | yes | yes |
| ExtraTreeRegressor | yes | yes |

## Ensemble & Boosting

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| RandomForestClassifier | yes | yes |
| RandomForestRegressor | yes | yes |
| ExtraTreesClassifier | yes | yes |
| ExtraTreesRegressor | yes | yes |
| BaggingClassifier | yes | yes |
| BaggingRegressor | yes | yes |
| AdaBoostClassifier | yes | yes |
| AdaBoostRegressor | yes | yes |
| GradientBoostingClassifier | yes | yes |
| GradientBoostingRegressor | yes | yes |
| HistGradientBoostingClassifier | yes | yes |
| HistGradientBoostingRegressor | yes | yes |
| VotingClassifier | yes | yes |
| VotingRegressor | yes | yes |
| StackingClassifier | yes | yes |
| StackingRegressor | yes | yes |
| RandomTreesEmbedding | yes | yes |

## Naive Bayes

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| GaussianNB | yes | yes |
| MultinomialNB | yes | yes |
| BernoulliNB | yes | yes |
| ComplementNB | yes | yes |
| CategoricalNB | yes | yes |

## Discriminant Analysis

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| LinearDiscriminantAnalysis | yes | yes |
| QuadraticDiscriminantAnalysis | yes | yes |

## Neural Networks

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| MLPClassifier | yes | yes |
| MLPRegressor | yes | yes |
| BernoulliRBM | yes | yes |

## Clustering

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| KMeans | yes | yes |
| MiniBatchKMeans | yes | yes |
| BisectingKMeans | yes | yes |
| AffinityPropagation | yes | yes |
| MeanShift | yes | yes |
| SpectralClustering | yes | yes |
| AgglomerativeClustering | yes | yes |
| DBSCAN | yes | yes |
| OPTICS | yes | yes |
| Birch | yes | yes |
| FeatureAgglomeration | yes | yes |
| SpectralBiclustering | yes | yes |
| SpectralCoclustering | yes | yes |

## Mixture Models

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| GaussianMixture | yes | yes |
| BayesianGaussianMixture | yes | yes |

## Decomposition & Dimensionality Reduction

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| PCA | yes | yes |
| IncrementalPCA | yes | yes |
| KernelPCA | yes | yes |
| SparsePCA | yes | yes |
| MiniBatchSparsePCA | yes | yes |
| TruncatedSVD | yes | yes |
| FastICA | yes | yes |
| FactorAnalysis | yes | yes |
| NMF | yes | yes |
| MiniBatchNMF | yes | yes |
| DictionaryLearning | yes | yes |
| MiniBatchDictionaryLearning | yes | yes |
| LatentDirichletAllocation | yes | yes |
| LDA (Linear Discriminant Analysis transformer) | no | yes |

## Manifold Learning

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| TSNE | yes | yes |
| MDS | yes | yes |
| Isomap | yes | yes |
| LocallyLinearEmbedding | yes | yes |
| SpectralEmbedding | yes | yes |

## Cross-Decomposition

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| PLSCanonical | yes | yes |
| PLSRegression | yes | yes |
| CCA | yes | yes |
| PLSSVD | yes | yes |

## Random Projection

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| GaussianRandomProjection | yes | yes |
| SparseRandomProjection | yes | yes |

## Gaussian Processes

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| GaussianProcessClassifier | yes | yes |
| GaussianProcessRegressor | yes | yes |

## Covariance Estimation

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| EmpiricalCovariance | yes | yes |
| ShrunkCovariance | yes | yes |
| LedoitWolf | yes | yes |
| OAS | yes | yes |
| MinCovDet | yes | yes |
| EllipticEnvelope | yes | yes |

## Density Estimation

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| KernelDensity | yes | yes |

## Outlier Detection

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| IsolationForest | yes | yes |
| LocalOutlierFactor | yes | yes |
| EllipticEnvelope | yes | yes |

## Semi-Supervised Learning

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| LabelPropagation | yes | yes |
| LabelSpreading | yes | yes |
| SelfTrainingClassifier | yes | yes |

## Preprocessing & Feature Engineering

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| StandardScaler | yes | yes |
| MinMaxScaler | yes | yes |
| MaxAbsScaler | yes | yes |
| RobustScaler | yes | yes |
| Normalizer | yes | yes |
| Binarizer | yes | yes |
| PolynomialFeatures | yes | yes |
| OneHotEncoder | yes | yes |
| OrdinalEncoder | yes | yes |
| LabelEncoder | yes | yes |
| LabelBinarizer | yes | yes |
| MultiLabelBinarizer | yes | yes |
| KBinsDiscretizer | yes | yes |
| QuantileTransformer | yes | yes |
| PowerTransformer | yes | yes |
| FunctionTransformer | yes | yes |
| SplineTransformer | yes | yes |

## Imputation

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| SimpleImputer | yes | yes |
| KNNImputer | yes | yes |
| IterativeImputer | yes | yes |
| MissingIndicator | yes | yes |

## Feature Selection

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| VarianceThreshold | yes | yes |
| SelectKBest | yes | yes |
| SelectPercentile | yes | yes |
| SelectFpr | yes | yes |
| SelectFdr | yes | yes |
| SelectFwe | yes | yes |
| GenericUnivariateSelect | yes | yes |
| SelectFromModel | yes | yes |
| RFE | yes | yes |
| RFECV | yes | yes |
| SequentialFeatureSelector | yes | yes |

## Model Selection (CV splitters & search)

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| KFold | yes | yes |
| StratifiedKFold | yes | yes |
| GroupKFold | yes | yes |
| TimeSeriesSplit | yes | yes |
| RepeatedKFold | yes | yes |
| RepeatedStratifiedKFold | yes | yes |
| ShuffleSplit | yes | yes |
| StratifiedShuffleSplit | yes | yes |
| GroupShuffleSplit | yes | yes |
| PredefinedSplit | yes | yes |
| LeaveOneOut | yes | yes |
| LeavePOut | yes | yes |
| GridSearchCV | yes | yes |
| RandomizedSearchCV | yes | yes |
| HalvingGridSearchCV | yes | yes |
| HalvingRandomSearchCV | yes | yes |
| ParameterGrid | yes | yes |
| ParameterSampler | yes | yes |

## Pipeline & Composition

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| Pipeline | yes | yes |
| FeatureUnion | yes | yes |
| ColumnTransformer | yes | yes |
| TransformedTargetRegressor | yes | yes |

## Calibration & Isotonic

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| CalibratedClassifierCV | yes | yes |
| IsotonicRegression | yes | yes |

## Meta-Estimators (Multiclass/Multioutput)

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| OneVsRestClassifier | yes | yes |
| OneVsOneClassifier | yes | yes |
| OutputCodeClassifier | yes | yes |
| ClassifierChain | yes | yes |
| MultiOutputClassifier | yes | yes |
| MultiOutputRegressor | yes | yes |
| RegressorChain | yes | yes |

## Dummy Models

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| DummyClassifier | yes | yes |
| DummyRegressor | yes | yes |

## Inspection / Explainability

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| PermutationImportance | no | yes |
| PartialDependence | no | yes |

## Extras (non-sklearn)

| algorithm | sklearn | ingenuityml |
| --- | --- | --- |
| XGBClassifier | no | yes |
| XGBRegressor | no | yes |
| CatBoostClassifier | no | yes |
| CatBoostRegressor | no | yes |
