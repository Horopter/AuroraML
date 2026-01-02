# AuroraML vs scikit-learn: Algorithm Matrix

Ground truth is based on the current AuroraML codebase (Python bindings + C++ headers).
Use this matrix as the single source of truth for coverage; other docs link here.

Legend:
- `yes`: available in AuroraML Python API
- `C++ only`: implemented in C++ but not exposed in Python
- `partial`: present but not fully equivalent to scikit-learn
- `no`: not implemented in AuroraML

Notes:
- `SVC` and `SVR` are currently linear-only in AuroraML.
- `ColumnTransformer` is dense-only (sparse output not supported).
- `LDA` in AuroraML is a transformer; scikit-learn exposes LDA as a classifier (LinearDiscriminantAnalysis).
- `PermutationImportance` and `PartialDependence` are AuroraML classes; scikit-learn exposes similar functionality via functions/APIs.

<!-- SUMMARY_STATS_START -->
## Summary Statistics

Overall (all rows in this matrix):
- Total algorithms tracked: 208
- Implemented in AuroraML (any form): 173
- Implemented in AuroraML Python API: 163
- Partially implemented: 2
- Implemented in C++ only: 8
- Not implemented: 35

Scikit-learn coverage (rows where `sklearn` = `yes`):
- Total scikit-learn algorithms tracked: 201
- Implemented in AuroraML (any form): 166
- Implemented in AuroraML Python API: 156
- Partially implemented: 2
- Implemented in C++ only: 8
- Missing in AuroraML: 35
- Remaining for AuroraML Python parity: 45

Per-section scikit-learn coverage:

| section | sklearn total | auroraml yes | partial | C++ only | missing |
| --- | --- | --- | --- | --- | --- |
| Linear Models | 36 | 30 | 0 | 6 | 0 |
| Support Vector Machines | 7 | 5 | 2 | 0 | 0 |
| Neighbors | 8 | 8 | 0 | 0 | 0 |
| Decision Trees | 4 | 4 | 0 | 0 | 0 |
| Ensemble & Boosting | 17 | 12 | 0 | 0 | 5 |
| Naive Bayes | 5 | 4 | 0 | 0 | 1 |
| Discriminant Analysis | 2 | 1 | 0 | 0 | 1 |
| Neural Networks | 3 | 2 | 0 | 0 | 1 |
| Clustering | 13 | 8 | 0 | 0 | 5 |
| Mixture Models | 2 | 1 | 0 | 0 | 1 |
| Decomposition & Dimensionality Reduction | 13 | 13 | 0 | 0 | 0 |
| Manifold Learning | 5 | 0 | 0 | 1 | 4 |
| Cross-Decomposition | 4 | 0 | 0 | 0 | 4 |
| Random Projection | 2 | 0 | 0 | 0 | 2 |
| Gaussian Processes | 2 | 0 | 0 | 0 | 2 |
| Covariance Estimation | 6 | 6 | 0 | 0 | 0 |
| Density Estimation | 1 | 0 | 0 | 0 | 1 |
| Outlier Detection | 3 | 3 | 0 | 0 | 0 |
| Semi-Supervised Learning | 3 | 2 | 0 | 0 | 1 |
| Preprocessing & Feature Engineering | 17 | 10 | 0 | 0 | 7 |
| Imputation | 4 | 4 | 0 | 0 | 0 |
| Feature Selection | 11 | 11 | 0 | 0 | 0 |
| Model Selection (CV splitters & search) | 18 | 17 | 0 | 1 | 0 |
| Pipeline & Composition | 4 | 4 | 0 | 0 | 0 |
| Calibration & Isotonic | 2 | 2 | 0 | 0 | 0 |
| Meta-Estimators (Multiclass/Multioutput) | 7 | 7 | 0 | 0 | 0 |
| Dummy Models | 2 | 2 | 0 | 0 | 0 |
<!-- SUMMARY_STATS_END -->

## Linear Models

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| LinearRegression | yes | yes |
| Ridge | yes | yes |
| RidgeCV | yes | C++ only |
| Lasso | yes | yes |
| LassoCV | yes | C++ only |
| ElasticNet | yes | yes |
| ElasticNetCV | yes | C++ only |
| Lars | yes | yes |
| LarsCV | yes | yes |
| LassoLars | yes | yes |
| LassoLarsCV | yes | yes |
| LassoLarsIC | yes | yes |
| OrthogonalMatchingPursuit | yes | yes |
| OrthogonalMatchingPursuitCV | yes | yes |
| BayesianRidge | yes | C++ only |
| ARDRegression | yes | C++ only |
| HuberRegressor | yes | C++ only |
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

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| SVC | yes | partial (linear-only) |
| NuSVC | yes | yes |
| LinearSVC | yes | yes |
| SVR | yes | partial (linear-only) |
| NuSVR | yes | yes |
| LinearSVR | yes | yes |
| OneClassSVM | yes | yes |

## Neighbors

| algorithm | sklearn | auroraml |
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

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| DecisionTreeClassifier | yes | yes |
| DecisionTreeRegressor | yes | yes |
| ExtraTreeClassifier | yes | yes |
| ExtraTreeRegressor | yes | yes |

## Ensemble & Boosting

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| RandomForestClassifier | yes | yes |
| RandomForestRegressor | yes | yes |
| ExtraTreesClassifier | yes | no |
| ExtraTreesRegressor | yes | no |
| BaggingClassifier | yes | yes |
| BaggingRegressor | yes | yes |
| AdaBoostClassifier | yes | yes |
| AdaBoostRegressor | yes | yes |
| GradientBoostingClassifier | yes | yes |
| GradientBoostingRegressor | yes | yes |
| HistGradientBoostingClassifier | yes | no |
| HistGradientBoostingRegressor | yes | no |
| VotingClassifier | yes | yes |
| VotingRegressor | yes | yes |
| StackingClassifier | yes | yes |
| StackingRegressor | yes | yes |
| RandomTreesEmbedding | yes | no |

## Naive Bayes

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| GaussianNB | yes | yes |
| MultinomialNB | yes | yes |
| BernoulliNB | yes | yes |
| ComplementNB | yes | yes |
| CategoricalNB | yes | no |

## Discriminant Analysis

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| LinearDiscriminantAnalysis | yes | no |
| QuadraticDiscriminantAnalysis | yes | yes |

## Neural Networks

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| MLPClassifier | yes | yes |
| MLPRegressor | yes | yes |
| BernoulliRBM | yes | no |

## Clustering

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| KMeans | yes | yes |
| MiniBatchKMeans | yes | yes |
| BisectingKMeans | yes | no |
| AffinityPropagation | yes | no |
| MeanShift | yes | yes |
| SpectralClustering | yes | yes |
| AgglomerativeClustering | yes | yes |
| DBSCAN | yes | yes |
| OPTICS | yes | yes |
| Birch | yes | yes |
| FeatureAgglomeration | yes | no |
| SpectralBiclustering | yes | no |
| SpectralCoclustering | yes | no |

## Mixture Models

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| GaussianMixture | yes | yes |
| BayesianGaussianMixture | yes | no |

## Decomposition & Dimensionality Reduction

| algorithm | sklearn | auroraml |
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

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| TSNE | yes | C++ only |
| MDS | yes | no |
| Isomap | yes | no |
| LocallyLinearEmbedding | yes | no |
| SpectralEmbedding | yes | no |

## Cross-Decomposition

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| PLSCanonical | yes | no |
| PLSRegression | yes | no |
| CCA | yes | no |
| PLSSVD | yes | no |

## Random Projection

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| GaussianRandomProjection | yes | no |
| SparseRandomProjection | yes | no |

## Gaussian Processes

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| GaussianProcessClassifier | yes | no |
| GaussianProcessRegressor | yes | no |

## Covariance Estimation

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| EmpiricalCovariance | yes | yes |
| ShrunkCovariance | yes | yes |
| LedoitWolf | yes | yes |
| OAS | yes | yes |
| MinCovDet | yes | yes |
| EllipticEnvelope | yes | yes |

## Density Estimation

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| KernelDensity | yes | no |

## Outlier Detection

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| IsolationForest | yes | yes |
| LocalOutlierFactor | yes | yes |
| EllipticEnvelope | yes | yes |

## Semi-Supervised Learning

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| LabelPropagation | yes | yes |
| LabelSpreading | yes | yes |
| SelfTrainingClassifier | yes | no |

## Preprocessing & Feature Engineering

| algorithm | sklearn | auroraml |
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
| LabelBinarizer | yes | no |
| MultiLabelBinarizer | yes | no |
| KBinsDiscretizer | yes | no |
| QuantileTransformer | yes | no |
| PowerTransformer | yes | no |
| FunctionTransformer | yes | no |
| SplineTransformer | yes | no |

## Imputation

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| SimpleImputer | yes | yes |
| KNNImputer | yes | yes |
| IterativeImputer | yes | yes |
| MissingIndicator | yes | yes |

## Feature Selection

| algorithm | sklearn | auroraml |
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

| algorithm | sklearn | auroraml |
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
| HalvingGridSearchCV | yes | C++ only |
| HalvingRandomSearchCV | yes | yes |
| ParameterGrid | yes | yes |
| ParameterSampler | yes | yes |

## Pipeline & Composition

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| Pipeline | yes | yes |
| FeatureUnion | yes | yes |
| ColumnTransformer | yes | yes |
| TransformedTargetRegressor | yes | yes |

## Calibration & Isotonic

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| CalibratedClassifierCV | yes | yes |
| IsotonicRegression | yes | yes |

## Meta-Estimators (Multiclass/Multioutput)

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| OneVsRestClassifier | yes | yes |
| OneVsOneClassifier | yes | yes |
| OutputCodeClassifier | yes | yes |
| ClassifierChain | yes | yes |
| MultiOutputClassifier | yes | yes |
| MultiOutputRegressor | yes | yes |
| RegressorChain | yes | yes |

## Dummy Models

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| DummyClassifier | yes | yes |
| DummyRegressor | yes | yes |

## Inspection / Explainability

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| PermutationImportance | no | yes |
| PartialDependence | no | yes |

## Extras (non-sklearn)

| algorithm | sklearn | auroraml |
| --- | --- | --- |
| XGBClassifier | no | yes |
| XGBRegressor | no | yes |
| CatBoostClassifier | no | yes |
| CatBoostRegressor | no | yes |
