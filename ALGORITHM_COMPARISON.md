# AuroraML vs Scikit-learn: Algorithm Comparison

## Overview
This document provides a comprehensive comparison of algorithms available in scikit-learn vs those implemented in AuroraML.

---

## 📊 Classification Algorithms

### ✅ Implemented in AuroraML
- **DecisionTreeClassifier** - CART algorithm with Gini/Entropy
- **RandomForestClassifier** - Ensemble of decision trees
- **GradientBoostingClassifier** - Gradient boosting
- **AdaBoostClassifier** - Adaptive boosting
- **XGBClassifier** - XGBoost classifier
- **CatBoostClassifier** - CatBoost classifier
- **KNeighborsClassifier** - K-Nearest Neighbors
- **GaussianNB** - Gaussian Naive Bayes
- **LinearSVC** - Linear Support Vector Classifier
- **LogisticRegression** - Logistic regression (binary only)

### ❌ Missing from AuroraML
- **SVC** - Support Vector Classifier (RBF/poly/sigmoid kernels)
- **NuSVC** - Nu-Support Vector Classifier
- **SGDClassifier** - Stochastic Gradient Descent classifier
- **Perceptron** - Perceptron
- **PassiveAggressiveClassifier** - Passive-Aggressive classifier
- **RidgeClassifier** - Ridge classifier
- **RidgeClassifierCV** - Ridge classifier with cross-validation
- **LogisticRegressionCV** - Logistic regression with cross-validation
- **MLPClassifier** - Multi-layer Perceptron (Neural Network)
- **BernoulliNB** - Bernoulli Naive Bayes
- **MultinomialNB** - Multinomial Naive Bayes
- **ComplementNB** - Complement Naive Bayes
- **CategoricalNB** - Categorical Naive Bayes
- **ExtraTreeClassifier** - Extremely Randomized Tree
- **ExtraTreesClassifier** - Extremely Randomized Trees (ensemble)
- **HistGradientBoostingClassifier** - Histogram-based gradient boosting
- **NearestCentroid** - Nearest centroid classifier
- **RadiusNeighborsClassifier** - Radius-based neighbors classifier
- **LinearDiscriminantAnalysis** (LDA) - Available as transformer, not classifier
- **QuadraticDiscriminantAnalysis** (QDA)
- **GaussianProcessClassifier** - Gaussian Process classifier
- **CalibratedClassifierCV** - Probability calibration
- **DummyClassifier** - Dummy classifier for baseline
- **LabelPropagation** - Semi-supervised learning
- **LabelSpreading** - Semi-supervised learning
- **OneVsRestClassifier** - One-vs-Rest multiclass strategy
- **OneVsOneClassifier** - One-vs-One multiclass strategy
- **OutputCodeClassifier** - Error-correcting output codes
- **ClassifierChain** - Classifier chains for multi-label
- **MultiOutputClassifier** - Multi-output classifier wrapper
- **BaggingClassifier** - Bagging ensemble
- **StackingClassifier** - Stacking ensemble
- **VotingClassifier** - Voting ensemble

---

## 📈 Regression Algorithms

### ✅ Implemented in AuroraML
- **LinearRegression** - Ordinary Least Squares
- **Ridge** - Ridge regression (L2)
- **Lasso** - Lasso regression (L1)
- **ElasticNet** - Elastic Net (L1 + L2)
- **DecisionTreeRegressor** - CART algorithm with MSE
- **RandomForestRegressor** - Ensemble of decision trees
- **GradientBoostingRegressor** - Gradient boosting
- **AdaBoostRegressor** - Adaptive boosting
- **XGBRegressor** - XGBoost regressor
- **CatBoostRegressor** - CatBoost regressor
- **KNeighborsRegressor** - K-Nearest Neighbors
- **SVR** - Support Vector Regression

### ❌ Missing from AuroraML
- **SGDRegressor** - Stochastic Gradient Descent regressor
- **RANSACRegressor** - RANSAC robust regression
- **TheilSenRegressor** - Theil-Sen robust regression
- **HuberRegressor** - Huber robust regression
- **BayesianRidge** - Bayesian ridge regression
- **ARDRegression** - Automatic Relevance Determination
- **Lars** - Least Angle Regression
- **LassoLars** - Lasso with LARS
- **OrthogonalMatchingPursuit** - Orthogonal Matching Pursuit
- **PassiveAggressiveRegressor** - Passive-Aggressive regressor
- **KernelRidge** - Kernel ridge regression
- **GaussianProcessRegressor** - Gaussian Process regressor
- **MLPRegressor** - Multi-layer Perceptron (Neural Network)
- **ExtraTreeRegressor** - Extremely Randomized Tree
- **ExtraTreesRegressor** - Extremely Randomized Trees (ensemble)
- **HistGradientBoostingRegressor** - Histogram-based gradient boosting
- **RadiusNeighborsRegressor** - Radius-based neighbors regressor
- **BaggingRegressor** - Bagging ensemble
- **StackingRegressor** - Stacking ensemble
- **VotingRegressor** - Voting ensemble

---

## 🎯 Clustering Algorithms

### ✅ Implemented in AuroraML
- **KMeans** - K-Means clustering
- **DBSCAN** - Density-based clustering
- **AgglomerativeClustering** - Hierarchical clustering

### ❌ Missing from AuroraML
- **AffinityPropagation** - Affinity propagation clustering
- **MeanShift** - Mean shift clustering
- **SpectralClustering** - Spectral clustering
- **OPTICS** - OPTICS clustering
- **Birch** - BIRCH hierarchical clustering
- **MiniBatchKMeans** - Mini-batch K-Means
- **GaussianMixture** - Gaussian Mixture Models (from mixture module)
- **BayesianGaussianMixture** - Bayesian Gaussian Mixture

---

## 🔄 Preprocessing & Feature Engineering

### ✅ Implemented in AuroraML
- **StandardScaler** - Standardization (zero mean, unit variance)
- **MinMaxScaler** - Min-max scaling
- **RobustScaler** - Robust scaling (median/IQR)
- **Normalizer** - Sample-wise normalization
- **LabelEncoder** - Label encoding
- **OneHotEncoder** - One-hot encoding
- **OrdinalEncoder** - Ordinal encoding
- **PolynomialFeatures** - Polynomial features
- **SimpleImputer** - Missing value imputation

### ❌ Missing from AuroraML
- **MaxAbsScaler** - Max absolute value scaling
- **QuantileTransformer** - Quantile transformation
- **PowerTransformer** - Power transformation (Box-Cox, Yeo-Johnson)
- **RobustScaler** - Full implementation (partial)
- **TargetEncoder** - Target encoding
- **KBinsDiscretizer** - Binning discretizer
- **Binarizer** - Feature binarization
- **FunctionTransformer** - Custom function transformer
- **KernelCenterer** - Kernel centerer
- **AdditiveChi2Sampler** - Additive Chi-squared kernel sampler
- **Nystroem** - Nystroem kernel approximation
- **PolynomialCountSketch** - Polynomial count sketch
- **RBFSampler** - RBF kernel sampler
- **SkewedChi2Sampler** - Skewed Chi-squared kernel sampler
- **SplineTransformer** - Spline transformer
- **StandardScaler** - Full implementation (partial - missing with_std option)
- **FeatureHasher** - Feature hashing
- **DictVectorizer** - Dictionary vectorizer (from feature_extraction)
- **CountVectorizer** - Count vectorizer (from feature_extraction)
- **TfidfVectorizer** - TF-IDF vectorizer (from feature_extraction)
- **HashingVectorizer** - Hashing vectorizer (from feature_extraction)

---

## 📉 Dimensionality Reduction

### ✅ Implemented in AuroraML
- **PCA** - Principal Component Analysis
- **TruncatedSVD** - Truncated SVD
- **LDA** - Linear Discriminant Analysis (as transformer)

### ❌ Missing from AuroraML
- **KernelPCA** - Kernel PCA
- **IncrementalPCA** - Incremental PCA
- **SparsePCA** - Sparse PCA
- **MiniBatchSparsePCA** - Mini-batch sparse PCA
- **FactorAnalysis** - Factor Analysis
- **FastICA** - Fast Independent Component Analysis
- **DictionaryLearning** - Dictionary learning
- **MiniBatchDictionaryLearning** - Mini-batch dictionary learning
- **LatentDirichletAllocation** - LDA topic modeling (different from Linear Discriminant Analysis)
- **NMF** - Non-negative Matrix Factorization
- **MiniBatchNMF** - Mini-batch NMF
- **t-SNE** - t-distributed Stochastic Neighbor Embedding
- **MDS** - Multidimensional Scaling
- **Isomap** - Isomap
- **LocallyLinearEmbedding** - Locally Linear Embedding
- **SpectralEmbedding** - Spectral embedding

---

## 🎲 Model Selection & Evaluation

### ✅ Implemented in AuroraML
- **train_test_split** - Train/test split
- **KFold** - K-fold cross-validation
- **StratifiedKFold** - Stratified K-fold
- **GroupKFold** - Group K-fold
- **GridSearchCV** - Grid search
- **RandomizedSearchCV** - Randomized search
- **cross_val_score** - Cross-validation scoring

### ❌ Missing from AuroraML
- **RepeatedKFold** - Repeated K-fold
- **RepeatedStratifiedKFold** - Repeated stratified K-fold
- **LeaveOneOut** - Leave-one-out CV
- **LeavePOut** - Leave-p-out CV
- **ShuffleSplit** - Shuffle split
- **StratifiedShuffleSplit** - Stratified shuffle split
- **GroupShuffleSplit** - Group shuffle split
- **TimeSeriesSplit** - Time series cross-validation
- **PredefinedSplit** - Predefined split
- **ParameterGrid** - Parameter grid generator
- **ParameterSampler** - Parameter sampler
- **learning_curve** - Learning curve
- **validation_curve** - Validation curve
- **permutation_test_score** - Permutation test score
- **check_cv** - CV checker
- **check_scoring** - Scoring checker
- **get_scorer** - Get scorer
- **make_scorer** - Custom scorer

---

## 📊 Metrics

### ✅ Implemented in AuroraML
- **accuracy_score** - Classification accuracy
- **precision_score** - Precision (macro/weighted)
- **recall_score** - Recall (macro/weighted)
- **f1_score** - F1 score (macro/weighted)
- **confusion_matrix** - Confusion matrix
- **classification_report** - Classification report
- **mean_squared_error** - MSE
- **root_mean_squared_error** - RMSE
- **mean_absolute_error** - MAE
- **r2_score** - R² score
- **explained_variance_score** - Explained variance
- **mean_absolute_percentage_error** - MAPE

### ❌ Missing from AuroraML
- **balanced_accuracy_score** - Balanced accuracy
- **top_k_accuracy_score** - Top-k accuracy
- **roc_auc_score** - ROC AUC score
- **average_precision_score** - Average precision
- **log_loss** - Log loss
- **hinge_loss** - Hinge loss
- **cohen_kappa_score** - Cohen's kappa
- **matthews_corrcoef** - Matthews correlation coefficient
- **hamming_loss** - Hamming loss
- **jaccard_score** - Jaccard score
- **zero_one_loss** - Zero-one loss
- **brier_score_loss** - Brier score
- **median_absolute_error** - Median absolute error
- **max_error** - Max error
- **mean_poisson_deviance** - Mean Poisson deviance
- **mean_gamma_deviance** - Mean Gamma deviance
- **mean_tweedie_deviance** - Mean Tweedie deviance
- **d2_tweedie_score** - D² Tweedie score
- **d2_pinball_score** - D² Pinball score
- **d2_absolute_error_score** - D² Absolute error score
- **silhouette_score** - Silhouette score (clustering)
- **silhouette_samples** - Silhouette samples
- **calinski_harabasz_score** - Calinski-Harabasz index
- **davies_bouldin_score** - Davies-Bouldin index
- **adjusted_rand_score** - Adjusted Rand index
- **adjusted_mutual_info_score** - Adjusted mutual information
- **normalized_mutual_info_score** - Normalized mutual information
- **homogeneity_score** - Homogeneity score
- **completeness_score** - Completeness score
- **v_measure_score** - V-measure score
- **fowlkes_mallows_score** - Fowlkes-Mallows index

---

## 🔧 Utilities & Other Modules

### ✅ Implemented in AuroraML
- **PCG64** - Random number generator
- Model persistence (save/load) for most algorithms

### ❌ Missing from AuroraML
- **calibration** module - Probability calibration (CalibratedClassifierCV)
- **feature_selection** module - Feature selection methods
  - SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe
  - GenericUnivariateSelect, SelectFromModel, RFE, RFECV
  - VarianceThreshold, mutual_info_classif, chi2, f_classif, etc.
- **impute** module - Advanced imputation (KNNImputer, IterativeImputer)
- **pipeline** module - Pipeline, FeatureUnion
- **compose** module - ColumnTransformer, TransformedTargetRegressor
- **inspection** module - Permutation importance, partial dependence
- **manifold** module - Manifold learning (t-SNE, Isomap, etc.)
- **mixture** module - Gaussian Mixture Models
- **semi_supervised** module - Semi-supervised learning
- **isotonic** module - Isotonic regression
- **discriminant_analysis** module - LDA, QDA (LDA exists as transformer)
- **gaussian_process** module - Gaussian Process models
- **neural_network** module - MLPClassifier, MLPRegressor
- **kernel_approximation** module - Kernel approximation methods
- **kernel_ridge** module - Kernel ridge regression
- **naive_bayes** module - Additional NB variants (Multinomial, Bernoulli, etc.)
- **tree** module - Additional tree variants (ExtraTree)
- **ensemble** module - Additional ensemble methods (Bagging, Stacking, Voting)
- **cluster** module - Additional clustering methods (AffinityPropagation, MeanShift, etc.)
- **covariance** module - Covariance estimation
- **outlier_detection** module - Isolation Forest, Local Outlier Factor, One-Class SVM
- **utils** module - Various utilities (multiclass, resample, shuffle, etc.)
- **datasets** module - Sample datasets

---

## 📝 Summary Statistics

### Classification
- **AuroraML**: 10 algorithms
- **Scikit-learn**: ~50+ algorithms
- **Coverage**: ~20%

### Regression
- **AuroraML**: 11 algorithms
- **Scikit-learn**: ~30+ algorithms
- **Coverage**: ~37%

### Clustering
- **AuroraML**: 3 algorithms
- **Scikit-learn**: ~10+ algorithms
- **Coverage**: ~30%

### Preprocessing
- **AuroraML**: 9 transformers
- **Scikit-learn**: ~30+ transformers
- **Coverage**: ~30%

### Dimensionality Reduction
- **AuroraML**: 3 methods
- **Scikit-learn**: ~20+ methods
- **Coverage**: ~15%

### Model Selection
- **AuroraML**: 7 methods
- **Scikit-learn**: ~20+ methods
- **Coverage**: ~35%

### Metrics
- **AuroraML**: 12 metrics
- **Scikit-learn**: ~60+ metrics
- **Coverage**: ~20%

---

## 🎯 Priority Recommendations for AuroraML

### High Priority (Core ML Algorithms)
1. **SVC** - Support Vector Classifier with kernels (RBF, polynomial, sigmoid)
2. **SGDClassifier/SGDRegressor** - Stochastic Gradient Descent (very common)
3. **MLPClassifier/MLPRegressor** - Neural networks (multi-layer perceptron)
4. **MultinomialNB/BernoulliNB** - Additional Naive Bayes variants
5. **BaggingClassifier/BaggingRegressor** - Bagging ensemble
6. **StackingClassifier/StackingRegressor** - Stacking ensemble
7. **VotingClassifier/VotingRegressor** - Voting ensemble

### Medium Priority (Useful Features)
8. **Feature Selection** - SelectKBest, RFE, SelectFromModel
9. **Pipeline** - Pipeline for chaining transformers
10. **ColumnTransformer** - Feature-specific transformations
11. **More clustering** - SpectralClustering, OPTICS, AffinityPropagation
12. **More metrics** - ROC-AUC, balanced accuracy, silhouette score
13. **Outlier detection** - IsolationForest, LocalOutlierFactor
14. **Imputation** - KNNImputer, IterativeImputer

### Lower Priority (Niche Use Cases)
15. **Gaussian Process** models
16. **Manifold learning** (t-SNE, Isomap, etc.)
17. **Semi-supervised learning**
18. **Topic modeling** (LatentDirichletAllocation)
19. **Dictionary learning**
20. **Advanced robust regression** (RANSAC, TheilSen, Huber)

---

## 📅 Last Updated
January 2025

