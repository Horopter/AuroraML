# Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Find GoogleTest
find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Test executables
add_executable(test_preprocessing preprocessing/test_preprocessing.cpp)
target_link_libraries(test_preprocessing auroraml_core GTest::GTest GTest::Main)

add_executable(test_naive_bayes naive_bayes/test_naive_bayes.cpp)
target_link_libraries(test_naive_bayes auroraml_core GTest::GTest GTest::Main)

add_executable(test_kmeans cluster/test_kmeans.cpp)
target_link_libraries(test_kmeans auroraml_core GTest::GTest GTest::Main)

add_executable(test_pca decomposition/test_pca.cpp)
target_link_libraries(test_pca auroraml_core GTest::GTest GTest::Main)

add_executable(test_svm svm/test_svm.cpp)
target_link_libraries(test_svm auroraml_core GTest::GTest GTest::Main)

add_executable(test_decomposition decomposition/test_decomposition.cpp)
target_link_libraries(test_decomposition auroraml_core GTest::GTest GTest::Main)

add_executable(test_metrics metrics/test_metrics.cpp)
target_link_libraries(test_metrics auroraml_core GTest::GTest GTest::Main)

add_executable(test_model_selection model_selection/test_model_selection.cpp)
target_link_libraries(test_model_selection auroraml_core GTest::GTest GTest::Main)

add_executable(test_lda discriminant_analysis/test_lda.cpp)
target_link_libraries(test_lda auroraml_core GTest::GTest GTest::Main)

add_executable(test_adaboost_classifier ensemble/test_adaboost_classifier.cpp)
target_link_libraries(test_adaboost_classifier auroraml_core GTest::GTest GTest::Main)

add_executable(test_adaboost_regressor ensemble/test_adaboost_regressor.cpp)
target_link_libraries(test_adaboost_regressor auroraml_core GTest::GTest GTest::Main)

add_executable(test_xgboost ensemble/test_xgboost.cpp)
target_link_libraries(test_xgboost auroraml_core GTest::GTest GTest::Main)

add_executable(test_catboost ensemble/test_catboost.cpp)
target_link_libraries(test_catboost auroraml_core GTest::GTest GTest::Main)

add_executable(test_decision_tree_classifier tree/test_decision_tree_classifier.cpp)
target_link_libraries(test_decision_tree_classifier auroraml_core GTest::GTest GTest::Main)

add_executable(test_decision_tree_regressor tree/test_decision_tree_regressor.cpp)
target_link_libraries(test_decision_tree_regressor auroraml_core GTest::GTest GTest::Main)

add_executable(test_random_forest_classifier ensemble/test_random_forest_classifier.cpp)
target_link_libraries(test_random_forest_classifier auroraml_core GTest::GTest GTest::Main)

add_executable(test_random_forest_regressor ensemble/test_random_forest_regressor.cpp)
target_link_libraries(test_random_forest_regressor auroraml_core GTest::GTest GTest::Main)

add_executable(test_kneighbors_classifier neighbors/test_kneighbors_classifier.cpp)
target_link_libraries(test_kneighbors_classifier auroraml_core GTest::GTest GTest::Main)

add_executable(test_kneighbors_regressor neighbors/test_kneighbors_regressor.cpp)
target_link_libraries(test_kneighbors_regressor auroraml_core GTest::GTest GTest::Main)

add_executable(test_nearest_neighbors neighbors/test_nearest_neighbors.cpp)
target_link_libraries(test_nearest_neighbors auroraml_core GTest::GTest GTest::Main)

add_executable(test_neighbors_transformer neighbors/test_neighbors_transformer.cpp)
target_link_libraries(test_neighbors_transformer auroraml_core GTest::GTest GTest::Main)

add_executable(test_logistic_regression linear_model/test_logistic_regression.cpp)
target_link_libraries(test_logistic_regression auroraml_core GTest::GTest GTest::Main)

add_executable(test_elasticnet linear_model/test_elasticnet.cpp)
target_link_libraries(test_elasticnet auroraml_core GTest::GTest GTest::Main)

add_executable(test_dbscan cluster/test_dbscan.cpp)
target_link_libraries(test_dbscan auroraml_core GTest::GTest GTest::Main)

add_executable(test_agglomerative_clustering cluster/test_agglomerative_clustering.cpp)
target_link_libraries(test_agglomerative_clustering auroraml_core GTest::GTest GTest::Main)

add_executable(test_linear_regression linear_model/test_linear_regression.cpp)
target_link_libraries(test_linear_regression auroraml_core GTest::GTest GTest::Main)

add_executable(test_ridge linear_model/test_ridge.cpp)
target_link_libraries(test_ridge auroraml_core GTest::GTest GTest::Main)

add_executable(test_lasso linear_model/test_lasso.cpp)
target_link_libraries(test_lasso auroraml_core GTest::GTest GTest::Main)

add_executable(test_svr svm/test_svr.cpp)
target_link_libraries(test_svr auroraml_core GTest::GTest GTest::Main)

add_executable(test_truncated_svd decomposition/test_truncated_svd.cpp)
target_link_libraries(test_truncated_svd auroraml_core GTest::GTest GTest::Main)

add_executable(test_decomposition_extended decomposition/test_decomposition_extended.cpp)
target_link_libraries(test_decomposition_extended auroraml_core GTest::GTest GTest::Main)

add_executable(test_gradient_boosting_classifier ensemble/test_gradient_boosting_classifier.cpp)
target_link_libraries(test_gradient_boosting_classifier auroraml_core GTest::GTest GTest::Main)

add_executable(test_gradient_boosting_regressor ensemble/test_gradient_boosting_regressor.cpp)
target_link_libraries(test_gradient_boosting_regressor auroraml_core GTest::GTest GTest::Main)

add_executable(test_calibration calibration/test_calibration.cpp)
target_link_libraries(test_calibration auroraml_core GTest::GTest GTest::Main)

add_executable(test_compose pipeline/test_compose.cpp)
target_link_libraries(test_compose auroraml_core GTest::GTest GTest::Main)

add_executable(test_discriminant_analysis discriminant_analysis/test_discriminant_analysis.cpp)
target_link_libraries(test_discriminant_analysis auroraml_core GTest::GTest GTest::Main)

add_executable(test_dummy_classifier dummy/test_dummy_classifier.cpp)
target_link_libraries(test_dummy_classifier auroraml_core GTest::GTest GTest::Main)

add_executable(test_dummy_regressor dummy/test_dummy_regressor.cpp)
target_link_libraries(test_dummy_regressor auroraml_core GTest::GTest GTest::Main)

add_executable(test_covariance covariance/test_covariance.cpp)
target_link_libraries(test_covariance auroraml_core GTest::GTest GTest::Main)

add_executable(test_meta_estimators meta/test_meta_estimators.cpp)
target_link_libraries(test_meta_estimators auroraml_core GTest::GTest GTest::Main)
add_executable(test_feature_selection preprocessing/test_feature_selection.cpp)
target_link_libraries(test_feature_selection auroraml_core GTest::GTest GTest::Main)

add_executable(test_impute preprocessing/test_impute.cpp)
target_link_libraries(test_impute auroraml_core GTest::GTest GTest::Main)

add_executable(test_inspection inspection/test_inspection.cpp)
target_link_libraries(test_inspection auroraml_core GTest::GTest GTest::Main)

add_executable(test_isotonic calibration/test_isotonic.cpp)
target_link_libraries(test_isotonic auroraml_core GTest::GTest GTest::Main)

add_executable(test_logistic_regression_cv linear_model/test_logistic_regression_cv.cpp)
target_link_libraries(test_logistic_regression_cv auroraml_core GTest::GTest GTest::Main)

add_executable(test_linear_model_sparse linear_model/test_linear_model_sparse.cpp)
target_link_libraries(test_linear_model_sparse auroraml_core GTest::GTest GTest::Main)

add_executable(test_linear_model_robust linear_model/test_linear_model_robust.cpp)
target_link_libraries(test_linear_model_robust auroraml_core GTest::GTest GTest::Main)

add_executable(test_linear_model_online linear_model/test_linear_model_online.cpp)
target_link_libraries(test_linear_model_online auroraml_core GTest::GTest GTest::Main)

add_executable(test_linear_model_glm linear_model/test_linear_model_glm.cpp)
target_link_libraries(test_linear_model_glm auroraml_core GTest::GTest GTest::Main)

add_executable(test_linear_model_classifiers_cv linear_model/test_linear_model_classifiers_cv.cpp)
target_link_libraries(test_linear_model_classifiers_cv auroraml_core GTest::GTest GTest::Main)

add_executable(test_linear_model_multitask linear_model/test_linear_model_multitask.cpp)
target_link_libraries(test_linear_model_multitask auroraml_core GTest::GTest GTest::Main)

add_executable(test_mixture mixture/test_mixture.cpp)
target_link_libraries(test_mixture auroraml_core GTest::GTest GTest::Main)

add_executable(test_naive_bayes_variants naive_bayes/test_naive_bayes_variants.cpp)
target_link_libraries(test_naive_bayes_variants auroraml_core GTest::GTest GTest::Main)

add_executable(test_nearest_centroid neighbors/test_nearest_centroid.cpp)
target_link_libraries(test_nearest_centroid auroraml_core GTest::GTest GTest::Main)

add_executable(test_outlier_detection outlier/test_outlier_detection.cpp)
target_link_libraries(test_outlier_detection auroraml_core GTest::GTest GTest::Main)

add_executable(test_pipeline pipeline/test_pipeline.cpp)
target_link_libraries(test_pipeline auroraml_core GTest::GTest GTest::Main)

add_executable(test_preprocessing_extended preprocessing/test_preprocessing_extended.cpp)
target_link_libraries(test_preprocessing_extended auroraml_core GTest::GTest GTest::Main)

add_executable(test_radiusneighborsclassifier neighbors/test_radiusneighborsclassifier.cpp)
target_link_libraries(test_radiusneighborsclassifier auroraml_core GTest::GTest GTest::Main)

add_executable(test_radiusneighborsregressor neighbors/test_radiusneighborsregressor.cpp)
target_link_libraries(test_radiusneighborsregressor auroraml_core GTest::GTest GTest::Main)

add_executable(test_random utils/test_random.cpp)
target_link_libraries(test_random auroraml_core GTest::GTest GTest::Main)

add_executable(test_semi_supervised semi_supervised/test_semi_supervised.cpp)
target_link_libraries(test_semi_supervised auroraml_core GTest::GTest GTest::Main)

add_executable(test_utils utils/test_utils.cpp)
target_link_libraries(test_utils auroraml_core GTest::GTest GTest::Main)

# Enable testing
enable_testing()

# Add tests with 5-minute (300 second) timeout
set(TEST_TIMEOUT 300)

add_test(NAME PreprocessingTest COMMAND test_preprocessing)
set_tests_properties(PreprocessingTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME NaiveBayesTest COMMAND test_naive_bayes)
set_tests_properties(NaiveBayesTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME KMeansTest COMMAND test_kmeans)
set_tests_properties(KMeansTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME PCATest COMMAND test_pca)
set_tests_properties(PCATest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME SVMTest COMMAND test_svm)
set_tests_properties(SVMTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME DecompositionTest COMMAND test_decomposition)
set_tests_properties(DecompositionTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME MetricsTest COMMAND test_metrics)
set_tests_properties(MetricsTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME ModelSelectionTest COMMAND test_model_selection)
set_tests_properties(ModelSelectionTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LDATest COMMAND test_lda)
set_tests_properties(LDATest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME AdaBoostClassifierTest COMMAND test_adaboost_classifier)
set_tests_properties(AdaBoostClassifierTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME AdaBoostRegressorTest COMMAND test_adaboost_regressor)
set_tests_properties(AdaBoostRegressorTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME XGBoostTest COMMAND test_xgboost)
set_tests_properties(XGBoostTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME CatBoostTest COMMAND test_catboost)
set_tests_properties(CatBoostTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME DecisionTreeClassifierTest COMMAND test_decision_tree_classifier)
set_tests_properties(DecisionTreeClassifierTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME DecisionTreeRegressorTest COMMAND test_decision_tree_regressor)
set_tests_properties(DecisionTreeRegressorTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME RandomForestClassifierTest COMMAND test_random_forest_classifier)
set_tests_properties(RandomForestClassifierTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME RandomForestRegressorTest COMMAND test_random_forest_regressor)
set_tests_properties(RandomForestRegressorTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME KNeighborsClassifierTest COMMAND test_kneighbors_classifier)
set_tests_properties(KNeighborsClassifierTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME KNeighborsRegressorTest COMMAND test_kneighbors_regressor)
set_tests_properties(KNeighborsRegressorTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME NearestNeighborsTest COMMAND test_nearest_neighbors)
set_tests_properties(NearestNeighborsTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME NeighborsTransformerTest COMMAND test_neighbors_transformer)
set_tests_properties(NeighborsTransformerTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LogisticRegressionTest COMMAND test_logistic_regression)
set_tests_properties(LogisticRegressionTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME ElasticNetTest COMMAND test_elasticnet)
set_tests_properties(ElasticNetTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME DBSCANTest COMMAND test_dbscan)
set_tests_properties(DBSCANTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME AgglomerativeClusteringTest COMMAND test_agglomerative_clustering)
set_tests_properties(AgglomerativeClusteringTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LinearRegressionTest COMMAND test_linear_regression)
set_tests_properties(LinearRegressionTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME RidgeTest COMMAND test_ridge)
set_tests_properties(RidgeTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LassoTest COMMAND test_lasso)
set_tests_properties(LassoTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME SVRTest COMMAND test_svr)
set_tests_properties(SVRTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME TruncatedSVDTest COMMAND test_truncated_svd)
set_tests_properties(TruncatedSVDTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME DecompositionExtendedTest COMMAND test_decomposition_extended)
set_tests_properties(DecompositionExtendedTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME GradientBoostingClassifierTest COMMAND test_gradient_boosting_classifier)
set_tests_properties(GradientBoostingClassifierTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME GradientBoostingRegressorTest COMMAND test_gradient_boosting_regressor)
set_tests_properties(GradientBoostingRegressorTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME CalibrationTest COMMAND test_calibration)
set_tests_properties(CalibrationTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME ComposeTest COMMAND test_compose)
set_tests_properties(ComposeTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME DiscriminantAnalysisTest COMMAND test_discriminant_analysis)
set_tests_properties(DiscriminantAnalysisTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME DummyClassifierTest COMMAND test_dummy_classifier)
set_tests_properties(DummyClassifierTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME DummyRegressorTest COMMAND test_dummy_regressor)
set_tests_properties(DummyRegressorTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME CovarianceTest COMMAND test_covariance)
set_tests_properties(CovarianceTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME MetaEstimatorsTest COMMAND test_meta_estimators)
set_tests_properties(MetaEstimatorsTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME FeatureSelectionTest COMMAND test_feature_selection)
set_tests_properties(FeatureSelectionTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME ImputeTest COMMAND test_impute)
set_tests_properties(ImputeTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME InspectionTest COMMAND test_inspection)
set_tests_properties(InspectionTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME IsotonicTest COMMAND test_isotonic)
set_tests_properties(IsotonicTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LogisticRegressionCVTest COMMAND test_logistic_regression_cv)
set_tests_properties(LogisticRegressionCVTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LinearModelSparseTest COMMAND test_linear_model_sparse)
set_tests_properties(LinearModelSparseTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LinearModelRobustTest COMMAND test_linear_model_robust)
set_tests_properties(LinearModelRobustTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LinearModelOnlineTest COMMAND test_linear_model_online)
set_tests_properties(LinearModelOnlineTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LinearModelGLMTest COMMAND test_linear_model_glm)
set_tests_properties(LinearModelGLMTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LinearModelClassifiersCVTest COMMAND test_linear_model_classifiers_cv)
set_tests_properties(LinearModelClassifiersCVTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME LinearModelMultiTaskTest COMMAND test_linear_model_multitask)
set_tests_properties(LinearModelMultiTaskTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME MixtureTest COMMAND test_mixture)
set_tests_properties(MixtureTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME NaiveBayesVariantsTest COMMAND test_naive_bayes_variants)
set_tests_properties(NaiveBayesVariantsTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME NearestCentroidTest COMMAND test_nearest_centroid)
set_tests_properties(NearestCentroidTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME OutlierDetectionTest COMMAND test_outlier_detection)
set_tests_properties(OutlierDetectionTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME PipelineTest COMMAND test_pipeline)
set_tests_properties(PipelineTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME PreprocessingExtendedTest COMMAND test_preprocessing_extended)
set_tests_properties(PreprocessingExtendedTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME RadiusNeighborsClassifierTest COMMAND test_radiusneighborsclassifier)
set_tests_properties(RadiusNeighborsClassifierTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME RadiusNeighborsRegressorTest COMMAND test_radiusneighborsregressor)
set_tests_properties(RadiusNeighborsRegressorTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME RandomTest COMMAND test_random)
set_tests_properties(RandomTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME SemiSupervisedTest COMMAND test_semi_supervised)
set_tests_properties(SemiSupervisedTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})

add_test(NAME UtilsTest COMMAND test_utils)
set_tests_properties(UtilsTest PROPERTIES TIMEOUT ${TEST_TIMEOUT})
