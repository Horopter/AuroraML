# Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Find GoogleTest
find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Test executables
add_executable(test_linear_models test_linear_models.cpp)
target_link_libraries(test_linear_models auroraml_core GTest::GTest GTest::Main)

add_executable(test_preprocessing test_preprocessing.cpp)
target_link_libraries(test_preprocessing auroraml_core GTest::GTest GTest::Main)

add_executable(test_neighbors test_neighbors.cpp)
target_link_libraries(test_neighbors auroraml_core GTest::GTest GTest::Main)

add_executable(test_naive_bayes test_naive_bayes.cpp)
target_link_libraries(test_naive_bayes auroraml_core GTest::GTest GTest::Main)

add_executable(test_kmeans test_kmeans.cpp)
target_link_libraries(test_kmeans auroraml_core GTest::GTest GTest::Main)

add_executable(test_pca test_pca.cpp)
target_link_libraries(test_pca auroraml_core GTest::GTest GTest::Main)

add_executable(test_tree test_tree.cpp)
target_link_libraries(test_tree auroraml_core GTest::GTest GTest::Main)

add_executable(test_svm test_svm.cpp)
target_link_libraries(test_svm auroraml_core GTest::GTest GTest::Main)

add_executable(test_ensemble test_ensemble.cpp)
target_link_libraries(test_ensemble auroraml_core GTest::GTest GTest::Main)

add_executable(test_cluster test_cluster.cpp)
target_link_libraries(test_cluster auroraml_core GTest::GTest GTest::Main)

add_executable(test_decomposition test_decomposition.cpp)
target_link_libraries(test_decomposition auroraml_core GTest::GTest GTest::Main)

add_executable(test_metrics test_metrics.cpp)
target_link_libraries(test_metrics auroraml_core GTest::GTest GTest::Main)

add_executable(test_model_selection test_model_selection.cpp)
target_link_libraries(test_model_selection auroraml_core GTest::GTest GTest::Main)

add_executable(test_lda test_lda.cpp)
target_link_libraries(test_lda auroraml_core GTest::GTest GTest::Main)

add_executable(test_gradient_boosting test_gradient_boosting.cpp)
target_link_libraries(test_gradient_boosting auroraml_core GTest::GTest GTest::Main)

# Enable testing
enable_testing()

# Add tests
add_test(NAME LinearModelsTest COMMAND test_linear_models)
add_test(NAME PreprocessingTest COMMAND test_preprocessing)
add_test(NAME NeighborsTest COMMAND test_neighbors)
add_test(NAME NaiveBayesTest COMMAND test_naive_bayes)
add_test(NAME KMeansTest COMMAND test_kmeans)
add_test(NAME PCATest COMMAND test_pca)
add_test(NAME TreeTest COMMAND test_tree)
add_test(NAME SVMTest COMMAND test_svm)
add_test(NAME EnsembleTest COMMAND test_ensemble)
add_test(NAME ClusterTest COMMAND test_cluster)
add_test(NAME DecompositionTest COMMAND test_decomposition)
add_test(NAME MetricsTest COMMAND test_metrics)
add_test(NAME ModelSelectionTest COMMAND test_model_selection)
add_test(NAME LDATest COMMAND test_lda)
add_test(NAME GradientBoostingTest COMMAND test_gradient_boosting)
