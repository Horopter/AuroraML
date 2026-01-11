#include <gtest/gtest.h>
#include "ingenuityml/neighbors.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>

namespace ingenuityml {
namespace test {

class KNeighborsClassifierTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 50;
        n_features = 2;
        
        X = MatrixXd::Random(n_samples, n_features);
        
        // Create classification problem
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        // Create test data
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases
TEST_F(KNeighborsClassifierTest, KNeighborsClassifierFit) {
    neighbors::KNeighborsClassifier knn(3);
    knn.fit(X, y_classification);
    
    EXPECT_TRUE(knn.is_fitted());
}

TEST_F(KNeighborsClassifierTest, KNeighborsClassifierPredict) {
    neighbors::KNeighborsClassifier knn(3);
    knn.fit(X, y_classification);
    
    VectorXi y_pred = knn.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are valid class labels
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_TRUE(y_pred(i) == 0 || y_pred(i) == 1);
    }
}

TEST_F(KNeighborsClassifierTest, KNeighborsClassifierPredictProba) {
    neighbors::KNeighborsClassifier knn(3);
    knn.fit(X, y_classification);
    
    MatrixXd y_proba = knn.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_F(KNeighborsClassifierTest, KNeighborsClassifierPerformance) {
    neighbors::KNeighborsClassifier knn(3);
    knn.fit(X, y_classification);
    
    VectorXi y_pred = knn.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, y_pred);
    EXPECT_GT(accuracy, 0.5);
}

TEST_F(KNeighborsClassifierTest, KNeighborsClassifierDifferentK) {
    std::vector<int> k_values = {1, 3, 5};
    
    for (int k : k_values) {
        neighbors::KNeighborsClassifier knn(k);
        knn.fit(X, y_classification);
        EXPECT_TRUE(knn.is_fitted());
        
        VectorXi y_pred = knn.predict_classes(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

// Negative test cases
TEST_F(KNeighborsClassifierTest, KNeighborsClassifierNotFitted) {
    neighbors::KNeighborsClassifier knn(3);
    
    EXPECT_FALSE(knn.is_fitted());
    EXPECT_THROW(knn.predict_classes(X), std::runtime_error);
    EXPECT_THROW(knn.predict_proba(X), std::runtime_error);
}

TEST_F(KNeighborsClassifierTest, KNeighborsClassifierWrongFeatureCount) {
    neighbors::KNeighborsClassifier knn(3);
    knn.fit(X, y_classification);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(knn.predict_classes(X_wrong), std::invalid_argument);
}

TEST_F(KNeighborsClassifierTest, KNeighborsClassifierEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    neighbors::KNeighborsClassifier knn(3);
    EXPECT_THROW(knn.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(KNeighborsClassifierTest, KNeighborsClassifierNegativeK) {
    // Negative K throws exception during construction
    EXPECT_THROW(neighbors::KNeighborsClassifier knn(-1), std::invalid_argument);
}

TEST_F(KNeighborsClassifierTest, KNeighborsClassifierZeroK) {
    // Zero K throws exception during construction
    EXPECT_THROW(neighbors::KNeighborsClassifier knn(0), std::invalid_argument);
}

TEST_F(KNeighborsClassifierTest, KNeighborsClassifierTooLargeK) {
    // Too large K is currently not validated - just passes through
    // This test verifies the current behavior (no exception thrown)
    neighbors::KNeighborsClassifier knn(n_samples + 1);
    knn.fit(X, y_classification);
    EXPECT_TRUE(knn.is_fitted());
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    // Enable test shuffling within this file
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;  // Reproducible shuffle
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

