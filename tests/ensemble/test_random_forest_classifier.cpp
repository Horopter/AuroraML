#include <gtest/gtest.h>
#include "ingenuityml/random_forest.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>

namespace ingenuityml {
namespace test {

class RandomForestClassifierTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 3;
        
        X = MatrixXd::Random(n_samples, n_features);
        
        // Create classification problem
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        // Create test data
        X_test = MatrixXd::Random(30, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases
TEST_F(RandomForestClassifierTest, RandomForestClassifierFit) {
    ensemble::RandomForestClassifier rf(10, 5, -1, 42);
    rf.fit(X, y_classification);
    
    EXPECT_TRUE(rf.is_fitted());
}

TEST_F(RandomForestClassifierTest, RandomForestClassifierPredictClasses) {
    ensemble::RandomForestClassifier rf(10, 5, -1, 42);
    rf.fit(X, y_classification);
    
    VectorXi y_pred = rf.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are valid class labels (0 or 1 for binary classification)
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_TRUE(y_pred(i) == 0 || y_pred(i) == 1);
    }
}

TEST_F(RandomForestClassifierTest, RandomForestClassifierPredictProba) {
    ensemble::RandomForestClassifier rf(10, 5, -1, 42);
    rf.fit(X, y_classification);
    
    MatrixXd y_proba = rf.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_F(RandomForestClassifierTest, RandomForestClassifierPerformance) {
    ensemble::RandomForestClassifier rf(10, 5, -1, 42);
    rf.fit(X, y_classification);
    
    VectorXi y_pred = rf.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, y_pred);
    EXPECT_GT(accuracy, 0.7);
}

// Negative test cases
TEST_F(RandomForestClassifierTest, RandomForestClassifierNotFitted) {
    ensemble::RandomForestClassifier rf(10);
    
    EXPECT_FALSE(rf.is_fitted());
    EXPECT_THROW(rf.predict_classes(X), std::runtime_error);
    EXPECT_THROW(rf.predict_proba(X), std::runtime_error);
}

TEST_F(RandomForestClassifierTest, RandomForestClassifierWrongFeatureCount) {
    ensemble::RandomForestClassifier rf(10, 5, -1, 42);
    rf.fit(X, y_classification);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(rf.predict_classes(X_wrong), std::runtime_error);
}

TEST_F(RandomForestClassifierTest, RandomForestClassifierEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    ensemble::RandomForestClassifier rf(10);
    EXPECT_THROW(rf.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(RandomForestClassifierTest, RandomForestClassifierNegativeEstimators) {
    ensemble::RandomForestClassifier rf(-5);
    EXPECT_THROW(rf.fit(X, y_classification), std::length_error);
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

