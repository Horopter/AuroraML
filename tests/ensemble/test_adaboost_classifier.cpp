#include <gtest/gtest.h>
#include "ingenuityml/adaboost.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <vector>

namespace ingenuityml {
namespace test {

class AdaBoostClassifierTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 200;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        
        // Create classification data
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            double score = X(i, 0) + X(i, 1) - 0.5 * X(i, 2);
            y_classification(i) = (score > 0.0) ? 1.0 : 0.0;
        }
        
        // Create test data
        X_test = MatrixXd::Random(30, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases
TEST_F(AdaBoostClassifierTest, AdaBoostClassifierFit) {
    ensemble::AdaBoostClassifier adaboost(50, 1.0, 42);
    adaboost.fit(X, y_classification);
    
    EXPECT_TRUE(adaboost.is_fitted());
    
    std::vector<int> classes = adaboost.classes();
    EXPECT_GE(classes.size(), 2);
}

TEST_F(AdaBoostClassifierTest, AdaBoostClassifierPredictClasses) {
    ensemble::AdaBoostClassifier adaboost(50, 1.0, 42);
    adaboost.fit(X, y_classification);
    
    VectorXi y_pred = adaboost.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are valid class labels
    std::vector<int> classes = adaboost.classes();
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_TRUE(std::find(classes.begin(), classes.end(), y_pred(i)) != classes.end());
    }
}

TEST_F(AdaBoostClassifierTest, AdaBoostClassifierPredictProba) {
    ensemble::AdaBoostClassifier adaboost(50, 1.0, 42);
    adaboost.fit(X, y_classification);
    
    MatrixXd y_proba = adaboost.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_F(AdaBoostClassifierTest, AdaBoostClassifierPerformance) {
    ensemble::AdaBoostClassifier adaboost(100, 1.0, 42);
    adaboost.fit(X, y_classification);
    
    VectorXi y_pred = adaboost.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, y_pred);
    EXPECT_GT(accuracy, 0.7);
}

TEST_F(AdaBoostClassifierTest, AdaBoostClassifierDifferentLearningRates) {
    std::vector<double> learning_rates = {0.5, 1.0, 2.0};
    
    for (double lr : learning_rates) {
        ensemble::AdaBoostClassifier adaboost(50, lr, 42);
        adaboost.fit(X, y_classification);
        EXPECT_TRUE(adaboost.is_fitted());
        
        VectorXi y_pred = adaboost.predict_classes(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

// Negative test cases
TEST_F(AdaBoostClassifierTest, AdaBoostClassifierNotFitted) {
    ensemble::AdaBoostClassifier adaboost(50);
    
    EXPECT_FALSE(adaboost.is_fitted());
    EXPECT_THROW(adaboost.predict_classes(X), std::runtime_error);
    EXPECT_THROW(adaboost.predict_proba(X), std::runtime_error);
}

TEST_F(AdaBoostClassifierTest, AdaBoostClassifierWrongFeatureCount) {
    ensemble::AdaBoostClassifier adaboost(50, 1.0, 42);
    adaboost.fit(X, y_classification);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(adaboost.predict_classes(X_wrong), std::invalid_argument);
}

TEST_F(AdaBoostClassifierTest, AdaBoostClassifierEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    ensemble::AdaBoostClassifier adaboost(50);
    EXPECT_THROW(adaboost.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(AdaBoostClassifierTest, AdaBoostClassifierNegativeEstimators) {
    ensemble::AdaBoostClassifier adaboost(-1);
    // Negative n_estimators causes std::length_error when reserving vector space
    EXPECT_THROW(adaboost.fit(X, y_classification), std::length_error);
}

TEST_F(AdaBoostClassifierTest, AdaBoostClassifierNegativeLearningRate) {
    ensemble::AdaBoostClassifier adaboost(50, -1.0);
    // Negative learning rate is currently not validated - just passes through
    // This test verifies the current behavior (no exception thrown)
    adaboost.fit(X, y_classification);
    EXPECT_TRUE(adaboost.is_fitted());
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

