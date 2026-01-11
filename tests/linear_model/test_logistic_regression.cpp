#include <gtest/gtest.h>
#include "ingenuityml/linear_model.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class LogisticRegressionTest : public ::testing::Test {
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
TEST_F(LogisticRegressionTest, LogisticRegressionFit) {
    linear_model::LogisticRegression lr(1.0, 1000, 42);
    lr.fit(X, y_classification);
    
    EXPECT_TRUE(lr.is_fitted());
}

TEST_F(LogisticRegressionTest, LogisticRegressionPredict) {
    linear_model::LogisticRegression lr(1.0, 1000, 42);
    lr.fit(X, y_classification);
    
    VectorXi y_pred = lr.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are valid class labels
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_TRUE(y_pred(i) == 0 || y_pred(i) == 1);
    }
}

TEST_F(LogisticRegressionTest, LogisticRegressionPredictProba) {
    linear_model::LogisticRegression lr(1.0, 1000, 42);
    lr.fit(X, y_classification);
    
    MatrixXd y_proba = lr.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_F(LogisticRegressionTest, LogisticRegressionPerformance) {
    linear_model::LogisticRegression lr(1.0, 1000, 42);
    lr.fit(X, y_classification);
    
    VectorXi y_pred = lr.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, y_pred);
    EXPECT_GT(accuracy, 0.5);
}

TEST_F(LogisticRegressionTest, LogisticRegressionDecisionFunction) {
    linear_model::LogisticRegression lr(1.0, 1000, 42);
    lr.fit(X, y_classification);
    
    VectorXd decision = lr.decision_function(X_test);
    EXPECT_EQ(decision.size(), X_test.rows());
}

// Negative test cases
TEST_F(LogisticRegressionTest, LogisticRegressionNotFitted) {
    linear_model::LogisticRegression lr(1.0);
    
    EXPECT_FALSE(lr.is_fitted());
    EXPECT_THROW(lr.predict_classes(X), std::runtime_error);
    EXPECT_THROW(lr.predict_proba(X), std::runtime_error);
}

TEST_F(LogisticRegressionTest, LogisticRegressionWrongFeatureCount) {
    linear_model::LogisticRegression lr(1.0, 1000, 42);
    lr.fit(X, y_classification);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(lr.predict_classes(X_wrong), std::runtime_error);
}

TEST_F(LogisticRegressionTest, LogisticRegressionEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    linear_model::LogisticRegression lr(1.0);
    EXPECT_THROW(lr.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(LogisticRegressionTest, LogisticRegressionDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    linear_model::LogisticRegression lr(1.0);
    EXPECT_THROW(lr.fit(X_wrong, y_wrong), std::invalid_argument);
}

TEST_F(LogisticRegressionTest, LogisticRegressionNegativeC) {
    linear_model::LogisticRegression lr(-1.0);
    EXPECT_THROW(lr.fit(X, y_classification), std::invalid_argument);
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

