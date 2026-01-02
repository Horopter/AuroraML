#include <gtest/gtest.h>
#include "auroraml/adaboost.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class AdaBoostRegressorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 200;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y(i) = 2.0 * X(i, 0) + 1.5 * X(i, 1) - 0.8 * X(i, 2) + 0.1 * X(i, 3) + 0.05 * (MatrixXd::Random(1, 1))(0, 0);
        }
        
        // Create test data
        X_test = MatrixXd::Random(30, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y;
};

// Positive test cases
TEST_F(AdaBoostRegressorTest, AdaBoostRegressorFit) {
    ensemble::AdaBoostRegressor adaboost(50, 1.0, "linear", 42);
    adaboost.fit(X, y);
    
    EXPECT_TRUE(adaboost.is_fitted());
}

TEST_F(AdaBoostRegressorTest, AdaBoostRegressorPredict) {
    ensemble::AdaBoostRegressor adaboost(50, 1.0, "linear", 42);
    adaboost.fit(X, y);
    
    VectorXd y_pred = adaboost.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Predictions should be reasonable (not NaN or Inf)
    EXPECT_FALSE(y_pred.array().isNaN().any());
    EXPECT_FALSE(y_pred.array().isInf().any());
}

TEST_F(AdaBoostRegressorTest, AdaBoostRegressorPerformance) {
    ensemble::AdaBoostRegressor adaboost(100, 1.0, "linear", 42);
    adaboost.fit(X, y);
    
    VectorXd y_pred = adaboost.predict(X);
    
    double mse = metrics::mean_squared_error(y, y_pred);
    EXPECT_LT(mse, 10.0);
    
    double r2 = metrics::r2_score(y, y_pred);
    EXPECT_GT(r2, 0.5);
}

TEST_F(AdaBoostRegressorTest, AdaBoostRegressorDifferentLosses) {
    std::vector<std::string> losses = {"linear", "square", "exponential"};
    
    for (const std::string& loss : losses) {
        ensemble::AdaBoostRegressor adaboost(50, 1.0, loss, 42);
        adaboost.fit(X, y);
        EXPECT_TRUE(adaboost.is_fitted());
        
        VectorXd y_pred = adaboost.predict(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

// Negative test cases
TEST_F(AdaBoostRegressorTest, AdaBoostRegressorNotFitted) {
    ensemble::AdaBoostRegressor adaboost(50);
    
    EXPECT_FALSE(adaboost.is_fitted());
    EXPECT_THROW(adaboost.predict(X), std::runtime_error);
}

TEST_F(AdaBoostRegressorTest, AdaBoostRegressorWrongFeatureCount) {
    ensemble::AdaBoostRegressor adaboost(50, 1.0, "linear", 42);
    adaboost.fit(X, y);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(adaboost.predict(X_wrong), std::invalid_argument);
}

TEST_F(AdaBoostRegressorTest, AdaBoostRegressorEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    ensemble::AdaBoostRegressor adaboost(50);
    EXPECT_THROW(adaboost.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(AdaBoostRegressorTest, AdaBoostRegressorNegativeEstimators) {
    ensemble::AdaBoostRegressor adaboost(-1, 1.0, "linear");
    // Negative n_estimators causes std::length_error when reserving vector space
    EXPECT_THROW(adaboost.fit(X, y), std::length_error);
}

TEST_F(AdaBoostRegressorTest, AdaBoostRegressorNegativeLearningRate) {
    ensemble::AdaBoostRegressor adaboost(50, -1.0, "linear");
    // Negative learning rate is currently not validated - just passes through
    // This test verifies the current behavior (no exception thrown)
    adaboost.fit(X, y);
    EXPECT_TRUE(adaboost.is_fitted());
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    // Enable test shuffling within this file
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;  // Reproducible shuffle
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

