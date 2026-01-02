#include <gtest/gtest.h>
#include "auroraml/random_forest.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class RandomForestRegressorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 3;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_regression = VectorXd::Random(n_samples);
        
        // Create test data
        X_test = MatrixXd::Random(30, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_regression;
};

// Positive test cases
TEST_F(RandomForestRegressorTest, RandomForestRegressorFit) {
    ensemble::RandomForestRegressor rf(10, 5, -1, 42);
    rf.fit(X, y_regression);
    
    EXPECT_TRUE(rf.is_fitted());
}

TEST_F(RandomForestRegressorTest, RandomForestRegressorPredict) {
    ensemble::RandomForestRegressor rf(10, 5, -1, 42);
    rf.fit(X, y_regression);
    
    VectorXd y_pred = rf.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are reasonable
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(RandomForestRegressorTest, RandomForestRegressorPerformance) {
    ensemble::RandomForestRegressor rf(10, 5, -1, 42);
    rf.fit(X, y_regression);
    
    VectorXd y_pred = rf.predict(X);
    
    double mse = metrics::mean_squared_error(y_regression, y_pred);
    EXPECT_LT(mse, 10.0);
    
    double r2 = metrics::r2_score(y_regression, y_pred);
    EXPECT_GT(r2, 0.5);
}

// Negative test cases
TEST_F(RandomForestRegressorTest, RandomForestRegressorNotFitted) {
    ensemble::RandomForestRegressor rf(10);
    
    EXPECT_FALSE(rf.is_fitted());
    EXPECT_THROW(rf.predict(X), std::runtime_error);
}

TEST_F(RandomForestRegressorTest, RandomForestRegressorWrongFeatureCount) {
    ensemble::RandomForestRegressor rf(10, 5, -1, 42);
    rf.fit(X, y_regression);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(rf.predict(X_wrong), std::runtime_error);
}

TEST_F(RandomForestRegressorTest, RandomForestRegressorEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    ensemble::RandomForestRegressor rf(10);
    EXPECT_THROW(rf.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(RandomForestRegressorTest, RandomForestRegressorNegativeEstimators) {
    ensemble::RandomForestRegressor rf(-5);
    EXPECT_THROW(rf.fit(X, y_regression), std::length_error);
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

