#include <gtest/gtest.h>
#include "auroraml/tree.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class DecisionTreeRegressorTest : public ::testing::Test {
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
TEST_F(DecisionTreeRegressorTest, DecisionTreeRegressorFit) {
    tree::DecisionTreeRegressor dt("mse", 5, 2, 1, 42);
    dt.fit(X, y_regression);
    
    EXPECT_TRUE(dt.is_fitted());
}

TEST_F(DecisionTreeRegressorTest, DecisionTreeRegressorPredict) {
    tree::DecisionTreeRegressor dt("mse", 5, 2, 1, 42);
    dt.fit(X, y_regression);
    
    VectorXd y_pred = dt.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are reasonable
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(DecisionTreeRegressorTest, DecisionTreeRegressorPerformance) {
    tree::DecisionTreeRegressor dt("mse", 5, 2, 1, 42);
    dt.fit(X, y_regression);
    
    VectorXd y_pred = dt.predict(X);
    
    double mse = metrics::mean_squared_error(y_regression, y_pred);
    EXPECT_LT(mse, 10.0);
    
    double r2 = metrics::r2_score(y_regression, y_pred);
    EXPECT_GT(r2, 0.5);
}

// Negative test cases
TEST_F(DecisionTreeRegressorTest, DecisionTreeRegressorNotFitted) {
    tree::DecisionTreeRegressor dt("mse", 5);
    
    EXPECT_FALSE(dt.is_fitted());
    EXPECT_THROW(dt.predict(X), std::runtime_error);
}

TEST_F(DecisionTreeRegressorTest, DecisionTreeRegressorWrongFeatureCount) {
    tree::DecisionTreeRegressor dt("mse", 5, 2, 1, 42);
    dt.fit(X, y_regression);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(dt.predict(X_wrong), std::invalid_argument);
}

TEST_F(DecisionTreeRegressorTest, DecisionTreeRegressorEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    tree::DecisionTreeRegressor dt("mse", 5);
    EXPECT_THROW(dt.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(DecisionTreeRegressorTest, DecisionTreeRegressorDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    tree::DecisionTreeRegressor dt("mse", 5);
    EXPECT_THROW(dt.fit(X_wrong, y_wrong), std::invalid_argument);
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

