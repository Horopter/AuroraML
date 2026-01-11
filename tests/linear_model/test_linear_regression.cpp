#include <gtest/gtest.h>
#include "ingenuityml/linear_model.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class LinearRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 5;
        
        X = MatrixXd::Random(n_samples, n_features);
        true_coef = VectorXd::Random(n_features);
        y = X * true_coef + VectorXd::Random(n_samples) * 0.1;
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y, true_coef;
};

// Positive test cases
TEST_F(LinearRegressionTest, LinearRegressionFit) {
    linear_model::LinearRegression lr;
    lr.fit(X, y);
    
    EXPECT_TRUE(lr.is_fitted());
    EXPECT_EQ(lr.coef().size(), n_features);
}

TEST_F(LinearRegressionTest, LinearRegressionPredict) {
    linear_model::LinearRegression lr;
    lr.fit(X, y);
    
    VectorXd y_pred = lr.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are reasonable
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(LinearRegressionTest, LinearRegressionCoefficients) {
    linear_model::LinearRegression lr;
    lr.fit(X, y);
    
    VectorXd coef = lr.coef();
    double intercept = lr.intercept();
    
    EXPECT_EQ(coef.size(), n_features);
    EXPECT_FALSE(std::isnan(intercept));
    EXPECT_FALSE(std::isinf(intercept));
}

TEST_F(LinearRegressionTest, LinearRegressionPerformance) {
    linear_model::LinearRegression lr;
    lr.fit(X, y);
    
    VectorXd y_pred = lr.predict(X);
    
    double mse = metrics::mean_squared_error(y, y_pred);
    EXPECT_LT(mse, 10.0);
    
    double r2 = metrics::r2_score(y, y_pred);
    EXPECT_GT(r2, 0.5);
}

TEST_F(LinearRegressionTest, LinearRegressionParameters) {
    linear_model::LinearRegression lr(true, true, 1);
    
    Params params = lr.get_params();
    EXPECT_GT(params.size(), 0);
    EXPECT_TRUE(params.find("fit_intercept") != params.end());
}

TEST_F(LinearRegressionTest, LinearRegressionIsFitted) {
    linear_model::LinearRegression lr;
    EXPECT_FALSE(lr.is_fitted());
    
    lr.fit(X, y);
    EXPECT_TRUE(lr.is_fitted());
}

// Negative test cases
TEST_F(LinearRegressionTest, LinearRegressionNotFitted) {
    linear_model::LinearRegression lr;
    
    EXPECT_FALSE(lr.is_fitted());
    EXPECT_THROW(lr.predict(X_test), std::runtime_error);
}

TEST_F(LinearRegressionTest, LinearRegressionWrongFeatureCount) {
    linear_model::LinearRegression lr;
    lr.fit(X, y);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(lr.predict(X_wrong), std::runtime_error);
}

TEST_F(LinearRegressionTest, LinearRegressionEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    linear_model::LinearRegression lr;
    EXPECT_THROW(lr.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(LinearRegressionTest, LinearRegressionDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    linear_model::LinearRegression lr;
    EXPECT_THROW(lr.fit(X_wrong, y_wrong), std::invalid_argument);
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

