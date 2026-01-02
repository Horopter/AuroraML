#include <gtest/gtest.h>
#include "auroraml/isotonic.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class IsotonicTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 50;
        
        X = VectorXd::LinSpaced(n_samples, 0.0, 10.0);
        y = X.array() + 0.1 * VectorXd::Random(n_samples).array();
        
        X_test = VectorXd::LinSpaced(20, 0.5, 9.5);
    }
    
    int n_samples;
    VectorXd X, y, X_test;
};

// Positive test cases
TEST_F(IsotonicTest, IsotonicRegressionFit) {
    isotonic::IsotonicRegression iso;
    iso.fit(X, y);
    
    EXPECT_TRUE(iso.is_fitted());
}

TEST_F(IsotonicTest, IsotonicRegressionPredict) {
    isotonic::IsotonicRegression iso;
    iso.fit(X, y);
    
    VectorXd y_pred = iso.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.size());
    EXPECT_FALSE(y_pred.array().isNaN().any());
}

TEST_F(IsotonicTest, IsotonicRegressionTransform) {
    isotonic::IsotonicRegression iso;
    iso.fit(X, y);
    
    // Transform is for 1D input (VectorXd)
    VectorXd y_transformed = iso.transform(X);
    EXPECT_EQ(y_transformed.size(), X.size());
}

TEST_F(IsotonicTest, IsotonicRegressionPerformance) {
    isotonic::IsotonicRegression iso;
    iso.fit(X, y);
    
    VectorXd y_pred = iso.predict(X);
    
    double mse = metrics::mean_squared_error(y, y_pred);
    EXPECT_LT(mse, 10.0);
}

// Negative test cases
TEST_F(IsotonicTest, IsotonicRegressionNotFitted) {
    isotonic::IsotonicRegression iso;
    EXPECT_THROW(iso.predict(X_test), std::runtime_error);
}

TEST_F(IsotonicTest, IsotonicRegressionEmptyData) {
    VectorXd X_empty = VectorXd::Zero(0);
    VectorXd y_empty = VectorXd::Zero(0);
    
    isotonic::IsotonicRegression iso;
    EXPECT_THROW(iso.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(IsotonicTest, IsotonicRegressionDimensionMismatch) {
    VectorXd X_wrong = VectorXd::Random(n_samples);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    isotonic::IsotonicRegression iso;
    EXPECT_THROW(iso.fit(X_wrong, y_wrong), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
