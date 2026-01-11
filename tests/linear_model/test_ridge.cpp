#include <gtest/gtest.h>
#include "ingenuityml/linear_model.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class RidgeTest : public ::testing::Test {
protected:
    void SetUp() override {
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
TEST_F(RidgeTest, RidgeFit) {
    linear_model::Ridge ridge(1.0);
    ridge.fit(X, y);
    
    EXPECT_TRUE(ridge.is_fitted());
    EXPECT_EQ(ridge.coef().size(), n_features);
}

TEST_F(RidgeTest, RidgePredict) {
    linear_model::Ridge ridge(1.0);
    ridge.fit(X, y);
    
    VectorXd y_pred = ridge.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    EXPECT_FALSE(y_pred.array().isNaN().any());
}

TEST_F(RidgeTest, RidgeAlphaParameter) {
    std::vector<double> alphas = {0.1, 1.0, 10.0};
    
    for (double alpha : alphas) {
        linear_model::Ridge ridge(alpha);
        ridge.fit(X, y);
        EXPECT_TRUE(ridge.is_fitted());
        
        VectorXd y_pred = ridge.predict(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

TEST_F(RidgeTest, RidgePerformance) {
    linear_model::Ridge ridge(1.0);
    ridge.fit(X, y);
    
    VectorXd y_pred = ridge.predict(X);
    
    double r2 = metrics::r2_score(y, y_pred);
    EXPECT_GT(r2, 0.7);
}

TEST_F(RidgeTest, RidgeIsFitted) {
    linear_model::Ridge ridge(1.0);
    EXPECT_FALSE(ridge.is_fitted());
    
    ridge.fit(X, y);
    EXPECT_TRUE(ridge.is_fitted());
}

// Negative test cases
TEST_F(RidgeTest, RidgeNotFitted) {
    linear_model::Ridge ridge(1.0);
    EXPECT_THROW(ridge.predict(X_test), std::runtime_error);
}

TEST_F(RidgeTest, RidgeWrongFeatureCount) {
    linear_model::Ridge ridge(1.0);
    ridge.fit(X, y);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(ridge.predict(X_wrong), std::runtime_error);
}

TEST_F(RidgeTest, RidgeEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    linear_model::Ridge ridge(1.0);
    EXPECT_THROW(ridge.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(RidgeTest, RidgeDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    linear_model::Ridge ridge(1.0);
    EXPECT_THROW(ridge.fit(X_wrong, y_wrong), std::invalid_argument);
}

TEST_F(RidgeTest, RidgeNegativeAlpha) {
    linear_model::Ridge ridge(-1.0);
    EXPECT_THROW(ridge.fit(X, y), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

