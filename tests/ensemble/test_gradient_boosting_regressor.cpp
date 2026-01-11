#include <gtest/gtest.h>
#include "ingenuityml/gradient_boosting.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class GradientBoostingRegressorTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y(i) = X(i, 0) + 2.0 * X(i, 1) - 0.5 * X(i, 2) + 0.1 * X(i, 3) + 0.1 * (MatrixXd::Random(1, 1))(0, 0);
        }
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y;
};

// Positive test cases
TEST_F(GradientBoostingRegressorTest, GradientBoostingRegressorFit) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    gbr.fit(X, y);
    
    EXPECT_TRUE(gbr.is_fitted());
    EXPECT_EQ(gbr.n_estimators(), 10);
}

TEST_F(GradientBoostingRegressorTest, GradientBoostingRegressorPredict) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    gbr.fit(X, y);
    
    VectorXd y_pred = gbr.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    EXPECT_TRUE(y_pred.allFinite());
}

TEST_F(GradientBoostingRegressorTest, GradientBoostingRegressorPerformance) {
    ensemble::GradientBoostingRegressor gbr(20, 0.1, 3);
    gbr.fit(X, y);
    
    VectorXd y_pred = gbr.predict(X);
    
    double mse = metrics::mean_squared_error(y, y_pred);
    EXPECT_LT(mse, 10.0);
    
    double r2 = metrics::r2_score(y, y_pred);
    EXPECT_GT(r2, 0.5);
}

// Negative test cases
TEST_F(GradientBoostingRegressorTest, GradientBoostingRegressorNotFitted) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    
    EXPECT_FALSE(gbr.is_fitted());
    EXPECT_THROW(gbr.predict(X_test), std::runtime_error);
}

TEST_F(GradientBoostingRegressorTest, GradientBoostingRegressorWrongFeatureCount) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    gbr.fit(X, y);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(gbr.predict(X_wrong), std::invalid_argument);
}

TEST_F(GradientBoostingRegressorTest, GradientBoostingRegressorEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    EXPECT_THROW(gbr.fit(X_empty, y_empty), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

