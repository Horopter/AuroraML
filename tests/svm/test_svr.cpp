#include <gtest/gtest.h>
#include "auroraml/svm.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class SVRTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 3;
        
        X = MatrixXd::Random(n_samples, n_features);
        y = X.col(0) + 0.5 * X.col(1) - 0.3 * X.col(2) + VectorXd::Random(n_samples) * 0.1;
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y;
};

// Positive test cases
TEST_F(SVRTest, SVRFit) {
    svm::SVR svr(1.0, 0.1, 42);
    svr.fit(X, y);
    
    EXPECT_TRUE(svr.is_fitted());
}

TEST_F(SVRTest, SVRPredict) {
    svm::SVR svr(1.0, 0.1, 42);
    svr.fit(X, y);
    
    VectorXd y_pred = svr.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    EXPECT_FALSE(y_pred.array().isNaN().any());
}

TEST_F(SVRTest, SVRPerformance) {
    svm::SVR svr(1.0, 0.1, 42);
    svr.fit(X, y);
    
    VectorXd y_pred = svr.predict(X);
    
    double mse = metrics::mean_squared_error(y, y_pred);
    EXPECT_LT(mse, 10.0);
}

// Negative test cases
TEST_F(SVRTest, SVRNotFitted) {
    svm::SVR svr(1.0, 0.1);
    EXPECT_THROW(svr.predict(X_test), std::runtime_error);
}

TEST_F(SVRTest, SVRWrongFeatureCount) {
    svm::SVR svr(1.0, 0.1, 42);
    svr.fit(X, y);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(svr.predict(X_wrong), std::runtime_error);
}

TEST_F(SVRTest, SVREmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    svm::SVR svr(1.0, 0.1);
    EXPECT_THROW(svr.fit(X_empty, y_empty), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

