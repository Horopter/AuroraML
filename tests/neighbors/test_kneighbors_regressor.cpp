#include <gtest/gtest.h>
#include "auroraml/neighbors.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class KNeighborsRegressorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 50;
        n_features = 2;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_regression = VectorXd::Random(n_samples);
        
        // Create test data
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_regression;
};

// Positive test cases
TEST_F(KNeighborsRegressorTest, KNeighborsRegressorFit) {
    neighbors::KNeighborsRegressor knn(3);
    knn.fit(X, y_regression);
    
    EXPECT_TRUE(knn.is_fitted());
}

TEST_F(KNeighborsRegressorTest, KNeighborsRegressorPredict) {
    neighbors::KNeighborsRegressor knn(3);
    knn.fit(X, y_regression);
    
    VectorXd y_pred = knn.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are reasonable
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(KNeighborsRegressorTest, KNeighborsRegressorPerformance) {
    neighbors::KNeighborsRegressor knn(3);
    knn.fit(X, y_regression);
    
    VectorXd y_pred = knn.predict(X);
    
    double mse = metrics::mean_squared_error(y_regression, y_pred);
    EXPECT_LT(mse, 10.0);
}

TEST_F(KNeighborsRegressorTest, KNeighborsRegressorDifferentK) {
    std::vector<int> k_values = {1, 3, 5};
    
    for (int k : k_values) {
        neighbors::KNeighborsRegressor knn(k);
        knn.fit(X, y_regression);
        EXPECT_TRUE(knn.is_fitted());
        
        VectorXd y_pred = knn.predict(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

// Negative test cases
TEST_F(KNeighborsRegressorTest, KNeighborsRegressorNotFitted) {
    neighbors::KNeighborsRegressor knn(3);
    
    EXPECT_FALSE(knn.is_fitted());
    EXPECT_THROW(knn.predict(X), std::runtime_error);
}

TEST_F(KNeighborsRegressorTest, KNeighborsRegressorWrongFeatureCount) {
    neighbors::KNeighborsRegressor knn(3);
    knn.fit(X, y_regression);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(knn.predict(X_wrong), std::runtime_error);
}

TEST_F(KNeighborsRegressorTest, KNeighborsRegressorEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    neighbors::KNeighborsRegressor knn(3);
    EXPECT_THROW(knn.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(KNeighborsRegressorTest, KNeighborsRegressorNegativeK) {
    neighbors::KNeighborsRegressor knn(-1);
    EXPECT_THROW(knn.fit(X, y_regression), std::invalid_argument);
}

TEST_F(KNeighborsRegressorTest, KNeighborsRegressorZeroK) {
    neighbors::KNeighborsRegressor knn(0);
    EXPECT_THROW(knn.fit(X, y_regression), std::invalid_argument);
}

TEST_F(KNeighborsRegressorTest, KNeighborsRegressorTooLargeK) {
    neighbors::KNeighborsRegressor knn(n_samples + 1);
    EXPECT_THROW(knn.fit(X, y_regression), std::invalid_argument);
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

