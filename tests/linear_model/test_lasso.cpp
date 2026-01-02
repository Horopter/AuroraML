#include <gtest/gtest.h>
#include "auroraml/linear_model.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class LassoTest : public ::testing::Test {
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
TEST_F(LassoTest, LassoFit) {
    linear_model::Lasso lasso(0.1);
    lasso.fit(X, y);
    
    EXPECT_TRUE(lasso.is_fitted());
    EXPECT_EQ(lasso.coef().size(), n_features);
}

TEST_F(LassoTest, LassoPredict) {
    linear_model::Lasso lasso(0.1);
    lasso.fit(X, y);
    
    VectorXd y_pred = lasso.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    EXPECT_FALSE(y_pred.array().isNaN().any());
}

TEST_F(LassoTest, LassoAlphaParameter) {
    std::vector<double> alphas = {0.1, 1.0, 10.0};
    
    for (double alpha : alphas) {
        linear_model::Lasso lasso(alpha);
        lasso.fit(X, y);
        EXPECT_TRUE(lasso.is_fitted());
        
        VectorXd y_pred = lasso.predict(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

TEST_F(LassoTest, LassoPerformance) {
    linear_model::Lasso lasso(0.1);
    lasso.fit(X, y);
    
    VectorXd y_pred = lasso.predict(X);
    
    double r2 = metrics::r2_score(y, y_pred);
    EXPECT_GT(r2, 0.7);
}

TEST_F(LassoTest, LassoIsFitted) {
    linear_model::Lasso lasso(0.1);
    EXPECT_FALSE(lasso.is_fitted());
    
    lasso.fit(X, y);
    EXPECT_TRUE(lasso.is_fitted());
}

// Negative test cases
TEST_F(LassoTest, LassoNotFitted) {
    linear_model::Lasso lasso(0.1);
    EXPECT_THROW(lasso.predict(X_test), std::runtime_error);
}

TEST_F(LassoTest, LassoWrongFeatureCount) {
    linear_model::Lasso lasso(0.1);
    lasso.fit(X, y);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(lasso.predict(X_wrong), std::runtime_error);
}

TEST_F(LassoTest, LassoEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    linear_model::Lasso lasso(0.1);
    EXPECT_THROW(lasso.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(LassoTest, LassoDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    linear_model::Lasso lasso(0.1);
    EXPECT_THROW(lasso.fit(X_wrong, y_wrong), std::invalid_argument);
}

TEST_F(LassoTest, LassoNegativeAlpha) {
    linear_model::Lasso lasso(-1.0);
    EXPECT_THROW(lasso.fit(X, y), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

