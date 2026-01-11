#include <gtest/gtest.h>
#include "ingenuityml/linear_model.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class ElasticNetTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 3;
        
        X = MatrixXd::Random(n_samples, n_features);
        true_coef = VectorXd::Random(n_features);
        y = X * true_coef + VectorXd::Random(n_samples) * 0.1;
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y, true_coef;
};

// Positive test cases
TEST_F(ElasticNetTest, ElasticNetFit) {
    linear_model::ElasticNet enet(0.1, 0.5);
    enet.fit(X, y);
    
    EXPECT_TRUE(enet.is_fitted());
    EXPECT_EQ(enet.coef().size(), n_features);
}

TEST_F(ElasticNetTest, ElasticNetPredict) {
    linear_model::ElasticNet enet(0.1, 0.5);
    enet.fit(X, y);
    
    VectorXd y_pred = enet.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(ElasticNetTest, ElasticNetPerformance) {
    linear_model::ElasticNet enet(0.1, 0.5);
    enet.fit(X, y);
    
    VectorXd y_pred = enet.predict(X);
    
    double mse = metrics::mean_squared_error(y, y_pred);
    EXPECT_LT(mse, 10.0);
    
    double r2 = metrics::r2_score(y, y_pred);
    EXPECT_GT(r2, 0.5);
}

TEST_F(ElasticNetTest, ElasticNetDifferentAlpha) {
    std::vector<double> alphas = {0.1, 1.0, 10.0};
    
    for (double alpha : alphas) {
        linear_model::ElasticNet enet(alpha, 0.5);
        enet.fit(X, y);
        EXPECT_TRUE(enet.is_fitted());
        
        VectorXd y_pred = enet.predict(X);
        EXPECT_EQ(y_pred.size(), n_samples);
    }
}

TEST_F(ElasticNetTest, ElasticNetDifferentL1Ratio) {
    std::vector<double> l1_ratios = {0.1, 0.5, 0.9};
    
    for (double l1_ratio : l1_ratios) {
        linear_model::ElasticNet enet(0.1, l1_ratio);
        enet.fit(X, y);
        EXPECT_TRUE(enet.is_fitted());
        
        VectorXd y_pred = enet.predict(X);
        EXPECT_EQ(y_pred.size(), n_samples);
    }
}

// Negative test cases
TEST_F(ElasticNetTest, ElasticNetNotFitted) {
    linear_model::ElasticNet enet(0.1, 0.5);
    
    EXPECT_FALSE(enet.is_fitted());
    EXPECT_THROW(enet.predict(X), std::runtime_error);
}

TEST_F(ElasticNetTest, ElasticNetWrongFeatureCount) {
    linear_model::ElasticNet enet(0.1, 0.5);
    enet.fit(X, y);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(enet.predict(X_wrong), std::runtime_error);
}

TEST_F(ElasticNetTest, ElasticNetEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    linear_model::ElasticNet enet(0.1, 0.5);
    EXPECT_THROW(enet.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(ElasticNetTest, ElasticNetDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    linear_model::ElasticNet enet(0.1, 0.5);
    EXPECT_THROW(enet.fit(X_wrong, y_wrong), std::invalid_argument);
}

TEST_F(ElasticNetTest, ElasticNetNegativeAlpha) {
    linear_model::ElasticNet enet(-1.0, 0.5);
    EXPECT_THROW(enet.fit(X, y), std::invalid_argument);
}

TEST_F(ElasticNetTest, ElasticNetInvalidL1Ratio) {
    // Test l1_ratio < 0
    linear_model::ElasticNet enet1(0.1, -0.1);
    EXPECT_THROW(enet1.fit(X, y), std::invalid_argument);
    
    // Test l1_ratio > 1
    linear_model::ElasticNet enet2(0.1, 1.5);
    EXPECT_THROW(enet2.fit(X, y), std::invalid_argument);
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

