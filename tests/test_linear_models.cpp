#include <gtest/gtest.h>
#include "auroraml/linear_model.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class LinearModelsTest : public ::testing::Test {
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

TEST_F(LinearModelsTest, LinearRegressionFit) {
    linear_model::LinearRegression lr;
    lr.fit(X, y);
    
    EXPECT_TRUE(lr.is_fitted());
    EXPECT_EQ(lr.coef().size(), n_features);
}

TEST_F(LinearModelsTest, LinearRegressionPredict) {
    linear_model::LinearRegression lr;
    lr.fit(X, y);
    
    VectorXd y_pred = lr.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(LinearModelsTest, RidgeRegressionFit) {
    linear_model::Ridge ridge(1.0);
    ridge.fit(X, y);
    
    EXPECT_TRUE(ridge.is_fitted());
    EXPECT_EQ(ridge.coef().size(), n_features);
}

TEST_F(LinearModelsTest, RidgeRegressionPredict) {
    linear_model::Ridge ridge(1.0);
    ridge.fit(X, y);
    
    VectorXd y_pred = ridge.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(LinearModelsTest, LassoRegressionFit) {
    linear_model::Lasso lasso(0.1);
    lasso.fit(X, y);
    
    EXPECT_TRUE(lasso.is_fitted());
    EXPECT_EQ(lasso.coef().size(), n_features);
}

TEST_F(LinearModelsTest, LassoRegressionPredict) {
    linear_model::Lasso lasso(0.1);
    lasso.fit(X, y);
    
    VectorXd y_pred = lasso.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(LinearModelsTest, ModelPersistence) {
    linear_model::LinearRegression lr;
    lr.fit(X, y);
    
    // Save model
    lr.save_to_file("test_model.bin");
    
    // Load model
    linear_model::LinearRegression lr_loaded;
    lr_loaded.load_from_file("test_model.bin");
    
    EXPECT_TRUE(lr_loaded.is_fitted());
    
    // Compare predictions
    VectorXd y_pred_orig = lr.predict(X);
    VectorXd y_pred_loaded = lr_loaded.predict(X);
    
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_NEAR(y_pred_orig(i), y_pred_loaded(i), 1e-10);
    }
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
