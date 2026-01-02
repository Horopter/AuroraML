#include <gtest/gtest.h>
#include "auroraml/linear_model.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class LinearModelGLMTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 80;
        n_features = 3;
        X = MatrixXd::Random(n_samples, n_features);
        true_coef = VectorXd::Random(n_features);
        VectorXd eta = X * true_coef;
        VectorXd mu = eta.array().exp();
        y_pos = mu;
        X_test = MatrixXd::Random(20, n_features);
    }

    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_pos, true_coef;
};

TEST_F(LinearModelGLMTest, PoissonRegressorFitPredict) {
    linear_model::PoissonRegressor model(0.0, true, 200, 1e-4, 0.01);
    model.fit(X, y_pos);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
    EXPECT_TRUE((preds.array() > 0.0).all());
}

TEST_F(LinearModelGLMTest, GammaRegressorFitPredict) {
    VectorXd y_gamma = y_pos.array() + 0.5;
    linear_model::GammaRegressor model(0.0, true, 200, 1e-4, 0.01);
    model.fit(X, y_gamma);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
    EXPECT_TRUE((preds.array() > 0.0).all());
}

TEST_F(LinearModelGLMTest, TweedieRegressorFitPredict) {
    linear_model::TweedieRegressor model(1.5, 0.0, true, 200, 1e-4, 0.01);
    model.fit(X, y_pos);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
    EXPECT_TRUE((preds.array() > 0.0).all());
}

TEST_F(LinearModelGLMTest, GammaRegressorInvalidY) {
    VectorXd y_bad = VectorXd::Constant(n_samples, -1.0);
    linear_model::GammaRegressor model;
    EXPECT_THROW(model.fit(X, y_bad), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
