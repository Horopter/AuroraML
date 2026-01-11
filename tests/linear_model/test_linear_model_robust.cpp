#include <gtest/gtest.h>
#include "ingenuityml/linear_model.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class LinearModelRobustTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        X = MatrixXd::Random(n_samples, n_features);
        true_coef = VectorXd::Random(n_features);
        y = X * true_coef + VectorXd::Random(n_samples) * 0.05;
        for (int i = 0; i < 10; ++i) {
            y(i) += 5.0;
        }
        X_test = MatrixXd::Random(20, n_features);
    }

    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y, true_coef;
};

TEST_F(LinearModelRobustTest, RANSACRegressorFitPredict) {
    linear_model::RANSACRegressor model(50, -1, -1.0, 42, true);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelRobustTest, TheilSenRegressorFitPredict) {
    linear_model::TheilSenRegressor model(50, 42, true);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelRobustTest, QuantileRegressorFitPredict) {
    linear_model::QuantileRegressor model(0.5, 0.0, true, 500, 1e-4, 0.05);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelRobustTest, QuantileRegressorInvalidQuantile) {
    linear_model::QuantileRegressor model(1.0);
    EXPECT_THROW(model.fit(X, y), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
