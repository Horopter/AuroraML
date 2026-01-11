#include <gtest/gtest.h>
#include "ingenuityml/dummy.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class DummyRegressorTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 50;
        n_features = 3;
        X = MatrixXd::Random(n_samples, n_features);
        X_test = MatrixXd::Random(10, n_features);
        y = VectorXd::LinSpaced(n_samples, 0.0, 1.0);
    }

    int n_samples;
    int n_features;
    MatrixXd X;
    MatrixXd X_test;
    VectorXd y;
};

TEST_F(DummyRegressorTest, MeanStrategy) {
    ensemble::DummyRegressor reg("mean");
    reg.fit(X, y);
    VectorXd preds = reg.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
    EXPECT_NEAR(preds(0), y.mean(), 1e-9);
}

TEST_F(DummyRegressorTest, MedianStrategy) {
    ensemble::DummyRegressor reg("median");
    reg.fit(X, y);
    VectorXd preds = reg.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
    double expected = 0.5;
    EXPECT_NEAR(preds(0), expected, 1e-9);
}

TEST_F(DummyRegressorTest, QuantileStrategy) {
    ensemble::DummyRegressor reg("quantile", 0.25);
    reg.fit(X, y);
    VectorXd preds = reg.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
    EXPECT_NEAR(preds(0), 0.25, 1e-9);
}

TEST_F(DummyRegressorTest, ConstantStrategy) {
    ensemble::DummyRegressor reg("constant", 0.5, 2.5);
    reg.fit(X, y);
    VectorXd preds = reg.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
    EXPECT_NEAR(preds(0), 2.5, 1e-9);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
