#include <gtest/gtest.h>
#include "auroraml/linear_model.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class LinearModelMultiTaskTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 90;
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

TEST_F(LinearModelMultiTaskTest, MultiTaskLassoFitPredict) {
    linear_model::MultiTaskLasso model(0.1);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelMultiTaskTest, MultiTaskLassoCVFitPredict) {
    linear_model::MultiTaskLassoCV model({0.05, 0.1, 0.5}, 3);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    EXPECT_GT(model.best_alpha(), 0.0);
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelMultiTaskTest, MultiTaskElasticNetFitPredict) {
    linear_model::MultiTaskElasticNet model(0.1, 0.5);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelMultiTaskTest, MultiTaskElasticNetCVFitPredict) {
    linear_model::MultiTaskElasticNetCV model({0.05, 0.1}, {0.2, 0.8}, 3);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    EXPECT_GT(model.best_alpha(), 0.0);
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelMultiTaskTest, MultiTaskElasticNetInvalidRatio) {
    linear_model::MultiTaskElasticNet model(0.1, 1.5);
    EXPECT_THROW(model.fit(X, y), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
