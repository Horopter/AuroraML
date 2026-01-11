#include <gtest/gtest.h>
#include "ingenuityml/linear_model.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class LinearModelSparseTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 80;
        n_features = 6;
        X = MatrixXd::Random(n_samples, n_features);
        true_coef = VectorXd::Zero(n_features);
        true_coef(0) = 1.5;
        true_coef(2) = -2.0;
        y = X * true_coef + VectorXd::Random(n_samples) * 0.05;
        X_test = MatrixXd::Random(20, n_features);
    }

    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y, true_coef;
};

TEST_F(LinearModelSparseTest, LarsFitPredict) {
    linear_model::Lars model(3, true, 200, 1e-4);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    EXPECT_EQ(model.coef().size(), n_features);
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelSparseTest, LarsCVFitPredict) {
    linear_model::LarsCV model(3, true, 200, 1e-4);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    EXPECT_GT(model.best_n_nonzero_coefs(), 0);
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelSparseTest, LassoLarsFitPredict) {
    linear_model::LassoLars model(0.1);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelSparseTest, LassoLarsCVFitPredict) {
    linear_model::LassoLarsCV model({0.05, 0.1, 0.5}, 3);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    EXPECT_GT(model.best_alpha(), 0.0);
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelSparseTest, LassoLarsICFitPredict) {
    linear_model::LassoLarsIC model({0.05, 0.1, 0.5}, "aic");
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    EXPECT_GT(model.best_alpha(), 0.0);
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelSparseTest, OrthogonalMatchingPursuitFitPredict) {
    linear_model::OrthogonalMatchingPursuit model(3);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelSparseTest, OrthogonalMatchingPursuitCVFitPredict) {
    linear_model::OrthogonalMatchingPursuitCV model(3);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    EXPECT_GT(model.best_n_nonzero_coefs(), 0);
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelSparseTest, LassoLarsNegativeAlpha) {
    linear_model::LassoLars model(-1.0);
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
