#include <gtest/gtest.h>
#include "auroraml/linear_model.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class LinearModelClassifierCVTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        X_test = MatrixXd::Random(20, n_features);
    }

    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y;
};

TEST_F(LinearModelClassifierCVTest, RidgeClassifierFitPredict) {
    linear_model::RidgeClassifier model(1.0);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    VectorXi preds = model.predict_classes(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
    MatrixXd proba = model.predict_proba(X_test);
    EXPECT_EQ(proba.rows(), X_test.rows());
    EXPECT_EQ(proba.cols(), 2);
}

TEST_F(LinearModelClassifierCVTest, RidgeClassifierCVFitPredict) {
    linear_model::RidgeClassifierCV model({0.1, 1.0, 10.0}, 3);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    EXPECT_GT(model.best_alpha(), 0.0);
    VectorXi preds = model.predict_classes(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelClassifierCVTest, LogisticRegressionCVFitPredict) {
    linear_model::LogisticRegressionCV model({0.1, 1.0, 10.0}, 3);
    model.fit(X, y);
    EXPECT_TRUE(model.is_fitted());
    VectorXi preds = model.predict_classes(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
    MatrixXd proba = model.predict_proba(X_test);
    EXPECT_EQ(proba.rows(), X_test.rows());
    EXPECT_EQ(proba.cols(), 2);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
