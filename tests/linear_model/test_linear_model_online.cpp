#include <gtest/gtest.h>
#include "ingenuityml/linear_model.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class LinearModelOnlineTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 120;
        n_features = 5;
        X = MatrixXd::Random(n_samples, n_features);
        true_coef = VectorXd::Random(n_features);
        y_reg = X * true_coef + VectorXd::Random(n_samples) * 0.1;
        y_cls = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_cls(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        X_test = MatrixXd::Random(20, n_features);
    }

    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_reg, y_cls, true_coef;
};

TEST_F(LinearModelOnlineTest, SGDRegressorFitPredict) {
    linear_model::SGDRegressor model("squared_loss", "l2", 0.0001, 0.15, true, 200, 1e-3, "invscaling", 0.01, 0.5, true, 42, 0.1);
    model.fit(X, y_reg);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelOnlineTest, SGDClassifierFitPredict) {
    linear_model::SGDClassifier model("hinge", "l2", 0.0001, 0.15, true, 200, 1e-3, "invscaling", 0.01, 0.5, true, 42);
    model.fit(X, y_cls);
    EXPECT_TRUE(model.is_fitted());
    VectorXi preds = model.predict_classes(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
    for (int i = 0; i < preds.size(); ++i) {
        EXPECT_TRUE(preds(i) == 0 || preds(i) == 1);
    }
}

TEST_F(LinearModelOnlineTest, PassiveAggressiveRegressorFitPredict) {
    linear_model::PassiveAggressiveRegressor model(1.0, 0.1, true, 200, true, 42, "epsilon_insensitive");
    model.fit(X, y_reg);
    EXPECT_TRUE(model.is_fitted());
    VectorXd preds = model.predict(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelOnlineTest, PassiveAggressiveClassifierFitPredict) {
    linear_model::PassiveAggressiveClassifier model(1.0, true, 200, true, 42);
    model.fit(X, y_cls);
    EXPECT_TRUE(model.is_fitted());
    VectorXi preds = model.predict_classes(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelOnlineTest, PerceptronFitPredict) {
    linear_model::Perceptron model(true, 200, 1e-3, true, 42);
    model.fit(X, y_cls);
    EXPECT_TRUE(model.is_fitted());
    VectorXi preds = model.predict_classes(X_test);
    EXPECT_EQ(preds.size(), X_test.rows());
}

TEST_F(LinearModelOnlineTest, SGDRegressorNegativeAlpha) {
    linear_model::SGDRegressor model("squared_loss", "l2", -0.1);
    EXPECT_THROW(model.fit(X, y_reg), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
