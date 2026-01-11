#include <gtest/gtest.h>
#include "ingenuityml/meta_estimators.hpp"
#include "ingenuityml/naive_bayes.hpp"
#include "ingenuityml/linear_model.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class MetaEstimatorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 80;
        n_features = 4;
        X = MatrixXd::Random(n_samples, n_features);

        y_class = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            int cls = 0;
            if (X(i, 0) > 0) cls += 1;
            if (X(i, 1) > 0) cls += 1;
            y_class(i) = cls;
        }

        Y_multi = MatrixXd::Zero(n_samples, 2);
        for (int i = 0; i < n_samples; ++i) {
            Y_multi(i, 0) = X(i, 0) > 0 ? 1.0 : 0.0;
            Y_multi(i, 1) = X(i, 1) > 0 ? 1.0 : 0.0;
        }

        Y_reg = MatrixXd::Zero(n_samples, 2);
        for (int i = 0; i < n_samples; ++i) {
            Y_reg(i, 0) = X.row(i).sum();
            Y_reg(i, 1) = X.row(i).mean();
        }
    }

    int n_samples;
    int n_features;
    MatrixXd X;
    VectorXd y_class;
    MatrixXd Y_multi;
    MatrixXd Y_reg;
};

TEST_F(MetaEstimatorsTest, OneVsRestClassifierFitPredict) {
    auto factory = []() { return std::make_shared<naive_bayes::GaussianNB>(); };
    meta::OneVsRestClassifier clf(factory);
    clf.fit(X, y_class);
    VectorXi preds = clf.predict_classes(X);
    EXPECT_EQ(preds.size(), n_samples);
}

TEST_F(MetaEstimatorsTest, OneVsOneClassifierFitPredict) {
    auto factory = []() { return std::make_shared<naive_bayes::GaussianNB>(); };
    meta::OneVsOneClassifier clf(factory);
    clf.fit(X, y_class);
    VectorXi preds = clf.predict_classes(X);
    EXPECT_EQ(preds.size(), n_samples);
}

TEST_F(MetaEstimatorsTest, OutputCodeClassifierFitPredict) {
    auto factory = []() { return std::make_shared<naive_bayes::GaussianNB>(); };
    meta::OutputCodeClassifier clf(factory, 5, 42);
    clf.fit(X, y_class);
    VectorXi preds = clf.predict_classes(X);
    EXPECT_EQ(preds.size(), n_samples);
}

TEST_F(MetaEstimatorsTest, MultiOutputClassifierFitPredict) {
    auto factory = []() { return std::make_shared<naive_bayes::GaussianNB>(); };
    meta::MultiOutputClassifier clf(factory);
    clf.fit(X, Y_multi);
    MatrixXi preds = clf.predict(X);
    EXPECT_EQ(preds.rows(), n_samples);
    EXPECT_EQ(preds.cols(), 2);
}

TEST_F(MetaEstimatorsTest, ClassifierChainFitPredict) {
    auto factory = []() { return std::make_shared<naive_bayes::GaussianNB>(); };
    meta::ClassifierChain clf(factory);
    clf.fit(X, Y_multi);
    MatrixXi preds = clf.predict(X);
    EXPECT_EQ(preds.rows(), n_samples);
    EXPECT_EQ(preds.cols(), 2);
}

TEST_F(MetaEstimatorsTest, MultiOutputRegressorFitPredict) {
    auto factory = []() { return std::make_shared<linear_model::LinearRegression>(); };
    meta::MultiOutputRegressor reg(factory);
    reg.fit(X, Y_reg);
    MatrixXd preds = reg.predict(X);
    EXPECT_EQ(preds.rows(), n_samples);
    EXPECT_EQ(preds.cols(), 2);
}

TEST_F(MetaEstimatorsTest, RegressorChainFitPredict) {
    auto factory = []() { return std::make_shared<linear_model::LinearRegression>(); };
    meta::RegressorChain reg(factory);
    reg.fit(X, Y_reg);
    MatrixXd preds = reg.predict(X);
    EXPECT_EQ(preds.rows(), n_samples);
    EXPECT_EQ(preds.cols(), 2);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
