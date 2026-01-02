#include <gtest/gtest.h>
#include "auroraml/feature_selection.hpp"
#include "auroraml/linear_model.hpp"
#include "auroraml/model_selection.hpp"
#include <Eigen/Dense>
#include <functional>
#include <cmath>

namespace auroraml {
namespace test {

class FeatureSelectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 10;
        
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Random(n_samples);
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        y_dummy = VectorXd::Zero(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y, y_classification, y_dummy;
};

// Positive test cases
TEST_F(FeatureSelectionTest, VarianceThresholdFit) {
    feature_selection::VarianceThreshold vt(0.01);
    vt.fit(X, y_dummy);
    
    EXPECT_TRUE(vt.is_fitted());
}

TEST_F(FeatureSelectionTest, VarianceThresholdTransform) {
    feature_selection::VarianceThreshold vt(0.01);
    vt.fit(X, y_dummy);
    
    MatrixXd X_transformed = vt.transform(X);
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_LE(X_transformed.cols(), X.cols());
}

TEST_F(FeatureSelectionTest, VarianceThresholdGetSupport) {
    feature_selection::VarianceThreshold vt(0.01);
    vt.fit(X, y_dummy);
    
    std::vector<int> support = vt.get_support();
    EXPECT_LE(support.size(), n_features);
}

TEST_F(FeatureSelectionTest, SelectKBestFit) {
    auto score_func = [](const VectorXd& X_feature, const VectorXd& y) -> double {
        // Simple correlation-based score
        double mean_x = X_feature.mean();
        double mean_y = y.mean();
        double numerator = (X_feature.array() - mean_x).matrix().dot((y.array() - mean_y).matrix());
        double denom_x = (X_feature.array() - mean_x).matrix().norm();
        double denom_y = (y.array() - mean_y).matrix().norm();
        if (denom_x < 1e-10 || denom_y < 1e-10) return 0.0;
        return std::abs(numerator / (denom_x * denom_y));
    };
    
    feature_selection::SelectKBest selector(score_func, 3);
    selector.fit(X, y);
    
    EXPECT_TRUE(selector.is_fitted());
}

TEST_F(FeatureSelectionTest, SelectKBestTransform) {
    auto score_func = [](const VectorXd& X_feature, const VectorXd& y) -> double {
        double mean_x = X_feature.mean();
        double mean_y = y.mean();
        double numerator = (X_feature.array() - mean_x).matrix().dot((y.array() - mean_y).matrix());
        double denom_x = (X_feature.array() - mean_x).matrix().norm();
        double denom_y = (y.array() - mean_y).matrix().norm();
        if (denom_x < 1e-10 || denom_y < 1e-10) return 0.0;
        return std::abs(numerator / (denom_x * denom_y));
    };
    
    feature_selection::SelectKBest selector(score_func, 3);
    selector.fit(X, y);
    
    MatrixXd X_transformed = selector.transform(X);
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_EQ(X_transformed.cols(), 3);
}

TEST_F(FeatureSelectionTest, SelectFprFitTransform) {
    feature_selection::SelectFpr selector(feature_selection::scores::f_classif, 0.1);
    selector.fit(X, y_classification);

    EXPECT_TRUE(selector.is_fitted());
    MatrixXd X_transformed = selector.transform(X);
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_LE(X_transformed.cols(), X.cols());
    EXPECT_GT(X_transformed.cols(), 0);
    EXPECT_EQ(selector.scores().size(), X.cols());
}

TEST_F(FeatureSelectionTest, SelectFdrFitTransform) {
    feature_selection::SelectFdr selector(feature_selection::scores::f_classif, 0.1);
    selector.fit(X, y_classification);

    EXPECT_TRUE(selector.is_fitted());
    MatrixXd X_transformed = selector.transform(X);
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_LE(X_transformed.cols(), X.cols());
    EXPECT_GT(X_transformed.cols(), 0);
    EXPECT_EQ(selector.scores().size(), X.cols());
}

TEST_F(FeatureSelectionTest, SelectFweFitTransform) {
    feature_selection::SelectFwe selector(feature_selection::scores::f_classif, 0.1);
    selector.fit(X, y_classification);

    EXPECT_TRUE(selector.is_fitted());
    MatrixXd X_transformed = selector.transform(X);
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_LE(X_transformed.cols(), X.cols());
    EXPECT_GT(X_transformed.cols(), 0);
    EXPECT_EQ(selector.scores().size(), X.cols());
}

TEST_F(FeatureSelectionTest, GenericUnivariateSelectPercentile) {
    feature_selection::GenericUnivariateSelect selector(feature_selection::scores::f_classif, "percentile", 20.0);
    selector.fit(X, y_classification);

    MatrixXd X_transformed = selector.transform(X);
    int expected = std::max(1, static_cast<int>(std::ceil(n_features * 0.2)));
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_EQ(X_transformed.cols(), expected);
}

TEST_F(FeatureSelectionTest, SelectFromModelFitTransform) {
    linear_model::LinearRegression estimator;
    feature_selection::SelectFromModel selector(estimator, 0.0, 3);
    selector.fit(X, y);

    EXPECT_TRUE(selector.is_fitted());
    MatrixXd X_transformed = selector.transform(X);
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_LE(X_transformed.cols(), 3);
    EXPECT_GT(X_transformed.cols(), 0);
    EXPECT_EQ(selector.importances().size(), X.cols());
}

TEST_F(FeatureSelectionTest, RFEFitTransform) {
    linear_model::LinearRegression estimator;
    feature_selection::RFE selector(estimator, 3, 2);
    selector.fit(X, y);

    EXPECT_TRUE(selector.is_fitted());
    MatrixXd X_transformed = selector.transform(X);
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_EQ(X_transformed.cols(), 3);
}

TEST_F(FeatureSelectionTest, RFECVFitTransform) {
    MatrixXd X_small = MatrixXd::Random(30, 6);
    VectorXd y_small = VectorXd::Zero(30);
    for (int i = 0; i < 30; ++i) {
        y_small(i) = (X_small(i, 0) + X_small(i, 1) > 0.0) ? 1.0 : 0.0;
    }

    linear_model::LogisticRegression estimator;
    model_selection::KFold cv(3, true, 42);
    feature_selection::RFECV selector(estimator, cv, 1, "accuracy", 2);
    selector.fit(X_small, y_small);

    EXPECT_TRUE(selector.is_fitted());
    MatrixXd X_transformed = selector.transform(X_small);
    EXPECT_EQ(X_transformed.rows(), X_small.rows());
    EXPECT_GE(X_transformed.cols(), 2);
    EXPECT_LE(X_transformed.cols(), 6);
}

TEST_F(FeatureSelectionTest, SequentialFeatureSelectorFitTransform) {
    MatrixXd X_small = MatrixXd::Random(30, 6);
    VectorXd y_small = VectorXd::Zero(30);
    for (int i = 0; i < 30; ++i) {
        y_small(i) = (X_small(i, 0) + X_small(i, 1) > 0.0) ? 1.0 : 0.0;
    }

    linear_model::LogisticRegression estimator;
    model_selection::KFold cv(3, true, 42);
    feature_selection::SequentialFeatureSelector selector(estimator, cv, 3, "forward", "accuracy");
    selector.fit(X_small, y_small);

    EXPECT_TRUE(selector.is_fitted());
    MatrixXd X_transformed = selector.transform(X_small);
    EXPECT_EQ(X_transformed.rows(), X_small.rows());
    EXPECT_EQ(X_transformed.cols(), 3);
}

// Negative test cases
TEST_F(FeatureSelectionTest, VarianceThresholdNotFitted) {
    feature_selection::VarianceThreshold vt(0.01);
    EXPECT_THROW(vt.transform(X), std::runtime_error);
}

TEST_F(FeatureSelectionTest, VarianceThresholdEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    feature_selection::VarianceThreshold vt(0.01);
    EXPECT_THROW(vt.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(FeatureSelectionTest, SelectKBestNotFitted) {
    auto score_func = [](const VectorXd& X_feature, const VectorXd& y) -> double {
        return 1.0;
    };
    
    feature_selection::SelectKBest selector(score_func, 3);
    EXPECT_THROW(selector.transform(X), std::runtime_error);
}

TEST_F(FeatureSelectionTest, SelectKBestInvalidK) {
    auto score_func = [](const VectorXd& X_feature, const VectorXd& y) -> double {
        return 1.0;
    };
    
    feature_selection::SelectKBest selector(score_func, n_features + 1);
    EXPECT_THROW(selector.fit(X, y), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
