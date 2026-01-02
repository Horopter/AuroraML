#include <gtest/gtest.h>
#include "auroraml/gradient_boosting.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>

namespace auroraml {
namespace test {

class GradientBoostingClassifierTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            double score = X(i, 0) + X(i, 1) - X(i, 2);
            y_classification(i) = (score > 0.0) ? 1.0 : 0.0;
        }
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases
TEST_F(GradientBoostingClassifierTest, GradientBoostingClassifierFit) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    gbc.fit(X, y_classification);
    
    EXPECT_TRUE(gbc.is_fitted());
    EXPECT_EQ(gbc.n_estimators(), 10);
}

TEST_F(GradientBoostingClassifierTest, GradientBoostingClassifierPredictClasses) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    gbc.fit(X, y_classification);
    
    VectorXi y_pred = gbc.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    std::vector<int> classes = gbc.classes();
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_TRUE(std::find(classes.begin(), classes.end(), y_pred(i)) != classes.end());
    }
}

TEST_F(GradientBoostingClassifierTest, GradientBoostingClassifierPredictProba) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    gbc.fit(X, y_classification);
    
    MatrixXd y_proba = gbc.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 2);
    
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-10);
    }
}

TEST_F(GradientBoostingClassifierTest, GradientBoostingClassifierPerformance) {
    ensemble::GradientBoostingClassifier gbc(20, 0.1, 3);
    gbc.fit(X, y_classification);
    
    VectorXi y_pred = gbc.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, y_pred);
    EXPECT_GT(accuracy, 0.7);
}

// Negative test cases
TEST_F(GradientBoostingClassifierTest, GradientBoostingClassifierNotFitted) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    
    EXPECT_FALSE(gbc.is_fitted());
    EXPECT_THROW(gbc.predict_classes(X_test), std::runtime_error);
}

TEST_F(GradientBoostingClassifierTest, GradientBoostingClassifierWrongFeatureCount) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    gbc.fit(X, y_classification);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(gbc.predict_classes(X_wrong), std::invalid_argument);
}

TEST_F(GradientBoostingClassifierTest, GradientBoostingClassifierEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    EXPECT_THROW(gbc.fit(X_empty, y_empty), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

