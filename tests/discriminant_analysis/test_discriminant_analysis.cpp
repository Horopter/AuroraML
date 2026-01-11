#include <gtest/gtest.h>
#include "ingenuityml/discriminant_analysis.hpp"
#include "ingenuityml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>

namespace ingenuityml {
namespace test {

class DiscriminantAnalysisTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases
TEST_F(DiscriminantAnalysisTest, QuadraticDiscriminantAnalysisFit) {
    discriminant_analysis::QuadraticDiscriminantAnalysis qda;
    qda.fit(X, y_classification);
    
    EXPECT_TRUE(qda.is_fitted());
}

TEST_F(DiscriminantAnalysisTest, QuadraticDiscriminantAnalysisPredict) {
    discriminant_analysis::QuadraticDiscriminantAnalysis qda;
    qda.fit(X, y_classification);
    
    VectorXi y_pred = qda.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
}

TEST_F(DiscriminantAnalysisTest, QuadraticDiscriminantAnalysisPredictProba) {
    discriminant_analysis::QuadraticDiscriminantAnalysis qda;
    qda.fit(X, y_classification);
    
    MatrixXd y_proba = qda.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 2);
    
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_F(DiscriminantAnalysisTest, QuadraticDiscriminantAnalysisPerformance) {
    discriminant_analysis::QuadraticDiscriminantAnalysis qda;
    qda.fit(X, y_classification);
    
    VectorXi y_pred = qda.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, y_pred);
    EXPECT_GT(accuracy, 0.5);
}

// Negative test cases
TEST_F(DiscriminantAnalysisTest, QuadraticDiscriminantAnalysisNotFitted) {
    discriminant_analysis::QuadraticDiscriminantAnalysis qda;
    EXPECT_THROW(qda.predict_classes(X_test), std::runtime_error);
}

TEST_F(DiscriminantAnalysisTest, QuadraticDiscriminantAnalysisEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    discriminant_analysis::QuadraticDiscriminantAnalysis qda;
    EXPECT_THROW(qda.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(DiscriminantAnalysisTest, QuadraticDiscriminantAnalysisSingleClass) {
    VectorXd y_single = VectorXd::Zero(n_samples);
    
    discriminant_analysis::QuadraticDiscriminantAnalysis qda;
    EXPECT_THROW(qda.fit(X, y_single), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
