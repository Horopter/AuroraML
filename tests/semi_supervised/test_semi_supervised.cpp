#include <gtest/gtest.h>
#include "auroraml/semi_supervised.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>

namespace auroraml {
namespace test {

class SemiSupervisedTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : -1.0; // -1 for unlabeled
            if (i % 3 == 0) {
                y_classification(i) = -1; // Make some unlabeled
            }
        }
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases
TEST_F(SemiSupervisedTest, LabelPropagationFit) {
    semi_supervised::LabelPropagation lp(20.0, 30, 1e-3);
    lp.fit(X, y_classification);
    
    EXPECT_TRUE(lp.is_fitted());
}

TEST_F(SemiSupervisedTest, LabelPropagationPredict) {
    semi_supervised::LabelPropagation lp(20.0, 30, 1e-3);
    lp.fit(X, y_classification);
    
    VectorXi y_pred = lp.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
}

TEST_F(SemiSupervisedTest, LabelSpreadingFit) {
    semi_supervised::LabelSpreading ls(0.2, 20.0, 30, 1e-3);
    ls.fit(X, y_classification);
    
    EXPECT_TRUE(ls.is_fitted());
}

TEST_F(SemiSupervisedTest, LabelSpreadingPredict) {
    semi_supervised::LabelSpreading ls(0.2, 20.0, 30, 1e-3);
    ls.fit(X, y_classification);
    
    VectorXi y_pred = ls.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
}

// Negative test cases
TEST_F(SemiSupervisedTest, LabelPropagationNotFitted) {
    semi_supervised::LabelPropagation lp(20.0);
    EXPECT_THROW(lp.predict_classes(X_test), std::runtime_error);
}

TEST_F(SemiSupervisedTest, LabelPropagationEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    semi_supervised::LabelPropagation lp(20.0);
    EXPECT_THROW(lp.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(SemiSupervisedTest, LabelSpreadingNotFitted) {
    semi_supervised::LabelSpreading ls(20.0);
    EXPECT_THROW(ls.predict_classes(X_test), std::runtime_error);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
