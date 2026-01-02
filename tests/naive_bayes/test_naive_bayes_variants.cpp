#include <gtest/gtest.h>
#include "auroraml/naive_bayes_variants.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>

namespace auroraml {
namespace test {

class NaiveBayesVariantsTest : public ::testing::Test {
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
TEST_F(NaiveBayesVariantsTest, MultinomialNBFit) {
    naive_bayes::MultinomialNB mnb(1.0);
    mnb.fit(X, y_classification);
    
    EXPECT_TRUE(mnb.is_fitted());
}

TEST_F(NaiveBayesVariantsTest, MultinomialNBPredict) {
    naive_bayes::MultinomialNB mnb(1.0);
    mnb.fit(X, y_classification);
    
    VectorXi y_pred = mnb.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
}

TEST_F(NaiveBayesVariantsTest, BernoulliNBFit) {
    naive_bayes::BernoulliNB bnb(1.0);
    bnb.fit(X, y_classification);
    
    EXPECT_TRUE(bnb.is_fitted());
}

TEST_F(NaiveBayesVariantsTest, BernoulliNBPredict) {
    naive_bayes::BernoulliNB bnb(1.0);
    bnb.fit(X, y_classification);
    
    VectorXi y_pred = bnb.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
}

TEST_F(NaiveBayesVariantsTest, ComplementNBFit) {
    naive_bayes::ComplementNB cnb(1.0);
    cnb.fit(X, y_classification);
    
    EXPECT_TRUE(cnb.is_fitted());
}

// Negative test cases
TEST_F(NaiveBayesVariantsTest, MultinomialNBNotFitted) {
    naive_bayes::MultinomialNB mnb(1.0);
    EXPECT_THROW(mnb.predict_classes(X_test), std::runtime_error);
}

TEST_F(NaiveBayesVariantsTest, MultinomialNBEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    naive_bayes::MultinomialNB mnb(1.0);
    EXPECT_THROW(mnb.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(NaiveBayesVariantsTest, BernoulliNBNotFitted) {
    naive_bayes::BernoulliNB bnb(1.0);
    EXPECT_THROW(bnb.predict_classes(X_test), std::runtime_error);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
