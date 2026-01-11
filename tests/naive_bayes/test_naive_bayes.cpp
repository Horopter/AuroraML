#include <gtest/gtest.h>
#include "ingenuityml/naive_bayes.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class NaiveBayesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 2;
        
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Zero(n_samples);
        
        // Create simple classification problem
        for (int i = 0; i < n_samples; ++i) {
            y(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y;
};

TEST_F(NaiveBayesTest, GaussianNBFit) {
    naive_bayes::GaussianNB gnb;
    gnb.fit(X, y);
    
    EXPECT_TRUE(gnb.is_fitted());
}

TEST_F(NaiveBayesTest, GaussianNBPredict) {
    naive_bayes::GaussianNB gnb;
    gnb.fit(X, y);
    
    VectorXi y_pred = gnb.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are valid class labels
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(y_pred(i) == 0 || y_pred(i) == 1);
    }
}

TEST_F(NaiveBayesTest, GaussianNBPredictProba) {
    naive_bayes::GaussianNB gnb;
    gnb.fit(X, y);
    
    MatrixXd y_proba = gnb.predict_proba(X);
    EXPECT_EQ(y_proba.rows(), n_samples);
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < n_samples; ++i) {
        double prob_sum = y_proba.row(i).sum();
        EXPECT_NEAR(prob_sum, 1.0, 1e-10);
    }
}

TEST_F(NaiveBayesTest, GaussianNBDecisionFunction) {
    naive_bayes::GaussianNB gnb;
    gnb.fit(X, y);
    
    VectorXd decision_values = gnb.decision_function(X);
    EXPECT_EQ(decision_values.size(), n_samples);
    
    // Check that decision values are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(decision_values(i)));
        EXPECT_FALSE(std::isinf(decision_values(i)));
    }
}

// Naive Bayes persistence tests
TEST_F(NaiveBayesTest, GaussianNBModelPersistence) {
    naive_bayes::GaussianNB nb(1e-9);
    nb.fit(X, y);
    
    // Save model
    nb.save("test_nb_classifier.bin");
    
    // Load model
    naive_bayes::GaussianNB nb_loaded(1e-9);
    nb_loaded.load("test_nb_classifier.bin");
    
    // Test that loaded model works
    EXPECT_TRUE(nb_loaded.is_fitted());
    VectorXi y_pred_original = nb.predict_classes(X);
    VectorXi y_pred_loaded = nb_loaded.predict_classes(X);
    
    // Predictions should be identical
    EXPECT_TRUE(y_pred_original.isApprox(y_pred_loaded));
    
    // Clean up
    std::remove("test_nb_classifier.bin");
}

TEST_F(NaiveBayesTest, GaussianNBNotFittedSave) {
    naive_bayes::GaussianNB nb(1e-9);
    EXPECT_THROW(nb.save("test.bin"), std::runtime_error);
}

TEST_F(NaiveBayesTest, GaussianNBLoadNonexistentFile) {
    naive_bayes::GaussianNB nb(1e-9);
    EXPECT_THROW(nb.load("nonexistent_file.bin"), std::runtime_error);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    // Enable test shuffling within this file
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;  // Reproducible shuffle
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
