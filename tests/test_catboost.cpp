#include <gtest/gtest.h>
#include "auroraml/catboost.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <vector>

namespace auroraml {
namespace test {

class CatBoostTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 200;
        n_features = 4;
        
        // Create regression data
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y(i) = 2.0 * X(i, 0) + 1.5 * X(i, 1) - 0.8 * X(i, 2) + 0.1 * X(i, 3) + 0.05 * (MatrixXd::Random(1, 1))(0, 0);
        }
        
        // Create classification data
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            double score = X(i, 0) + X(i, 1) - 0.5 * X(i, 2);
            y_classification(i) = (score > 0.0) ? 1.0 : 0.0;
        }
        
        // Create test data
        X_test = MatrixXd::Random(30, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y, y_classification;
};

// CatBoost Classifier Tests
TEST_F(CatBoostTest, CatBoostClassifierFit) {
    ensemble::CatBoostClassifier catboost(50, 0.03, 6, 3.0, 32.0, 1.0, 42);
    catboost.fit(X, y_classification);
    
    EXPECT_TRUE(catboost.is_fitted());
    
    std::vector<int> classes = catboost.classes();
    EXPECT_GE(classes.size(), 2);
}

TEST_F(CatBoostTest, CatBoostClassifierPredictClasses) {
    ensemble::CatBoostClassifier catboost(50, 0.03, 6, 3.0, 32.0, 1.0, 42);
    catboost.fit(X, y_classification);
    
    VectorXi y_pred = catboost.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are valid class labels
    std::vector<int> classes = catboost.classes();
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_TRUE(std::find(classes.begin(), classes.end(), y_pred(i)) != classes.end());
    }
}

TEST_F(CatBoostTest, CatBoostClassifierPredictProba) {
    ensemble::CatBoostClassifier catboost(50, 0.03, 6, 3.0, 32.0, 1.0, 42);
    catboost.fit(X, y_classification);
    
    MatrixXd y_proba = catboost.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
    
    // Check that probabilities are in [0, 1]
    EXPECT_TRUE((y_proba.array() >= -1e-6).all());
    EXPECT_TRUE((y_proba.array() <= 1.0 + 1e-6).all());
}

TEST_F(CatBoostTest, CatBoostClassifierDecisionFunction) {
    ensemble::CatBoostClassifier catboost(50, 0.03, 6, 3.0, 32.0, 1.0, 42);
    catboost.fit(X, y_classification);
    
    VectorXd decision = catboost.decision_function(X_test);
    EXPECT_EQ(decision.size(), X_test.rows());
}

TEST_F(CatBoostTest, CatBoostClassifierParameters) {
    ensemble::CatBoostClassifier catboost(100, 0.05, 8, 5.0, 64.0, 0.5, 42);
    
    Params params = catboost.get_params();
    EXPECT_EQ(params.at("n_estimators"), "100");
    EXPECT_EQ(params.at("learning_rate"), "0.050000");
    EXPECT_EQ(params.at("max_depth"), "8");
    EXPECT_EQ(params.at("l2_leaf_reg"), "5.000000");
    EXPECT_EQ(params.at("border_count"), "64.000000");
    EXPECT_EQ(params.at("bagging_temperature"), "0.500000");
    EXPECT_EQ(params.at("random_state"), "42");
}

TEST_F(CatBoostTest, CatBoostClassifierPerformance) {
    ensemble::CatBoostClassifier catboost(100, 0.03, 6, 3.0, 32.0, 1.0, 42);
    catboost.fit(X, y_classification);
    
    VectorXi y_pred = catboost.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, y_pred);
    EXPECT_GT(accuracy, 0.7);  // Should achieve reasonable accuracy
}

TEST_F(CatBoostTest, CatBoostClassifierDifferentDepths) {
    std::vector<int> depths = {3, 6, 10};
    
    for (int depth : depths) {
        ensemble::CatBoostClassifier catboost(50, 0.03, depth, 3.0, 32.0, 1.0, 42);
        catboost.fit(X, y_classification);
        EXPECT_TRUE(catboost.is_fitted());
        
        VectorXi y_pred = catboost.predict_classes(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

TEST_F(CatBoostTest, CatBoostClassifierL2Regularization) {
    // Test with different L2 regularization
    ensemble::CatBoostClassifier catboost_low(50, 0.03, 6, 1.0, 32.0, 1.0, 42);
    ensemble::CatBoostClassifier catboost_high(50, 0.03, 6, 10.0, 32.0, 1.0, 42);
    
    catboost_low.fit(X, y_classification);
    catboost_high.fit(X, y_classification);
    
    EXPECT_TRUE(catboost_low.is_fitted());
    EXPECT_TRUE(catboost_high.is_fitted());
    
    VectorXi pred_low = catboost_low.predict_classes(X_test);
    VectorXi pred_high = catboost_high.predict_classes(X_test);
    
    EXPECT_EQ(pred_low.size(), X_test.rows());
    EXPECT_EQ(pred_high.size(), X_test.rows());
}

// CatBoost Regressor Tests
TEST_F(CatBoostTest, CatBoostRegressorFit) {
    ensemble::CatBoostRegressor catboost(50, 0.03, 6, 3.0, 32.0, 1.0, 42);
    catboost.fit(X, y);
    
    EXPECT_TRUE(catboost.is_fitted());
}

TEST_F(CatBoostTest, CatBoostRegressorPredict) {
    ensemble::CatBoostRegressor catboost(50, 0.03, 6, 3.0, 32.0, 1.0, 42);
    catboost.fit(X, y);
    
    VectorXd y_pred = catboost.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Predictions should be reasonable
    EXPECT_FALSE(y_pred.array().isNaN().any());
    EXPECT_FALSE(y_pred.array().isInf().any());
}

TEST_F(CatBoostTest, CatBoostRegressorPerformance) {
    ensemble::CatBoostRegressor catboost(100, 0.03, 6, 3.0, 32.0, 1.0, 42);
    catboost.fit(X, y);
    
    VectorXd y_pred = catboost.predict(X);
    
    double mse = metrics::mean_squared_error(y, y_pred);
    EXPECT_LT(mse, 5.0);  // Should achieve reasonable MSE
    
    double r2 = metrics::r2_score(y, y_pred);
    EXPECT_GT(r2, 0.6);  // Should achieve reasonable R²
}

TEST_F(CatBoostTest, CatBoostRegressorParameters) {
    ensemble::CatBoostRegressor catboost(100, 0.05, 8, 5.0, 64.0, 0.5, 42);
    
    Params params = catboost.get_params();
    EXPECT_EQ(params.at("n_estimators"), "100");
    EXPECT_EQ(params.at("learning_rate"), "0.050000");
    EXPECT_EQ(params.at("max_depth"), "8");
    EXPECT_EQ(params.at("l2_leaf_reg"), "5.000000");
}

TEST_F(CatBoostTest, CatBoostRegressorDifferentLearningRates) {
    std::vector<double> learning_rates = {0.01, 0.03, 0.1};
    
    for (double lr : learning_rates) {
        ensemble::CatBoostRegressor catboost(50, lr, 6, 3.0, 32.0, 1.0, 42);
        catboost.fit(X, y);
        EXPECT_TRUE(catboost.is_fitted());
        
        VectorXd y_pred = catboost.predict(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

TEST_F(CatBoostTest, CatBoostRegressorL2Regularization) {
    // Test with different L2 regularization
    ensemble::CatBoostRegressor catboost_low(50, 0.03, 6, 1.0, 32.0, 1.0, 42);
    ensemble::CatBoostRegressor catboost_high(50, 0.03, 6, 10.0, 32.0, 1.0, 42);
    
    catboost_low.fit(X, y);
    catboost_high.fit(X, y);
    
    EXPECT_TRUE(catboost_low.is_fitted());
    EXPECT_TRUE(catboost_high.is_fitted());
    
    VectorXd pred_low = catboost_low.predict(X_test);
    VectorXd pred_high = catboost_high.predict(X_test);
    
    EXPECT_EQ(pred_low.size(), X_test.rows());
    EXPECT_EQ(pred_high.size(), X_test.rows());
}

TEST_F(CatBoostTest, CatBoostRegressorConsistency) {
    ensemble::CatBoostRegressor catboost1(50, 0.03, 6, 3.0, 32.0, 1.0, 42);
    ensemble::CatBoostRegressor catboost2(50, 0.03, 6, 3.0, 32.0, 1.0, 42);
    
    catboost1.fit(X, y);
    catboost2.fit(X, y);
    
    VectorXd pred1 = catboost1.predict(X_test);
    VectorXd pred2 = catboost2.predict(X_test);
    
    // With same seed, predictions should be similar
    for (int i = 0; i < pred1.size(); ++i) {
        EXPECT_NEAR(pred1(i), pred2(i), 1e-3);
    }
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

