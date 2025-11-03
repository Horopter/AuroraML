#include <gtest/gtest.h>
#include "auroraml/xgboost.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <vector>

namespace auroraml {
namespace test {

class XGBoostTest : public ::testing::Test {
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

// XGBoost Classifier Tests
TEST_F(XGBoostTest, XGBClassifierFit) {
    ensemble::XGBClassifier xgb(50, 0.1, 6, 0.0, 0.0, 1.0, 1, 1.0, 1.0, 42);
    xgb.fit(X, y_classification);
    
    EXPECT_TRUE(xgb.is_fitted());
    
    std::vector<int> classes = xgb.classes();
    EXPECT_GE(classes.size(), 2);
}

TEST_F(XGBoostTest, XGBClassifierPredictClasses) {
    ensemble::XGBClassifier xgb(50, 0.1, 6, 0.0, 0.0, 1.0, 1, 1.0, 1.0, 42);
    xgb.fit(X, y_classification);
    
    VectorXi y_pred = xgb.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are valid class labels
    std::vector<int> classes = xgb.classes();
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_TRUE(std::find(classes.begin(), classes.end(), y_pred(i)) != classes.end());
    }
}

TEST_F(XGBoostTest, XGBClassifierPredictProba) {
    ensemble::XGBClassifier xgb(50, 0.1, 6, 0.0, 0.0, 1.0, 1, 1.0, 1.0, 42);
    xgb.fit(X, y_classification);
    
    MatrixXd y_proba = xgb.predict_proba(X_test);
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

TEST_F(XGBoostTest, XGBClassifierDecisionFunction) {
    ensemble::XGBClassifier xgb(50, 0.1, 6, 0.0, 0.0, 1.0, 1, 1.0, 1.0, 42);
    xgb.fit(X, y_classification);
    
    VectorXd decision = xgb.decision_function(X_test);
    EXPECT_EQ(decision.size(), X_test.rows());
}

TEST_F(XGBoostTest, XGBClassifierParameters) {
    ensemble::XGBClassifier xgb(100, 0.05, 5, 0.1, 0.5, 2.0, 2, 0.8, 0.8, 42);
    
    Params params = xgb.get_params();
    EXPECT_EQ(params.at("n_estimators"), "100");
    EXPECT_EQ(params.at("learning_rate"), "0.050000");
    EXPECT_EQ(params.at("max_depth"), "5");
    EXPECT_EQ(params.at("gamma"), "0.100000");
    EXPECT_EQ(params.at("reg_alpha"), "0.500000");
    EXPECT_EQ(params.at("reg_lambda"), "2.000000");
    EXPECT_EQ(params.at("min_child_weight"), "2");
    EXPECT_EQ(params.at("subsample"), "0.800000");
    EXPECT_EQ(params.at("colsample_bytree"), "0.800000");
    EXPECT_EQ(params.at("random_state"), "42");
}

TEST_F(XGBoostTest, XGBClassifierPerformance) {
    ensemble::XGBClassifier xgb(100, 0.1, 6, 0.0, 0.0, 1.0, 1, 1.0, 1.0, 42);
    xgb.fit(X, y_classification);
    
    VectorXi y_pred = xgb.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, y_pred);
    EXPECT_GT(accuracy, 0.7);  // Should achieve reasonable accuracy
}

TEST_F(XGBoostTest, XGBClassifierRegularization) {
    // Test with different regularization parameters
    ensemble::XGBClassifier xgb_low_reg(50, 0.1, 6, 0.0, 0.0, 0.1, 1, 1.0, 1.0, 42);
    ensemble::XGBClassifier xgb_high_reg(50, 0.1, 6, 0.0, 0.0, 10.0, 1, 1.0, 1.0, 42);
    
    xgb_low_reg.fit(X, y_classification);
    xgb_high_reg.fit(X, y_classification);
    
    EXPECT_TRUE(xgb_low_reg.is_fitted());
    EXPECT_TRUE(xgb_high_reg.is_fitted());
    
    VectorXi pred_low = xgb_low_reg.predict_classes(X_test);
    VectorXi pred_high = xgb_high_reg.predict_classes(X_test);
    
    EXPECT_EQ(pred_low.size(), X_test.rows());
    EXPECT_EQ(pred_high.size(), X_test.rows());
}

TEST_F(XGBoostTest, XGBClassifierSubsampling) {
    // Test with subsampling
    ensemble::XGBClassifier xgb(50, 0.1, 6, 0.0, 0.0, 1.0, 1, 0.5, 1.0, 42);
    xgb.fit(X, y_classification);
    
    EXPECT_TRUE(xgb.is_fitted());
    VectorXi y_pred = xgb.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
}

// XGBoost Regressor Tests
TEST_F(XGBoostTest, XGBRegressorFit) {
    ensemble::XGBRegressor xgb(50, 0.1, 6, 0.0, 0.0, 1.0, 1, 1.0, 1.0, 42);
    xgb.fit(X, y);
    
    EXPECT_TRUE(xgb.is_fitted());
}

TEST_F(XGBoostTest, XGBRegressorPredict) {
    ensemble::XGBRegressor xgb(50, 0.1, 6, 0.0, 0.0, 1.0, 1, 1.0, 1.0, 42);
    xgb.fit(X, y);
    
    VectorXd y_pred = xgb.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Predictions should be reasonable
    EXPECT_FALSE(y_pred.array().isNaN().any());
    EXPECT_FALSE(y_pred.array().isInf().any());
}

TEST_F(XGBoostTest, XGBRegressorPerformance) {
    ensemble::XGBRegressor xgb(100, 0.1, 6, 0.0, 0.0, 1.0, 1, 1.0, 1.0, 42);
    xgb.fit(X, y);
    
    VectorXd y_pred = xgb.predict(X);
    
    double mse = metrics::mean_squared_error(y, y_pred);
    EXPECT_LT(mse, 5.0);  // Should achieve reasonable MSE
    
    double r2 = metrics::r2_score(y, y_pred);
    EXPECT_GT(r2, 0.6);  // Should achieve reasonable R²
}

TEST_F(XGBoostTest, XGBRegressorParameters) {
    ensemble::XGBRegressor xgb(100, 0.05, 5, 0.1, 0.5, 2.0, 2, 0.8, 0.8, 42);
    
    Params params = xgb.get_params();
    EXPECT_EQ(params.at("n_estimators"), "100");
    EXPECT_EQ(params.at("learning_rate"), "0.050000");
    EXPECT_EQ(params.at("max_depth"), "5");
}

TEST_F(XGBoostTest, XGBRegressorRegularization) {
    // Test with different regularization
    ensemble::XGBRegressor xgb_low(50, 0.1, 6, 0.0, 0.0, 0.1, 1, 1.0, 1.0, 42);
    ensemble::XGBRegressor xgb_high(50, 0.1, 6, 0.0, 0.0, 10.0, 1, 1.0, 1.0, 42);
    
    xgb_low.fit(X, y);
    xgb_high.fit(X, y);
    
    EXPECT_TRUE(xgb_low.is_fitted());
    EXPECT_TRUE(xgb_high.is_fitted());
    
    VectorXd pred_low = xgb_low.predict(X_test);
    VectorXd pred_high = xgb_high.predict(X_test);
    
    EXPECT_EQ(pred_low.size(), X_test.rows());
    EXPECT_EQ(pred_high.size(), X_test.rows());
}

TEST_F(XGBoostTest, XGBRegressorSubsampling) {
    ensemble::XGBRegressor xgb(50, 0.1, 6, 0.0, 0.0, 1.0, 1, 0.7, 0.7, 42);
    xgb.fit(X, y);
    
    EXPECT_TRUE(xgb.is_fitted());
    VectorXd y_pred = xgb.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
}

TEST_F(XGBoostTest, XGBRegressorConsistency) {
    ensemble::XGBRegressor xgb1(50, 0.1, 6, 0.0, 0.0, 1.0, 1, 1.0, 1.0, 42);
    ensemble::XGBRegressor xgb2(50, 0.1, 6, 0.0, 0.0, 1.0, 1, 1.0, 1.0, 42);
    
    xgb1.fit(X, y);
    xgb2.fit(X, y);
    
    VectorXd pred1 = xgb1.predict(X_test);
    VectorXd pred2 = xgb2.predict(X_test);
    
    // With same seed, predictions should be similar
    for (int i = 0; i < pred1.size(); ++i) {
        EXPECT_NEAR(pred1(i), pred2(i), 1e-4);
    }
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

