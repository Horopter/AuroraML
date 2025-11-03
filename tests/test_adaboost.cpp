#include <gtest/gtest.h>
#include "auroraml/adaboost.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <vector>

namespace auroraml {
namespace test {

class AdaBoostTest : public ::testing::Test {
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
        
        // Create multiclass data
        y_multiclass = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            double score = X(i, 0) + X(i, 1);
            if (score > 1.0) {
                y_multiclass(i) = 2.0;
            } else if (score > -1.0) {
                y_multiclass(i) = 1.0;
            } else {
                y_multiclass(i) = 0.0;
            }
        }
        
        // Create test data
        X_test = MatrixXd::Random(30, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y, y_classification, y_multiclass;
};

// AdaBoost Classifier Tests
TEST_F(AdaBoostTest, AdaBoostClassifierFit) {
    ensemble::AdaBoostClassifier adaboost(50, 1.0, 42);
    adaboost.fit(X, y_classification);
    
    EXPECT_TRUE(adaboost.is_fitted());
    
    std::vector<int> classes = adaboost.classes();
    EXPECT_GE(classes.size(), 2);
}

TEST_F(AdaBoostTest, AdaBoostClassifierPredictClasses) {
    ensemble::AdaBoostClassifier adaboost(50, 1.0, 42);
    adaboost.fit(X, y_classification);
    
    VectorXi y_pred = adaboost.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are valid class labels
    std::vector<int> classes = adaboost.classes();
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_TRUE(std::find(classes.begin(), classes.end(), y_pred(i)) != classes.end());
    }
}

TEST_F(AdaBoostTest, AdaBoostClassifierPredictProba) {
    ensemble::AdaBoostClassifier adaboost(50, 1.0, 42);
    adaboost.fit(X, y_classification);
    
    MatrixXd y_proba = adaboost.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
    
    // Check that probabilities are non-negative
    EXPECT_TRUE((y_proba.array() >= -1e-6).all());
    EXPECT_TRUE((y_proba.array() <= 1.0 + 1e-6).all());
}

TEST_F(AdaBoostTest, AdaBoostClassifierDecisionFunction) {
    ensemble::AdaBoostClassifier adaboost(50, 1.0, 42);
    adaboost.fit(X, y_classification);
    
    VectorXd decision = adaboost.decision_function(X_test);
    EXPECT_EQ(decision.size(), X_test.rows());
}

TEST_F(AdaBoostTest, AdaBoostClassifierParameters) {
    ensemble::AdaBoostClassifier adaboost(100, 0.5, 42);
    
    Params params = adaboost.get_params();
    EXPECT_EQ(params.at("n_estimators"), "100");
    EXPECT_EQ(params.at("learning_rate"), "0.500000");
    EXPECT_EQ(params.at("random_state"), "42");
    
    // Test set_params
    Params new_params = {{"n_estimators", "75"}, {"learning_rate", "0.8"}, {"random_state", "123"}};
    adaboost.set_params(new_params);
    Params updated = adaboost.get_params();
    EXPECT_EQ(updated.at("n_estimators"), "75");
}

TEST_F(AdaBoostTest, AdaBoostClassifierPerformance) {
    ensemble::AdaBoostClassifier adaboost(100, 1.0, 42);
    adaboost.fit(X, y_classification);
    
    VectorXi y_pred = adaboost.predict_classes(X);
    VectorXi y_true = y_classification.cast<int>();
    
    double accuracy = metrics::accuracy_score(y_true, y_pred);
    EXPECT_GT(accuracy, 0.7);  // Should achieve reasonable accuracy on training data
}

TEST_F(AdaBoostTest, AdaBoostClassifierDifferentLearningRates) {
    std::vector<double> learning_rates = {0.5, 1.0, 2.0};
    
    for (double lr : learning_rates) {
        ensemble::AdaBoostClassifier adaboost(50, lr, 42);
        adaboost.fit(X, y_classification);
        EXPECT_TRUE(adaboost.is_fitted());
        
        VectorXi y_pred = adaboost.predict_classes(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

TEST_F(AdaBoostTest, AdaBoostClassifierDifferentNEstimators) {
    std::vector<int> n_estimators = {10, 50, 100};
    
    for (int n_est : n_estimators) {
        ensemble::AdaBoostClassifier adaboost(n_est, 1.0, 42);
        adaboost.fit(X, y_classification);
        EXPECT_TRUE(adaboost.is_fitted());
        
        VectorXi y_pred = adaboost.predict_classes(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

// AdaBoost Regressor Tests
TEST_F(AdaBoostTest, AdaBoostRegressorFit) {
    ensemble::AdaBoostRegressor adaboost(50, 1.0, "linear", 42);
    adaboost.fit(X, y);
    
    EXPECT_TRUE(adaboost.is_fitted());
}

TEST_F(AdaBoostTest, AdaBoostRegressorPredict) {
    ensemble::AdaBoostRegressor adaboost(50, 1.0, "linear", 42);
    adaboost.fit(X, y);
    
    VectorXd y_pred = adaboost.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Predictions should be reasonable (not NaN or Inf)
    EXPECT_FALSE(y_pred.array().isNaN().any());
    EXPECT_FALSE(y_pred.array().isInf().any());
}

TEST_F(AdaBoostTest, AdaBoostRegressorPerformance) {
    ensemble::AdaBoostRegressor adaboost(100, 1.0, "linear", 42);
    adaboost.fit(X, y);
    
    VectorXd y_pred = adaboost.predict(X);
    
    double mse = metrics::mean_squared_error(y, y_pred);
    EXPECT_LT(mse, 10.0);  // Should achieve reasonable MSE on training data
    
    double r2 = metrics::r2_score(y, y_pred);
    EXPECT_GT(r2, 0.5);  // Should achieve reasonable R²
}

TEST_F(AdaBoostTest, AdaBoostRegressorDifferentLosses) {
    std::vector<std::string> losses = {"linear", "square", "exponential"};
    
    for (const std::string& loss : losses) {
        ensemble::AdaBoostRegressor adaboost(50, 1.0, loss, 42);
        adaboost.fit(X, y);
        EXPECT_TRUE(adaboost.is_fitted());
        
        VectorXd y_pred = adaboost.predict(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

TEST_F(AdaBoostTest, AdaBoostRegressorParameters) {
    ensemble::AdaBoostRegressor adaboost(100, 0.5, "linear", 42);
    
    Params params = adaboost.get_params();
    EXPECT_EQ(params.at("n_estimators"), "100");
    EXPECT_EQ(params.at("learning_rate"), "0.500000");
    EXPECT_EQ(params.at("loss"), "linear");
    EXPECT_EQ(params.at("random_state"), "42");
    
    // Test set_params
    Params new_params = {{"n_estimators", "75"}, {"learning_rate", "0.8"}, {"loss", "square"}, {"random_state", "123"}};
    adaboost.set_params(new_params);
    Params updated = adaboost.get_params();
    EXPECT_EQ(updated.at("n_estimators"), "75");
    EXPECT_EQ(updated.at("loss"), "square");
}

TEST_F(AdaBoostTest, AdaBoostRegressorDifferentLearningRates) {
    std::vector<double> learning_rates = {0.5, 1.0, 2.0};
    
    for (double lr : learning_rates) {
        ensemble::AdaBoostRegressor adaboost(50, lr, "linear", 42);
        adaboost.fit(X, y);
        EXPECT_TRUE(adaboost.is_fitted());
        
        VectorXd y_pred = adaboost.predict(X_test);
        EXPECT_EQ(y_pred.size(), X_test.rows());
    }
}

TEST_F(AdaBoostTest, AdaBoostRegressorConsistency) {
    ensemble::AdaBoostRegressor adaboost1(50, 1.0, "linear", 42);
    ensemble::AdaBoostRegressor adaboost2(50, 1.0, "linear", 42);
    
    adaboost1.fit(X, y);
    adaboost2.fit(X, y);
    
    VectorXd pred1 = adaboost1.predict(X_test);
    VectorXd pred2 = adaboost2.predict(X_test);
    
    // With same random seed, predictions should be similar (may have small numerical differences)
    for (int i = 0; i < pred1.size(); ++i) {
        EXPECT_NEAR(pred1(i), pred2(i), 1e-5);
    }
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

