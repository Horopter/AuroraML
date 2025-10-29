#include <gtest/gtest.h>
#include "auroraml/gradient_boosting.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class GradientBoostingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 4;
        
        // Create regression data
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y(i) = X(i, 0) + 2.0 * X(i, 1) - 0.5 * X(i, 2) + 0.1 * X(i, 3) + 0.1 * (MatrixXd::Random(1, 1))(0, 0);
        }
        
        // Create classification data
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            double score = X(i, 0) + X(i, 1) - X(i, 2);
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
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y, y_classification, y_multiclass;
};

// Gradient Boosting Classifier Tests
TEST_F(GradientBoostingTest, GradientBoostingClassifierFit) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    gbc.fit(X, y_classification);
    
    EXPECT_TRUE(gbc.is_fitted());
    EXPECT_EQ(gbc.n_estimators(), 10);
    EXPECT_EQ(gbc.learning_rate(), 0.1);
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierPredictClasses) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    gbc.fit(X, y_classification);
    
    VectorXi y_pred = gbc.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    // Check that predictions are valid class labels
    std::vector<int> classes = gbc.classes();
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_TRUE(std::find(classes.begin(), classes.end(), y_pred(i)) != classes.end());
    }
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierPredictProba) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    gbc.fit(X, y_classification);
    
    MatrixXd y_proba = gbc.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-10);
    }
    
    // Check that probabilities are non-negative
    EXPECT_TRUE((y_proba.array() >= 0.0).all());
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierMulticlass) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    gbc.fit(X, y_multiclass);
    
    VectorXi y_pred = gbc.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    
    MatrixXd y_proba = gbc.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 3);  // 3 classes
    
    // Check that probabilities sum to 1
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-10);
    }
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierGetSetParams) {
    ensemble::GradientBoostingClassifier gbc(50, 0.2, 5);
    
    Params params = gbc.get_params();
    EXPECT_EQ(params["n_estimators"], "50");
    EXPECT_EQ(params["learning_rate"], "0.200000");
    EXPECT_EQ(params["max_depth"], "5");
    
    Params new_params = {{"n_estimators", "100"}, {"learning_rate", "0.05"}};
    gbc.set_params(new_params);
    
    Params updated_params = gbc.get_params();
    EXPECT_EQ(updated_params["n_estimators"], "100");
    EXPECT_EQ(updated_params["learning_rate"], "0.050000");
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierClasses) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    gbc.fit(X, y_multiclass);
    
    std::vector<int> classes = gbc.classes();
    EXPECT_EQ(classes.size(), 3);
    
    // Classes should be sorted
    for (int i = 1; i < classes.size(); ++i) {
        EXPECT_LE(classes[i-1], classes[i]);
    }
}

// Gradient Boosting Regressor Tests
TEST_F(GradientBoostingTest, GradientBoostingRegressorFit) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    gbr.fit(X, y);
    
    EXPECT_TRUE(gbr.is_fitted());
    EXPECT_EQ(gbr.n_estimators(), 10);
    EXPECT_EQ(gbr.learning_rate(), 0.1);
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorPredict) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    gbr.fit(X, y);
    
    VectorXd y_pred = gbr.predict(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
    EXPECT_TRUE(y_pred.allFinite());
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorGetSetParams) {
    ensemble::GradientBoostingRegressor gbr(50, 0.2, 5);
    
    Params params = gbr.get_params();
    EXPECT_EQ(params["n_estimators"], "50");
    EXPECT_EQ(params["learning_rate"], "0.200000");
    EXPECT_EQ(params["max_depth"], "5");
    
    Params new_params = {{"n_estimators", "100"}, {"learning_rate", "0.05"}};
    gbr.set_params(new_params);
    
    Params updated_params = gbr.get_params();
    EXPECT_EQ(updated_params["n_estimators"], "100");
    EXPECT_EQ(updated_params["learning_rate"], "0.050000");
}

// Error handling tests
TEST_F(GradientBoostingTest, GradientBoostingClassifierNotFitted) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    
    EXPECT_FALSE(gbc.is_fitted());
    EXPECT_THROW(gbc.predict_classes(X_test), std::runtime_error);
    EXPECT_THROW(gbc.predict_proba(X_test), std::runtime_error);
    EXPECT_THROW(gbc.classes(), std::runtime_error);
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorNotFitted) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    
    EXPECT_FALSE(gbr.is_fitted());
    EXPECT_THROW(gbr.predict(X_test), std::runtime_error);
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    EXPECT_THROW(gbc.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    EXPECT_THROW(gbr.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierSingleClass) {
    VectorXd y_single_class = VectorXd::Zero(n_samples);  // All same class
    
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    EXPECT_THROW(gbc.fit(X, y_single_class), std::invalid_argument);
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierWrongFeatureCount) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    gbc.fit(X, y_classification);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(gbc.predict_classes(X_wrong), std::invalid_argument);
    EXPECT_THROW(gbc.predict_proba(X_wrong), std::invalid_argument);
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorWrongFeatureCount) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    gbr.fit(X, y);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(gbr.predict(X_wrong), std::invalid_argument);
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3);
    EXPECT_THROW(gbc.fit(X_wrong, y_wrong), std::invalid_argument);
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3);
    EXPECT_THROW(gbr.fit(X_wrong, y_wrong), std::invalid_argument);
}

// Consistency tests
TEST_F(GradientBoostingTest, GradientBoostingClassifierConsistency) {
    ensemble::GradientBoostingClassifier gbc1(10, 0.1, 3, 2, 1, 0.0, 42);
    ensemble::GradientBoostingClassifier gbc2(10, 0.1, 3, 2, 1, 0.0, 42);
    
    gbc1.fit(X, y_classification);
    gbc2.fit(X, y_classification);
    
    VectorXi y_pred1 = gbc1.predict_classes(X_test);
    VectorXi y_pred2 = gbc2.predict_classes(X_test);
    
    // Results should be identical with same random seed
    EXPECT_TRUE(y_pred1.isApprox(y_pred2));
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorConsistency) {
    ensemble::GradientBoostingRegressor gbr1(10, 0.1, 3, 2, 1, 0.0, 42);
    ensemble::GradientBoostingRegressor gbr2(10, 0.1, 3, 2, 1, 0.0, 42);
    
    gbr1.fit(X, y);
    gbr2.fit(X, y);
    
    VectorXd y_pred1 = gbr1.predict(X_test);
    VectorXd y_pred2 = gbr2.predict(X_test);
    
    // Results should be identical with same random seed
    EXPECT_TRUE(y_pred1.isApprox(y_pred2, 1e-10));
}

// Performance tests
TEST_F(GradientBoostingTest, GradientBoostingClassifierPerformance) {
    ensemble::GradientBoostingClassifier gbc(20, 0.1, 3);
    gbc.fit(X, y_classification);
    
    VectorXi y_pred = gbc.predict_classes(X);
    
    // Check reasonable accuracy on training data
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        if (y_pred(i) == static_cast<int>(y_classification(i))) {
            correct++;
        }
    }
    double accuracy = static_cast<double>(correct) / n_samples;
    EXPECT_GT(accuracy, 0.7);  // Should achieve reasonable accuracy
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorPerformance) {
    ensemble::GradientBoostingRegressor gbr(20, 0.1, 3);
    gbr.fit(X, y);
    
    VectorXd y_pred = gbr.predict(X);
    
    // Check reasonable R² score on training data
    double mse = (y - y_pred).squaredNorm() / n_samples;
    double y_var = (y.array() - y.mean()).square().sum() / n_samples;
    double r2 = 1.0 - mse / y_var;
    EXPECT_GT(r2, 0.5);  // Should achieve reasonable R² score
}

// Gradient Boosting persistence tests
TEST_F(GradientBoostingTest, GradientBoostingClassifierModelPersistence) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3, 2, 1, 0.0, 42);
    gbc.fit(X, y_classification);
    
    // Save model
    gbc.save("test_gbc_classifier.bin");
    
    // Load model
    ensemble::GradientBoostingClassifier gbc_loaded(10, 0.1, 3, 2, 1, 0.0, 42);
    gbc_loaded.load("test_gbc_classifier.bin");
    
    // Test that loaded model works
    EXPECT_TRUE(gbc_loaded.is_fitted());
    VectorXi y_pred_original = gbc.predict_classes(X);
    VectorXi y_pred_loaded = gbc_loaded.predict_classes(X);
    
    // Predictions should be identical
    EXPECT_TRUE(y_pred_original.isApprox(y_pred_loaded));
    
    // Clean up
    std::remove("test_gbc_classifier.bin");
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorModelPersistence) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3, 2, 1, 0.0, 42);
    gbr.fit(X, y);
    
    // Save model
    gbr.save("test_gbr_regressor.bin");
    
    // Load model
    ensemble::GradientBoostingRegressor gbr_loaded(10, 0.1, 3, 2, 1, 0.0, 42);
    gbr_loaded.load("test_gbr_regressor.bin");
    
    // Test that loaded model works
    EXPECT_TRUE(gbr_loaded.is_fitted());
    VectorXd y_pred_original = gbr.predict(X);
    VectorXd y_pred_loaded = gbr_loaded.predict(X);
    
    // Predictions should be identical
    EXPECT_TRUE(y_pred_original.isApprox(y_pred_loaded, 1e-10));
    
    // Clean up
    std::remove("test_gbr_regressor.bin");
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierNotFittedSave) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3, 2, 1, 0.0, 42);
    EXPECT_THROW(gbc.save("test.bin"), std::runtime_error);
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorNotFittedSave) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3, 2, 1, 0.0, 42);
    EXPECT_THROW(gbr.save("test.bin"), std::runtime_error);
}

TEST_F(GradientBoostingTest, GradientBoostingClassifierLoadNonexistentFile) {
    ensemble::GradientBoostingClassifier gbc(10, 0.1, 3, 2, 1, 0.0, 42);
    EXPECT_THROW(gbc.load("nonexistent_file.bin"), std::runtime_error);
}

TEST_F(GradientBoostingTest, GradientBoostingRegressorLoadNonexistentFile) {
    ensemble::GradientBoostingRegressor gbr(10, 0.1, 3, 2, 1, 0.0, 42);
    EXPECT_THROW(gbr.load("nonexistent_file.bin"), std::runtime_error);
}

} // namespace test
} // namespace cxml
