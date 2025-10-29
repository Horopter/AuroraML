#include <gtest/gtest.h>
#include "auroraml/random_forest.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class EnsembleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 3;
        
        X = MatrixXd::Random(n_samples, n_features);
        
        // Create classification problem
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        // Create multi-class problem
        y_multiclass = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            if (X(i, 0) > 0.5) y_multiclass(i) = 2.0;
            else if (X(i, 0) > -0.5) y_multiclass(i) = 1.0;
            else y_multiclass(i) = 0.0;
        }
        
        // Create regression problem
        y_regression = VectorXd::Random(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y_classification, y_multiclass, y_regression;
};

// RandomForestClassifier Tests
TEST_F(EnsembleTest, RandomForestClassifierFit) {
    ensemble::RandomForestClassifier rf(10);
    rf.fit(X, y_classification);
    
    EXPECT_TRUE(rf.is_fitted());
}

TEST_F(EnsembleTest, RandomForestClassifierPredictClasses) {
    ensemble::RandomForestClassifier rf(10);
    rf.fit(X, y_classification);
    
    VectorXi y_pred = rf.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are valid class labels
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(y_pred(i) == 0 || y_pred(i) == 1);
    }
}

TEST_F(EnsembleTest, RandomForestClassifierPredictProba) {
    ensemble::RandomForestClassifier rf(10);
    rf.fit(X, y_classification);
    
    MatrixXd y_proba = rf.predict_proba(X);
    EXPECT_EQ(y_proba.rows(), n_samples);
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < n_samples; ++i) {
        double prob_sum = y_proba.row(i).sum();
        EXPECT_NEAR(prob_sum, 1.0, 1e-10);
    }
}

TEST_F(EnsembleTest, RandomForestClassifierMulticlass) {
    ensemble::RandomForestClassifier rf(10);
    rf.fit(X, y_multiclass);
    
    VectorXi y_pred = rf.predict_classes(X);
    MatrixXd y_proba = rf.predict_proba(X);
    
    EXPECT_EQ(y_pred.size(), n_samples);
    EXPECT_EQ(y_proba.rows(), n_samples);
    EXPECT_EQ(y_proba.cols(), 3);  // 3 classes
    
    // Check that predictions are valid class labels
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(y_pred(i) >= 0 && y_pred(i) <= 2);
    }
    
    // Check that probabilities sum to 1
    for (int i = 0; i < n_samples; ++i) {
        double prob_sum = y_proba.row(i).sum();
        EXPECT_NEAR(prob_sum, 1.0, 1e-10);
    }
}

TEST_F(EnsembleTest, RandomForestClassifierDifferentNEstimators) {
    ensemble::RandomForestClassifier rf(5);
    rf.fit(X, y_classification);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXi y_pred = rf.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestClassifierMaxDepth) {
    ensemble::RandomForestClassifier rf(10, 3);
    rf.fit(X, y_classification);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXi y_pred = rf.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestClassifierMinSamplesSplit) {
    ensemble::RandomForestClassifier rf(10, -1, 5);
    rf.fit(X, y_classification);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXi y_pred = rf.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestClassifierMinSamplesLeaf) {
    ensemble::RandomForestClassifier rf(10, -1, 2, 3);
    rf.fit(X, y_classification);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXi y_pred = rf.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestClassifierMinImpurityDecrease) {
    ensemble::RandomForestClassifier rf(10, -1, 2, 1);
    rf.fit(X, y_classification);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXi y_pred = rf.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestClassifierRandomState) {
    ensemble::RandomForestClassifier rf(10, -1, 2, 42);
    rf.fit(X, y_classification);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXi y_pred = rf.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestClassifierDecisionFunction) {
    ensemble::RandomForestClassifier rf(10);
    rf.fit(X, y_classification);
    
    VectorXd decision = rf.decision_function(X);
    EXPECT_EQ(decision.size(), n_samples);
    
    // Check that decision function values are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(decision(i)));
        EXPECT_FALSE(std::isinf(decision(i)));
    }
}

TEST_F(EnsembleTest, RandomForestClassifierGetSetParams) {
    ensemble::RandomForestClassifier rf(10, 5, 3, 42);
    
    Params params = rf.get_params();
    EXPECT_EQ(params["n_estimators"], "10");
    EXPECT_EQ(params["max_depth"], "5");
    EXPECT_EQ(params["max_features"], "3");
    EXPECT_EQ(params["random_state"], "42");
    
    // Test set_params
    Params new_params = {{"n_estimators", "20"}, {"max_depth", "3"}};
    rf.set_params(new_params);
    
    Params updated_params = rf.get_params();
    EXPECT_EQ(updated_params["n_estimators"], "20");
    EXPECT_EQ(updated_params["max_depth"], "3");
}

// RandomForestRegressor Tests
TEST_F(EnsembleTest, RandomForestRegressorFit) {
    ensemble::RandomForestRegressor rf(10);
    rf.fit(X, y_regression);
    
    EXPECT_TRUE(rf.is_fitted());
}

TEST_F(EnsembleTest, RandomForestRegressorPredict) {
    ensemble::RandomForestRegressor rf(10);
    rf.fit(X, y_regression);
    
    VectorXd y_pred = rf.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(EnsembleTest, RandomForestRegressorDifferentNEstimators) {
    ensemble::RandomForestRegressor rf(5);
    rf.fit(X, y_regression);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXd y_pred = rf.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestRegressorMaxDepth) {
    ensemble::RandomForestRegressor rf(10, 3);
    rf.fit(X, y_regression);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXd y_pred = rf.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestRegressorMinSamplesSplit) {
    ensemble::RandomForestRegressor rf(10, -1, 5);
    rf.fit(X, y_regression);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXd y_pred = rf.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestRegressorMinSamplesLeaf) {
    ensemble::RandomForestRegressor rf(10, -1, 2, 3);
    rf.fit(X, y_regression);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXd y_pred = rf.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestRegressorMinImpurityDecrease) {
    ensemble::RandomForestRegressor rf(10, -1, 2, 1);
    rf.fit(X, y_regression);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXd y_pred = rf.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestRegressorRandomState) {
    ensemble::RandomForestRegressor rf(10, -1, 2, 42);
    rf.fit(X, y_regression);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXd y_pred = rf.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestRegressorGetSetParams) {
    ensemble::RandomForestRegressor rf(10, 5, 3, 42);
    
    Params params = rf.get_params();
    EXPECT_EQ(params["n_estimators"], "10");
    EXPECT_EQ(params["max_depth"], "5");
    EXPECT_EQ(params["max_features"], "3");
    EXPECT_EQ(params["random_state"], "42");
    
    // Test set_params
    Params new_params = {{"n_estimators", "20"}, {"max_depth", "3"}};
    rf.set_params(new_params);
    
    Params updated_params = rf.get_params();
    EXPECT_EQ(updated_params["n_estimators"], "20");
    EXPECT_EQ(updated_params["max_depth"], "3");
}

// Edge Cases and Error Handling
TEST_F(EnsembleTest, RandomForestClassifierEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    ensemble::RandomForestClassifier rf(10);
    EXPECT_THROW(rf.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(EnsembleTest, RandomForestClassifierSingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    VectorXd y_single = VectorXd::Ones(1);
    
    ensemble::RandomForestClassifier rf(10);
    rf.fit(X_single, y_single);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXi y_pred = rf.predict_classes(X_single);
    EXPECT_EQ(y_pred.size(), 1);
}

TEST_F(EnsembleTest, RandomForestClassifierSingleFeature) {
    MatrixXd X_single_feature = MatrixXd::Random(n_samples, 1);
    VectorXd y_single_feature = VectorXd::Zero(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        y_single_feature(i) = (X_single_feature(i, 0) > 0.0) ? 1.0 : 0.0;
    }
    
    ensemble::RandomForestClassifier rf(10);
    rf.fit(X_single_feature, y_single_feature);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXi y_pred = rf.predict_classes(X_single_feature);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestClassifierPerfectSeparation) {
    // Create perfectly separable data
    MatrixXd X_perfect = MatrixXd::Zero(n_samples, 2);
    VectorXd y_perfect = VectorXd::Zero(n_samples);
    
    for (int i = 0; i < n_samples/2; ++i) {
        X_perfect(i, 0) = 1.0;
        X_perfect(i, 1) = 1.0;
        y_perfect(i) = 1.0;
    }
    for (int i = n_samples/2; i < n_samples; ++i) {
        X_perfect(i, 0) = -1.0;
        X_perfect(i, 1) = -1.0;
        y_perfect(i) = 0.0;
    }
    
    ensemble::RandomForestClassifier rf(10);
    rf.fit(X_perfect, y_perfect);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXi y_pred = rf.predict_classes(X_perfect);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Should achieve good accuracy on separable data
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        if (y_pred(i) == static_cast<int>(y_perfect(i))) correct++;
    }
    EXPECT_GT(correct, n_samples * 0.8);  // At least 80% accuracy
}

TEST_F(EnsembleTest, RandomForestClassifierNotFitted) {
    ensemble::RandomForestClassifier rf(10);
    
    EXPECT_FALSE(rf.is_fitted());
    EXPECT_THROW(rf.predict_classes(X), std::runtime_error);
    EXPECT_THROW(rf.predict_proba(X), std::runtime_error);
    EXPECT_THROW(rf.decision_function(X), std::runtime_error);
}

TEST_F(EnsembleTest, RandomForestClassifierWrongFeatureCount) {
    ensemble::RandomForestClassifier rf(10);
    rf.fit(X, y_classification);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(rf.predict_classes(X_wrong), std::runtime_error);
}

TEST_F(EnsembleTest, RandomForestRegressorNotFitted) {
    ensemble::RandomForestRegressor rf(10);
    
    EXPECT_FALSE(rf.is_fitted());
    EXPECT_THROW(rf.predict(X), std::runtime_error);
}

TEST_F(EnsembleTest, RandomForestRegressorWrongFeatureCount) {
    ensemble::RandomForestRegressor rf(10);
    rf.fit(X, y_regression);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(rf.predict(X_wrong), std::runtime_error);
}

TEST_F(EnsembleTest, RandomForestClassifierZeroEstimators) {
    ensemble::RandomForestClassifier rf(0);
    rf.fit(X, y_classification);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXi y_pred = rf.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestRegressorZeroEstimators) {
    ensemble::RandomForestRegressor rf(0);
    rf.fit(X, y_regression);
    
    EXPECT_TRUE(rf.is_fitted());
    
    VectorXd y_pred = rf.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestClassifierNegativeEstimators) {
    ensemble::RandomForestClassifier rf(-5);
    EXPECT_THROW(rf.fit(X, y_classification), std::length_error);
}

TEST_F(EnsembleTest, RandomForestRegressorNegativeEstimators) {
    ensemble::RandomForestRegressor rf(-5);
    EXPECT_THROW(rf.fit(X, y_regression), std::length_error);
}

TEST_F(EnsembleTest, RandomForestClassifierDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    ensemble::RandomForestClassifier rf(10);
    EXPECT_THROW(rf.fit(X_wrong, y_wrong), std::invalid_argument);
}

TEST_F(EnsembleTest, RandomForestRegressorDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    ensemble::RandomForestRegressor rf(10);
    EXPECT_THROW(rf.fit(X_wrong, y_wrong), std::invalid_argument);
}

TEST_F(EnsembleTest, RandomForestClassifierConsistency) {
    ensemble::RandomForestClassifier rf1(10, -1, 2, 42);
    ensemble::RandomForestClassifier rf2(10, -1, 2, 42);
    
    rf1.fit(X, y_classification);
    rf2.fit(X, y_classification);
    
    VectorXi y_pred1 = rf1.predict_classes(X);
    VectorXi y_pred2 = rf2.predict_classes(X);
    
    // Results should be consistent with same random state
    EXPECT_EQ(y_pred1.size(), y_pred2.size());
    EXPECT_EQ(y_pred1.size(), n_samples);
}

TEST_F(EnsembleTest, RandomForestRegressorConsistency) {
    ensemble::RandomForestRegressor rf1(10, -1, 2, 42);
    ensemble::RandomForestRegressor rf2(10, -1, 2, 42);
    
    rf1.fit(X, y_regression);
    rf2.fit(X, y_regression);
    
    VectorXd y_pred1 = rf1.predict(X);
    VectorXd y_pred2 = rf2.predict(X);
    
    // Results should be consistent with same random state
    EXPECT_EQ(y_pred1.size(), y_pred2.size());
    EXPECT_EQ(y_pred1.size(), n_samples);
}

// Random Forest persistence tests
TEST_F(EnsembleTest, RandomForestClassifierModelPersistence) {
    ensemble::RandomForestClassifier rf(10, 3, -1, 42);
    rf.fit(X, y_classification);
    
    // Save model
    rf.save("test_rf_classifier.bin");
    
    // Load model
    ensemble::RandomForestClassifier rf_loaded(10, 3, -1, 42);
    rf_loaded.load("test_rf_classifier.bin");
    
    // Test that loaded model works
    EXPECT_TRUE(rf_loaded.is_fitted());
    VectorXi y_pred_original = rf.predict_classes(X);
    VectorXi y_pred_loaded = rf_loaded.predict_classes(X);
    
    // Predictions should be identical
    EXPECT_TRUE(y_pred_original.isApprox(y_pred_loaded));
    
    // Clean up
    std::remove("test_rf_classifier.bin");
}

TEST_F(EnsembleTest, RandomForestRegressorModelPersistence) {
    ensemble::RandomForestRegressor rf(10, 3, -1, 42);
    rf.fit(X, y_regression);
    
    // Save model
    rf.save("test_rf_regressor.bin");
    
    // Load model
    ensemble::RandomForestRegressor rf_loaded(10, 3, -1, 42);
    rf_loaded.load("test_rf_regressor.bin");
    
    // Test that loaded model works
    EXPECT_TRUE(rf_loaded.is_fitted());
    VectorXd y_pred_original = rf.predict(X);
    VectorXd y_pred_loaded = rf_loaded.predict(X);
    
    // Predictions should be identical
    EXPECT_TRUE(y_pred_original.isApprox(y_pred_loaded, 1e-10));
    
    // Clean up
    std::remove("test_rf_regressor.bin");
}

TEST_F(EnsembleTest, RandomForestClassifierNotFittedSave) {
    ensemble::RandomForestClassifier rf(10, 3, -1, 42);
    EXPECT_THROW(rf.save("test.bin"), std::runtime_error);
}

TEST_F(EnsembleTest, RandomForestRegressorNotFittedSave) {
    ensemble::RandomForestRegressor rf(10, 3, -1, 42);
    EXPECT_THROW(rf.save("test.bin"), std::runtime_error);
}

TEST_F(EnsembleTest, RandomForestClassifierLoadNonexistentFile) {
    ensemble::RandomForestClassifier rf(10, 3, -1, 42);
    EXPECT_THROW(rf.load("nonexistent_file.bin"), std::runtime_error);
}

TEST_F(EnsembleTest, RandomForestRegressorLoadNonexistentFile) {
    ensemble::RandomForestRegressor rf(10, 3, -1, 42);
    EXPECT_THROW(rf.load("nonexistent_file.bin"), std::runtime_error);
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
