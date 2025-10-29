#include <gtest/gtest.h>
#include "auroraml/tree.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class TreeTest : public ::testing::Test {
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
        
        // Create regression problem
        y_regression = VectorXd::Random(n_samples);
        
        // Create multi-class problem
        y_multiclass = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            if (X(i, 0) > 0.5) y_multiclass(i) = 2.0;
            else if (X(i, 0) > -0.5) y_multiclass(i) = 1.0;
            else y_multiclass(i) = 0.0;
        }
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y_classification, y_regression, y_multiclass;
};

// DecisionTreeClassifier Tests
TEST_F(TreeTest, DecisionTreeClassifierFit) {
    tree::DecisionTreeClassifier dt("gini");
    dt.fit(X, y_classification);
    
    EXPECT_TRUE(dt.is_fitted());
}

TEST_F(TreeTest, DecisionTreeClassifierPredictClasses) {
    tree::DecisionTreeClassifier dt("gini");
    dt.fit(X, y_classification);
    
    VectorXi y_pred = dt.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are valid class labels
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(y_pred(i) == 0 || y_pred(i) == 1);
    }
}

TEST_F(TreeTest, DecisionTreeClassifierPredictProba) {
    tree::DecisionTreeClassifier dt("gini");
    dt.fit(X, y_classification);
    
    MatrixXd y_proba = dt.predict_proba(X);
    EXPECT_EQ(y_proba.rows(), n_samples);
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < n_samples; ++i) {
        double prob_sum = y_proba.row(i).sum();
        EXPECT_NEAR(prob_sum, 1.0, 1e-10);
    }
}

TEST_F(TreeTest, DecisionTreeClassifierMulticlass) {
    tree::DecisionTreeClassifier dt("gini");
    dt.fit(X, y_multiclass);
    
    VectorXi y_pred = dt.predict_classes(X);
    MatrixXd y_proba = dt.predict_proba(X);
    
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

TEST_F(TreeTest, DecisionTreeClassifierEntropy) {
    tree::DecisionTreeClassifier dt("entropy");
    dt.fit(X, y_classification);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXi y_pred = dt.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(TreeTest, DecisionTreeClassifierMaxDepth) {
    tree::DecisionTreeClassifier dt("gini", 3);
    dt.fit(X, y_classification);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXi y_pred = dt.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(TreeTest, DecisionTreeClassifierMinSamplesSplit) {
    tree::DecisionTreeClassifier dt("gini", -1, 10);
    dt.fit(X, y_classification);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXi y_pred = dt.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(TreeTest, DecisionTreeClassifierMinSamplesLeaf) {
    tree::DecisionTreeClassifier dt("gini", -1, 2, 5);
    dt.fit(X, y_classification);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXi y_pred = dt.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(TreeTest, DecisionTreeClassifierMinImpurityDecrease) {
    tree::DecisionTreeClassifier dt("gini", -1, 2, 1, 0.01);
    dt.fit(X, y_classification);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXi y_pred = dt.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(TreeTest, DecisionTreeClassifierDecisionFunction) {
    tree::DecisionTreeClassifier dt("gini");
    dt.fit(X, y_classification);
    
    VectorXd decision = dt.decision_function(X);
    EXPECT_EQ(decision.size(), n_samples);
    
    // Check that decision function values are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(decision(i)));
        EXPECT_FALSE(std::isinf(decision(i)));
    }
}

TEST_F(TreeTest, DecisionTreeClassifierGetSetParams) {
    tree::DecisionTreeClassifier dt("gini", 5, 10, 3, 0.01);
    
    Params params = dt.get_params();
    EXPECT_EQ(params["criterion"], "gini");
    EXPECT_EQ(params["max_depth"], "5");
    EXPECT_EQ(params["min_samples_split"], "10");
    EXPECT_EQ(params["min_samples_leaf"], "3");
    EXPECT_EQ(params["min_impurity_decrease"], "0.010000");
    
    // Test set_params
    Params new_params = {{"criterion", "entropy"}, {"max_depth", "3"}};
    dt.set_params(new_params);
    
    Params updated_params = dt.get_params();
    EXPECT_EQ(updated_params["criterion"], "entropy");
    EXPECT_EQ(updated_params["max_depth"], "3");
}

// DecisionTreeRegressor Tests
TEST_F(TreeTest, DecisionTreeRegressorFit) {
    tree::DecisionTreeRegressor dt("mse");
    dt.fit(X, y_regression);
    
    EXPECT_TRUE(dt.is_fitted());
}

TEST_F(TreeTest, DecisionTreeRegressorPredict) {
    tree::DecisionTreeRegressor dt("mse");
    dt.fit(X, y_regression);
    
    VectorXd y_pred = dt.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(TreeTest, DecisionTreeRegressorMaxDepth) {
    tree::DecisionTreeRegressor dt("mse", 3);
    dt.fit(X, y_regression);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXd y_pred = dt.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(TreeTest, DecisionTreeRegressorMinSamplesSplit) {
    tree::DecisionTreeRegressor dt("mse", -1, 10);
    dt.fit(X, y_regression);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXd y_pred = dt.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(TreeTest, DecisionTreeRegressorMinSamplesLeaf) {
    tree::DecisionTreeRegressor dt("mse", -1, 2, 5);
    dt.fit(X, y_regression);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXd y_pred = dt.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(TreeTest, DecisionTreeRegressorMinImpurityDecrease) {
    tree::DecisionTreeRegressor dt("mse", -1, 2, 1, 0.01);
    dt.fit(X, y_regression);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXd y_pred = dt.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(TreeTest, DecisionTreeRegressorGetSetParams) {
    tree::DecisionTreeRegressor dt("mse", 5, 10, 3, 0.01);
    
    Params params = dt.get_params();
    EXPECT_EQ(params["criterion"], "mse");
    EXPECT_EQ(params["max_depth"], "5");
    EXPECT_EQ(params["min_samples_split"], "10");
    EXPECT_EQ(params["min_samples_leaf"], "3");
    EXPECT_EQ(params["min_impurity_decrease"], "0.010000");
    
    // Test set_params
    Params new_params = {{"max_depth", "3"}, {"min_samples_split", "5"}};
    dt.set_params(new_params);
    
    Params updated_params = dt.get_params();
    EXPECT_EQ(updated_params["max_depth"], "3");
    EXPECT_EQ(updated_params["min_samples_split"], "5");
}

// Edge Cases and Error Handling
TEST_F(TreeTest, DecisionTreeClassifierEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    tree::DecisionTreeClassifier dt("gini");
    EXPECT_THROW(dt.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(TreeTest, DecisionTreeClassifierSingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    VectorXd y_single = VectorXd::Ones(1);
    
    tree::DecisionTreeClassifier dt("gini");
    dt.fit(X_single, y_single);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXi y_pred = dt.predict_classes(X_single);
    EXPECT_EQ(y_pred.size(), 1);
    EXPECT_EQ(y_pred(0), 1);
}

TEST_F(TreeTest, DecisionTreeClassifierSingleFeature) {
    MatrixXd X_single_feature = MatrixXd::Random(n_samples, 1);
    VectorXd y_single_feature = VectorXd::Zero(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        y_single_feature(i) = (X_single_feature(i, 0) > 0.0) ? 1.0 : 0.0;
    }
    
    tree::DecisionTreeClassifier dt("gini");
    dt.fit(X_single_feature, y_single_feature);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXi y_pred = dt.predict_classes(X_single_feature);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(TreeTest, DecisionTreeClassifierPerfectSeparation) {
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
    
    tree::DecisionTreeClassifier dt("gini");
    dt.fit(X_perfect, y_perfect);
    
    EXPECT_TRUE(dt.is_fitted());
    
    VectorXi y_pred = dt.predict_classes(X_perfect);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Should achieve perfect accuracy
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        if (y_pred(i) == static_cast<int>(y_perfect(i))) correct++;
    }
    EXPECT_EQ(correct, n_samples);
}

TEST_F(TreeTest, DecisionTreeClassifierNotFitted) {
    tree::DecisionTreeClassifier dt("gini");
    
    EXPECT_FALSE(dt.is_fitted());
    EXPECT_THROW(dt.predict_classes(X), std::runtime_error);
    EXPECT_THROW(dt.predict_proba(X), std::runtime_error);
    EXPECT_THROW(dt.decision_function(X), std::runtime_error);
}

TEST_F(TreeTest, DecisionTreeRegressorNotFitted) {
    tree::DecisionTreeRegressor dt("mse");
    
    EXPECT_FALSE(dt.is_fitted());
    EXPECT_THROW(dt.predict(X), std::runtime_error);
}

TEST_F(TreeTest, DecisionTreeClassifierWrongFeatureCount) {
    tree::DecisionTreeClassifier dt("gini");
    dt.fit(X, y_classification);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(dt.predict_classes(X_wrong), std::invalid_argument);
}

TEST_F(TreeTest, DecisionTreeRegressorWrongFeatureCount) {
    tree::DecisionTreeRegressor dt("mse");
    dt.fit(X, y_regression);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(dt.predict(X_wrong), std::invalid_argument);
}

TEST_F(TreeTest, DecisionTreeClassifierInvalidCriterion) {
    EXPECT_THROW(tree::DecisionTreeClassifier dt("invalid"), std::invalid_argument);
}

TEST_F(TreeTest, DecisionTreeRegressorInvalidCriterion) {
    EXPECT_THROW(tree::DecisionTreeRegressor dt("invalid"), std::invalid_argument);
}

// Model persistence tests
TEST_F(TreeTest, DecisionTreeClassifierModelPersistence) {
    tree::DecisionTreeClassifier dt("gini", 3);
    dt.fit(X, y_classification);
    
    // Save model
    dt.save("test_dt_classifier.bin");
    
    // Load model
    tree::DecisionTreeClassifier dt_loaded("gini", 3);
    dt_loaded.load("test_dt_classifier.bin");
    
    // Test that loaded model works
    EXPECT_TRUE(dt_loaded.is_fitted());
    VectorXi y_pred_original = dt.predict_classes(X);
    VectorXi y_pred_loaded = dt_loaded.predict_classes(X);
    
    // Predictions should be identical
    EXPECT_TRUE(y_pred_original.isApprox(y_pred_loaded));
    
    // Clean up
    std::remove("test_dt_classifier.bin");
}

TEST_F(TreeTest, DecisionTreeRegressorModelPersistence) {
    tree::DecisionTreeRegressor dt("mse", 3);
    dt.fit(X, y_regression);
    
    // Save model
    dt.save("test_dt_regressor.bin");
    
    // Load model
    tree::DecisionTreeRegressor dt_loaded("mse", 3);
    dt_loaded.load("test_dt_regressor.bin");
    
    // Test that loaded model works
    EXPECT_TRUE(dt_loaded.is_fitted());
    VectorXd y_pred_original = dt.predict(X);
    VectorXd y_pred_loaded = dt_loaded.predict(X);
    
    // Predictions should be identical
    EXPECT_TRUE(y_pred_original.isApprox(y_pred_loaded, 1e-10));
    
    // Clean up
    std::remove("test_dt_regressor.bin");
}

TEST_F(TreeTest, DecisionTreeClassifierNotFittedSave) {
    tree::DecisionTreeClassifier dt("gini", 3);
    EXPECT_THROW(dt.save("test.bin"), std::runtime_error);
}

TEST_F(TreeTest, DecisionTreeRegressorNotFittedSave) {
    tree::DecisionTreeRegressor dt("mse", 3);
    EXPECT_THROW(dt.save("test.bin"), std::runtime_error);
}

TEST_F(TreeTest, DecisionTreeClassifierLoadNonexistentFile) {
    tree::DecisionTreeClassifier dt("gini", 3);
    EXPECT_THROW(dt.load("nonexistent_file.bin"), std::runtime_error);
}

TEST_F(TreeTest, DecisionTreeRegressorLoadNonexistentFile) {
    tree::DecisionTreeRegressor dt("mse", 3);
    EXPECT_THROW(dt.load("nonexistent_file.bin"), std::runtime_error);
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
