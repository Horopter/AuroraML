#include <gtest/gtest.h>
#include "auroraml/svm.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class SVMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 3;
        
        X = MatrixXd::Random(n_samples, n_features);
        
        // Create linearly separable classification problem
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : -1.0;
        }
        
        // Create multi-class problem (convert to binary for SVM)
        y_multiclass = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            if (X(i, 0) > 0.5) y_multiclass(i) = 1.0;
            else y_multiclass(i) = -1.0;
        }
        
        // Create regression problem
        y_regression = VectorXd::Random(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y_classification, y_multiclass, y_regression;
};

// LinearSVC Tests
TEST_F(SVMTest, LinearSVCFit) {
    svm::LinearSVC svm(1.0, 1000, 0.01);
    svm.fit(X, y_classification);
    
    EXPECT_TRUE(svm.is_fitted());
}

TEST_F(SVMTest, LinearSVCPredictClasses) {
    svm::LinearSVC svm(1.0, 1000, 0.01);
    svm.fit(X, y_classification);
    
    VectorXi y_pred = svm.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are valid class labels
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(y_pred(i) == 1 || y_pred(i) == 0);
    }
}

TEST_F(SVMTest, LinearSVCDifferentC) {
    svm::LinearSVC svm(0.1, 1000, 0.01);
    svm.fit(X, y_classification);
    
    EXPECT_TRUE(svm.is_fitted());
    
    VectorXi y_pred = svm.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(SVMTest, LinearSVCDifferentLearningRate) {
    svm::LinearSVC svm(1.0, 1000, 0.001);
    svm.fit(X, y_classification);
    
    EXPECT_TRUE(svm.is_fitted());
    
    VectorXi y_pred = svm.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(SVMTest, LinearSVCDifferentMaxIter) {
    svm::LinearSVC svm(1.0, 500, 0.01);
    svm.fit(X, y_classification);
    
    EXPECT_TRUE(svm.is_fitted());
    
    VectorXi y_pred = svm.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(SVMTest, LinearSVCGetSetParams) {
    svm::LinearSVC svm(1.0, 1000, 0.01, 42);
    
    Params params = svm.get_params();
    EXPECT_EQ(params["C"], "1.000000");
    EXPECT_EQ(params["lr"], "0.010000");
    EXPECT_EQ(params["max_iter"], "1000");
    EXPECT_EQ(params["random_state"], "42");
    
    // Test set_params
    Params new_params = {{"C", "0.5"}, {"lr", "0.005"}};
    svm.set_params(new_params);
    
    Params updated_params = svm.get_params();
    EXPECT_EQ(updated_params["C"], "0.500000");
    EXPECT_EQ(updated_params["lr"], "0.005000");
}

TEST_F(SVMTest, LinearSVCMulticlass) {
    svm::LinearSVC svm(1.0, 1000, 0.01);
    svm.fit(X, y_multiclass);
    
    VectorXi y_pred = svm.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are valid class labels
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(y_pred(i) == 1 || y_pred(i) == 0);
    }
}

TEST_F(SVMTest, LinearSVCDecisionFunction) {
    svm::LinearSVC svm(1.0, 1000, 0.01);
    svm.fit(X, y_classification);
    
    VectorXd decision = svm.decision_function(X);
    EXPECT_EQ(decision.size(), n_samples);
    
    // Check that decision function values are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(decision(i)));
        EXPECT_FALSE(std::isinf(decision(i)));
    }
}

// Edge Cases and Error Handling
TEST_F(SVMTest, LinearSVCEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    svm::LinearSVC svm(1.0, 1000, 0.01);
    EXPECT_THROW(svm.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(SVMTest, LinearSVCSingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    VectorXd y_single = VectorXd::Ones(1);
    
    svm::LinearSVC svm(1.0, 1000, 0.01);
    svm.fit(X_single, y_single);
    
    EXPECT_TRUE(svm.is_fitted());
    
    VectorXi y_pred = svm.predict_classes(X_single);
    EXPECT_EQ(y_pred.size(), 1);
}

TEST_F(SVMTest, LinearSVCSingleFeature) {
    MatrixXd X_single_feature = MatrixXd::Random(n_samples, 1);
    VectorXd y_single_feature = VectorXd::Zero(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        y_single_feature(i) = (X_single_feature(i, 0) > 0.0) ? 1.0 : -1.0;
    }
    
    svm::LinearSVC svm(1.0, 1000, 0.01);
    svm.fit(X_single_feature, y_single_feature);
    
    EXPECT_TRUE(svm.is_fitted());
    
    VectorXi y_pred = svm.predict_classes(X_single_feature);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(SVMTest, LinearSVCPerfectSeparation) {
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
        y_perfect(i) = -1.0;
    }
    
    svm::LinearSVC svm(1.0, 1000, 0.01);
    svm.fit(X_perfect, y_perfect);
    
    EXPECT_TRUE(svm.is_fitted());
    
    VectorXi y_pred = svm.predict_classes(X_perfect);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Should achieve good accuracy on separable data
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        int expected = (y_perfect(i) > 0) ? 1 : 0;  // Convert -1/1 to 0/1
        if (y_pred(i) == expected) correct++;
    }
    EXPECT_GT(correct, n_samples * 0.8);  // At least 80% accuracy
}

TEST_F(SVMTest, LinearSVCNotFitted) {
    svm::LinearSVC svm(1.0, 1000, 0.01);
    
    EXPECT_FALSE(svm.is_fitted());
    EXPECT_THROW(svm.predict_classes(X), std::runtime_error);
    EXPECT_THROW(svm.decision_function(X), std::runtime_error);
}

TEST_F(SVMTest, LinearSVCWrongFeatureCount) {
    svm::LinearSVC svm(1.0, 1000, 0.01);
    svm.fit(X, y_classification);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(svm.predict_classes(X_wrong), std::invalid_argument);
}

TEST_F(SVMTest, LinearSVCDimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    svm::LinearSVC svm(1.0, 1000, 0.01);
    EXPECT_THROW(svm.fit(X_wrong, y_wrong), std::invalid_argument);
}

TEST_F(SVMTest, LinearSVCNegativeC) {
    svm::LinearSVC svm(-1.0, 1000, 0.01);
    EXPECT_THROW(svm.fit(X, y_classification), std::invalid_argument);
}

TEST_F(SVMTest, LinearSVCNegativeLearningRate) {
    svm::LinearSVC svm(1.0, 1000, -0.01);
    EXPECT_THROW(svm.fit(X, y_classification), std::invalid_argument);
}

TEST_F(SVMTest, LinearSVCNegativeMaxIter) {
    svm::LinearSVC svm(1.0, -1000, 0.01);
    EXPECT_THROW(svm.fit(X, y_classification), std::invalid_argument);
}

TEST_F(SVMTest, LinearSVCZeroMaxIter) {
    svm::LinearSVC svm(1.0, 0, 0.01);
    EXPECT_THROW(svm.fit(X, y_classification), std::invalid_argument);
}

TEST_F(SVMTest, LinearSVCZeroLearningRate) {
    svm::LinearSVC svm(1.0, 1000, 0.0);
    EXPECT_THROW(svm.fit(X, y_classification), std::invalid_argument);
}

TEST_F(SVMTest, LinearSVCZeroC) {
    svm::LinearSVC svm(0.0, 1000, 0.01);
    EXPECT_THROW(svm.fit(X, y_classification), std::invalid_argument);
}

TEST_F(SVMTest, LinearSVCLargeC) {
    svm::LinearSVC svm(1000.0, 1000, 0.01);
    svm.fit(X, y_classification);
    
    EXPECT_TRUE(svm.is_fitted());
    
    VectorXi y_pred = svm.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(SVMTest, LinearSVCSmallLearningRate) {
    svm::LinearSVC svm(1.0, 10000, 0.0001);
    svm.fit(X, y_classification);
    
    EXPECT_TRUE(svm.is_fitted());
    
    VectorXi y_pred = svm.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(SVMTest, LinearSVCLargeMaxIter) {
    svm::LinearSVC svm(1.0, 10000, 0.01);
    svm.fit(X, y_classification);
    
    EXPECT_TRUE(svm.is_fitted());
    
    VectorXi y_pred = svm.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
}

TEST_F(SVMTest, LinearSVCConsistency) {
    svm::LinearSVC svm1(1.0, 1000, 0.01);
    svm::LinearSVC svm2(1.0, 1000, 0.01);
    
    svm1.fit(X, y_classification);
    svm2.fit(X, y_classification);
    
    VectorXi y_pred1 = svm1.predict_classes(X);
    VectorXi y_pred2 = svm2.predict_classes(X);
    
    // Results should be consistent (though not necessarily identical due to randomness)
    EXPECT_EQ(y_pred1.size(), y_pred2.size());
    EXPECT_EQ(y_pred1.size(), n_samples);
}

// SVM persistence tests
TEST_F(SVMTest, LinearSVCModelPersistence) {
    svm::LinearSVC svc(1.0, 1000, 0.01, 42);
    svc.fit(X, y_classification);
    
    // Save model
    svc.save("test_svc_classifier.bin");
    
    // Load model
    svm::LinearSVC svc_loaded(1.0, 1000, 0.01, 42);
    svc_loaded.load("test_svc_classifier.bin");
    
    // Test that loaded model works
    EXPECT_TRUE(svc_loaded.is_fitted());
    VectorXi y_pred_original = svc.predict_classes(X);
    VectorXi y_pred_loaded = svc_loaded.predict_classes(X);
    
    // Predictions should be identical
    EXPECT_TRUE(y_pred_original.isApprox(y_pred_loaded));
    
    // Clean up
    std::remove("test_svc_classifier.bin");
}

TEST_F(SVMTest, LinearSVCNotFittedSave) {
    svm::LinearSVC svc(1.0, 1000, 0.01, 42);
    EXPECT_THROW(svc.save("test.bin"), std::runtime_error);
}

TEST_F(SVMTest, LinearSVCLoadNonexistentFile) {
    svm::LinearSVC svc(1.0, 1000, 0.01, 42);
    EXPECT_THROW(svc.load("nonexistent_file.bin"), std::runtime_error);
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
