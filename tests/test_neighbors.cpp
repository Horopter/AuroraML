#include <gtest/gtest.h>
#include "auroraml/neighbors.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class NeighborsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 50;
        n_features = 2;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_classification = VectorXd::Zero(n_samples);
        y_regression = VectorXd::Random(n_samples);
        
        // Create simple classification problem
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y_classification, y_regression;
};

TEST_F(NeighborsTest, KNeighborsClassifierFit) {
    neighbors::KNeighborsClassifier knn(3);
    knn.fit(X, y_classification);
    
    EXPECT_TRUE(knn.is_fitted());
}

TEST_F(NeighborsTest, KNeighborsClassifierPredict) {
    neighbors::KNeighborsClassifier knn(3);
    knn.fit(X, y_classification);
    
    VectorXi y_pred = knn.predict_classes(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are valid class labels
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(y_pred(i) == 0 || y_pred(i) == 1);
    }
}

TEST_F(NeighborsTest, KNeighborsClassifierPredictProba) {
    neighbors::KNeighborsClassifier knn(3);
    knn.fit(X, y_classification);
    
    MatrixXd y_proba = knn.predict_proba(X);
    EXPECT_EQ(y_proba.rows(), n_samples);
    EXPECT_EQ(y_proba.cols(), 2);  // Binary classification
    
    // Check that probabilities sum to 1
    for (int i = 0; i < n_samples; ++i) {
        double prob_sum = y_proba.row(i).sum();
        EXPECT_NEAR(prob_sum, 1.0, 1e-10);
    }
}

TEST_F(NeighborsTest, KNeighborsRegressorFit) {
    neighbors::KNeighborsRegressor knn(3);
    knn.fit(X, y_regression);
    
    EXPECT_TRUE(knn.is_fitted());
}

TEST_F(NeighborsTest, KNeighborsRegressorPredict) {
    neighbors::KNeighborsRegressor knn(3);
    knn.fit(X, y_regression);
    
    VectorXd y_pred = knn.predict(X);
    EXPECT_EQ(y_pred.size(), n_samples);
    
    // Check that predictions are reasonable
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(std::isnan(y_pred(i)));
        EXPECT_FALSE(std::isinf(y_pred(i)));
    }
}

TEST_F(NeighborsTest, KNeighborsClassifierPersistence) {
    neighbors::KNeighborsClassifier knn(3);
    knn.fit(X, y_classification);
    
    // Save model
    knn.save("test_knn_clf.bin");
    
    // Load model
    neighbors::KNeighborsClassifier knn_loaded;
    knn_loaded.load("test_knn_clf.bin");
    
    EXPECT_TRUE(knn_loaded.is_fitted());
    
    // Compare predictions
    VectorXi y_pred_orig = knn.predict_classes(X);
    VectorXi y_pred_loaded = knn_loaded.predict_classes(X);
    
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_EQ(y_pred_orig(i), y_pred_loaded(i));
    }
}

TEST_F(NeighborsTest, KNeighborsRegressorPersistence) {
    neighbors::KNeighborsRegressor knn(3);
    knn.fit(X, y_regression);
    
    // Save model
    knn.save("test_knn_reg.bin");
    
    // Load model
    neighbors::KNeighborsRegressor knn_loaded;
    knn_loaded.load("test_knn_reg.bin");
    
    EXPECT_TRUE(knn_loaded.is_fitted());
    
    // Compare predictions
    VectorXd y_pred_orig = knn.predict(X);
    VectorXd y_pred_loaded = knn_loaded.predict(X);
    
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_NEAR(y_pred_orig(i), y_pred_loaded(i), 1e-10);
    }
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
