#include <gtest/gtest.h>
#include "ingenuityml/outlier_detection.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class OutlierDetectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_dummy = VectorXd::Zero(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y_dummy;
};

// Positive test cases
TEST_F(OutlierDetectionTest, IsolationForestFit) {
    outlier_detection::IsolationForest iso(100, 0.1, 42);
    iso.fit(X, y_dummy);
    
    EXPECT_TRUE(iso.is_fitted());
}

TEST_F(OutlierDetectionTest, IsolationForestPredict) {
    outlier_detection::IsolationForest iso(100, 0.1, 42);
    iso.fit(X, y_dummy);
    
    VectorXi predictions = iso.predict(X);
    EXPECT_EQ(predictions.size(), n_samples);
    
    // IsolationForest returns -1 for outliers, 1 for inliers
    for (int i = 0; i < predictions.size(); ++i) {
        EXPECT_TRUE(predictions(i) == -1 || predictions(i) == 1);
    }
}

TEST_F(OutlierDetectionTest, IsolationForestDecisionFunction) {
    outlier_detection::IsolationForest iso(100, -1, 0.1, 42);
    iso.fit(X, y_dummy);
    
    VectorXd scores = iso.decision_function(X);
    EXPECT_EQ(scores.size(), n_samples);
}

TEST_F(OutlierDetectionTest, LocalOutlierFactorFit) {
    outlier_detection::LocalOutlierFactor lof(20);
    lof.fit(X, y_dummy);
    
    EXPECT_TRUE(lof.is_fitted());
}

TEST_F(OutlierDetectionTest, LocalOutlierFactorPredict) {
    outlier_detection::LocalOutlierFactor lof(20);
    lof.fit(X, y_dummy);
    
    VectorXi predictions = lof.predict(X);
    EXPECT_EQ(predictions.size(), n_samples);
}

// Negative test cases
TEST_F(OutlierDetectionTest, IsolationForestNotFitted) {
    outlier_detection::IsolationForest iso(100);
    EXPECT_THROW(iso.predict(X), std::runtime_error);
}

TEST_F(OutlierDetectionTest, IsolationForestEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    outlier_detection::IsolationForest iso(100);
    EXPECT_THROW(iso.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(OutlierDetectionTest, LocalOutlierFactorNegativeNeighbors) {
    outlier_detection::LocalOutlierFactor lof(-1);
    EXPECT_THROW(lof.fit(X, y_dummy), std::invalid_argument);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
