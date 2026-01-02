#include <gtest/gtest.h>
#include "auroraml/calibration.hpp"
#include "auroraml/linear_model.hpp"
#include "auroraml/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <memory>

namespace auroraml {
namespace test {

class CalibrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_samples = 100;
        n_features = 4;
        
        X = MatrixXd::Random(n_samples, n_features);
        y_classification = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_classification(i) = (X(i, 0) + X(i, 1) > 0.0) ? 1.0 : 0.0;
        }
        
        X_test = MatrixXd::Random(20, n_features);
    }
    
    int n_samples, n_features;
    MatrixXd X, X_test;
    VectorXd y_classification;
};

// Positive test cases
TEST_F(CalibrationTest, CalibratedClassifierCVFit) {
    auto base = std::make_shared<linear_model::LogisticRegression>(1.0, 1000, 42);
    calibration::CalibratedClassifierCV calib(base, "sigmoid", 3);
    calib.fit(X, y_classification);
    
    EXPECT_TRUE(calib.is_fitted());
}

TEST_F(CalibrationTest, CalibratedClassifierCVPredict) {
    auto base = std::make_shared<linear_model::LogisticRegression>(1.0, 1000, 42);
    calibration::CalibratedClassifierCV calib(base, "sigmoid", 3);
    calib.fit(X, y_classification);
    
    VectorXi y_pred = calib.predict_classes(X_test);
    EXPECT_EQ(y_pred.size(), X_test.rows());
}

TEST_F(CalibrationTest, CalibratedClassifierCVPredictProba) {
    auto base = std::make_shared<linear_model::LogisticRegression>(1.0, 1000, 42);
    calibration::CalibratedClassifierCV calib(base, "sigmoid", 3);
    calib.fit(X, y_classification);
    
    MatrixXd y_proba = calib.predict_proba(X_test);
    EXPECT_EQ(y_proba.rows(), X_test.rows());
    EXPECT_EQ(y_proba.cols(), 2);
    
    for (int i = 0; i < y_proba.rows(); ++i) {
        double sum = y_proba.row(i).sum();
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

// Negative test cases
TEST_F(CalibrationTest, CalibratedClassifierCVNotFitted) {
    auto base = std::make_shared<linear_model::LogisticRegression>(1.0);
    calibration::CalibratedClassifierCV calib(base, "sigmoid", 3);
    EXPECT_THROW(calib.predict_classes(X_test), std::runtime_error);
}

TEST_F(CalibrationTest, CalibratedClassifierCVEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    auto base = std::make_shared<linear_model::LogisticRegression>(1.0);
    calibration::CalibratedClassifierCV calib(base, "sigmoid", 3);
    EXPECT_THROW(calib.fit(X_empty, y_empty), std::invalid_argument);
}

} // namespace test
} // namespace auroraml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
