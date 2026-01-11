#include <gtest/gtest.h>
#include "ingenuityml/preprocessing_extended.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class PreprocessingExtendedTest : public ::testing::Test {
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
TEST_F(PreprocessingExtendedTest, MaxAbsScalerFit) {
    preprocessing::MaxAbsScaler scaler;
    scaler.fit(X, y_dummy);
    
    EXPECT_TRUE(scaler.is_fitted());
}

TEST_F(PreprocessingExtendedTest, MaxAbsScalerTransform) {
    preprocessing::MaxAbsScaler scaler;
    scaler.fit(X, y_dummy);
    
    MatrixXd X_transformed = scaler.transform(X);
    EXPECT_EQ(X_transformed.rows(), X.rows());
    EXPECT_EQ(X_transformed.cols(), X.cols());
    
    // Values should be in [-1, 1]
    for (int i = 0; i < X_transformed.rows(); ++i) {
        for (int j = 0; j < X_transformed.cols(); ++j) {
            EXPECT_LE(std::abs(X_transformed(i, j)), 1.0 + 1e-10);
        }
    }
}

TEST_F(PreprocessingExtendedTest, MaxAbsScalerMaxAbs) {
    preprocessing::MaxAbsScaler scaler;
    scaler.fit(X, y_dummy);
    
    VectorXd max_abs = scaler.max_abs();
    EXPECT_EQ(max_abs.size(), n_features);
}

TEST_F(PreprocessingExtendedTest, BinarizerFit) {
    preprocessing::Binarizer binarizer(0.0);
    binarizer.fit(X, y_dummy);
    
    EXPECT_TRUE(binarizer.is_fitted());
}

TEST_F(PreprocessingExtendedTest, BinarizerTransform) {
    preprocessing::Binarizer binarizer(0.0);
    binarizer.fit(X, y_dummy);
    
    MatrixXd X_binary = binarizer.transform(X);
    EXPECT_EQ(X_binary.rows(), X.rows());
    EXPECT_EQ(X_binary.cols(), X.cols());
    
    // Should be binary (0 or 1)
    for (int i = 0; i < X_binary.rows(); ++i) {
        for (int j = 0; j < X_binary.cols(); ++j) {
            EXPECT_TRUE(X_binary(i, j) == 0.0 || X_binary(i, j) == 1.0);
        }
    }
}

// Negative test cases
TEST_F(PreprocessingExtendedTest, MaxAbsScalerNotFitted) {
    preprocessing::MaxAbsScaler scaler;
    EXPECT_THROW(scaler.transform(X), std::runtime_error);
}

TEST_F(PreprocessingExtendedTest, MaxAbsScalerEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    preprocessing::MaxAbsScaler scaler;
    EXPECT_THROW(scaler.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(PreprocessingExtendedTest, BinarizerNotFitted) {
    preprocessing::Binarizer binarizer(0.0);
    EXPECT_THROW(binarizer.transform(X), std::runtime_error);
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
