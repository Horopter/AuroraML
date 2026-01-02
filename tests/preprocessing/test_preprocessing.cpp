#include <gtest/gtest.h>
#include "auroraml/preprocessing.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class PreprocessingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 50;
        n_features = 3;
        
        X = MatrixXd::Random(n_samples, n_features);
        // Scale features differently
        X.col(0) *= 100;  // Large scale
        X.col(1) *= 0.1;  // Small scale
        X.col(2) *= 5;    // Medium scale
        
        y = VectorXd::Random(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X;
    VectorXd y;
};

TEST_F(PreprocessingTest, StandardScalerFit) {
    preprocessing::StandardScaler scaler;
    scaler.fit(X, y);
    
    EXPECT_TRUE(scaler.is_fitted());
    EXPECT_EQ(scaler.mean().size(), n_features);
    EXPECT_EQ(scaler.scale().size(), n_features);
}

TEST_F(PreprocessingTest, StandardScalerTransform) {
    preprocessing::StandardScaler scaler;
    scaler.fit(X, y);
    
    MatrixXd X_scaled = scaler.transform(X);
    EXPECT_EQ(X_scaled.rows(), n_samples);
    EXPECT_EQ(X_scaled.cols(), n_features);
    
    // Check that scaled data has approximately zero mean and unit variance
    VectorXd scaled_mean = X_scaled.colwise().mean();
    for (int j = 0; j < n_features; ++j) {
        EXPECT_NEAR(scaled_mean(j), 0.0, 1e-10);
    }
}

TEST_F(PreprocessingTest, StandardScalerInverseTransform) {
    preprocessing::StandardScaler scaler;
    scaler.fit(X, y);
    
    MatrixXd X_scaled = scaler.transform(X);
    MatrixXd X_reconstructed = scaler.inverse_transform(X_scaled);
    
    // Check reconstruction accuracy
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            EXPECT_NEAR(X(i, j), X_reconstructed(i, j), 1e-10);
        }
    }
}

TEST_F(PreprocessingTest, MinMaxScalerFit) {
    preprocessing::MinMaxScaler scaler;
    scaler.fit(X, y);
    
    EXPECT_TRUE(scaler.is_fitted());
    EXPECT_EQ(scaler.data_min().size(), n_features);
    EXPECT_EQ(scaler.data_max().size(), n_features);
}

TEST_F(PreprocessingTest, MinMaxScalerTransform) {
    preprocessing::MinMaxScaler scaler;
    scaler.fit(X, y);
    
    MatrixXd X_scaled = scaler.transform(X);
    EXPECT_EQ(X_scaled.rows(), n_samples);
    EXPECT_EQ(X_scaled.cols(), n_features);
    
    // Check that scaled data is in [0, 1] range
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            EXPECT_GE(X_scaled(i, j), 0.0);
            EXPECT_LE(X_scaled(i, j), 1.0);
        }
    }
}

TEST_F(PreprocessingTest, RobustScalerFit) {
    preprocessing::RobustScaler scaler;
    scaler.fit(X, y);
    
    EXPECT_TRUE(scaler.is_fitted());
}

TEST_F(PreprocessingTest, RobustScalerTransform) {
    preprocessing::RobustScaler scaler;
    scaler.fit(X, y);
    
    MatrixXd X_scaled = scaler.transform(X);
    EXPECT_EQ(X_scaled.rows(), n_samples);
    EXPECT_EQ(X_scaled.cols(), n_features);
    
    // Check that scaled data has reasonable values
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            EXPECT_FALSE(std::isnan(X_scaled(i, j)));
            EXPECT_FALSE(std::isinf(X_scaled(i, j)));
        }
    }
}

TEST_F(PreprocessingTest, OneHotEncoderFit) {
    // Create categorical data
    MatrixXd X_cat = MatrixXd::Zero(10, 2);
    X_cat.col(0) = VectorXd::LinSpaced(10, 0, 9);  // 0, 1, 2, ..., 9
    VectorXd col1_data(10);
    col1_data << 0, 1, 2, 3, 4, 0, 1, 2, 3, 4;
    X_cat.col(1) = col1_data;
    
    preprocessing::OneHotEncoder encoder;
    encoder.fit(X_cat, y);
    
    EXPECT_TRUE(encoder.is_fitted());
}

TEST_F(PreprocessingTest, OneHotEncoderTransform) {
    // Create categorical data
    MatrixXd X_cat = MatrixXd::Zero(5, 2);
    X_cat.col(0) = VectorXd::LinSpaced(5, 0, 4);  // 0, 1, 2, 3, 4
    VectorXd col1_data(5);
    col1_data << 0, 1, 2, 0, 1;
    X_cat.col(1) = col1_data;
    
    preprocessing::OneHotEncoder encoder;
    encoder.fit(X_cat, y);
    
    MatrixXd X_encoded = encoder.transform(X_cat);
    EXPECT_EQ(X_encoded.rows(), 5);
    EXPECT_EQ(X_encoded.cols(), 8);  // 5 + 3 categories
    
    // Check that each row sums to 2 (one hot per feature)
    for (int i = 0; i < 5; ++i) {
        double row_sum = X_encoded.row(i).sum();
        EXPECT_NEAR(row_sum, 2.0, 1e-10);
    }
}

TEST_F(PreprocessingTest, OrdinalEncoderFit) {
    // Create categorical data
    MatrixXd X_cat = MatrixXd::Zero(10, 2);
    VectorXd col0_data(10);
    col0_data << 1, 2, 3, 1, 2, 3, 1, 2, 3, 1;
    X_cat.col(0) = col0_data;
    VectorXd col1_data(10);
    col1_data << 1, 2, 1, 2, 1, 2, 1, 2, 1, 2;
    X_cat.col(1) = col1_data;
    
    preprocessing::OrdinalEncoder encoder;
    encoder.fit(X_cat, y);
    
    EXPECT_TRUE(encoder.is_fitted());
}

TEST_F(PreprocessingTest, OrdinalEncoderTransform) {
    // Create categorical data
    MatrixXd X_cat = MatrixXd::Zero(6, 2);
    VectorXd col0_data(6);
    col0_data << 1, 2, 3, 1, 2, 3;
    X_cat.col(0) = col0_data;
    VectorXd col1_data(6);
    col1_data << 1, 2, 1, 2, 1, 2;
    X_cat.col(1) = col1_data;
    
    preprocessing::OrdinalEncoder encoder;
    encoder.fit(X_cat, y);
    
    MatrixXd X_encoded = encoder.transform(X_cat);
    EXPECT_EQ(X_encoded.rows(), 6);
    EXPECT_EQ(X_encoded.cols(), 2);
    
    // Check that encoded values are non-negative integers
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_GE(X_encoded(i, j), 0.0);
            EXPECT_EQ(X_encoded(i, j), static_cast<int>(X_encoded(i, j)));
        }
    }
}

TEST_F(PreprocessingTest, OrdinalEncoderInverseTransform) {
    // Create categorical data
    MatrixXd X_cat = MatrixXd::Zero(6, 2);
    VectorXd col0_data(6);
    col0_data << 1, 2, 3, 1, 2, 3;
    X_cat.col(0) = col0_data;
    VectorXd col1_data(6);
    col1_data << 1, 2, 1, 2, 1, 2;
    X_cat.col(1) = col1_data;
    
    preprocessing::OrdinalEncoder encoder;
    encoder.fit(X_cat, y);
    
    MatrixXd X_encoded = encoder.transform(X_cat);
    MatrixXd X_decoded = encoder.inverse_transform(X_encoded);
    
    // Check reconstruction accuracy
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(X_cat(i, j), X_decoded(i, j), 1e-10);
        }
    }
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    // Enable test shuffling within this file
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;  // Reproducible shuffle
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
