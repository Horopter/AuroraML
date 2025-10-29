#include <gtest/gtest.h>
#include "auroraml/pca.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class PCATest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 5;
        n_components = 2;
        
        X = MatrixXd::Random(n_samples, n_features);
        y = VectorXd::Zero(n_samples);  // Dummy y for fit
    }
    
    int n_samples, n_features, n_components;
    MatrixXd X;
    VectorXd y;
};

TEST_F(PCATest, PCAFit) {
    decomposition::PCA pca(n_components);
    pca.fit(X, y);
    
    EXPECT_TRUE(pca.is_fitted());
    EXPECT_EQ(pca.components().rows(), n_components);
    EXPECT_EQ(pca.components().cols(), n_features);
}

TEST_F(PCATest, PCATransform) {
    decomposition::PCA pca(n_components);
    pca.fit(X, y);
    
    MatrixXd X_transformed = pca.transform(X);
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), n_components);
    
    // Check that transformed data has reasonable values
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_components; ++j) {
            EXPECT_FALSE(std::isnan(X_transformed(i, j)));
            EXPECT_FALSE(std::isinf(X_transformed(i, j)));
        }
    }
}

TEST_F(PCATest, PCAInverseTransform) {
    decomposition::PCA pca(n_components);
    pca.fit(X, y);
    
    MatrixXd X_transformed = pca.transform(X);
    MatrixXd X_reconstructed = pca.inverse_transform(X_transformed);
    
    EXPECT_EQ(X_reconstructed.rows(), n_samples);
    EXPECT_EQ(X_reconstructed.cols(), n_features);
    
    // Check that reconstructed data has reasonable values
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            EXPECT_FALSE(std::isnan(X_reconstructed(i, j)));
            EXPECT_FALSE(std::isinf(X_reconstructed(i, j)));
        }
    }
}

TEST_F(PCATest, PCAFitTransform) {
    decomposition::PCA pca(n_components);
    MatrixXd X_transformed = pca.fit_transform(X, y);
    
    EXPECT_TRUE(pca.is_fitted());
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), n_components);
}

TEST_F(PCATest, PCAExplainedVariance) {
    decomposition::PCA pca(n_components);
    pca.fit(X, y);
    
    VectorXd explained_variance = pca.explained_variance();
    EXPECT_EQ(explained_variance.size(), n_components);
    
    // Check that explained variance is non-negative and decreasing
    for (int i = 0; i < n_components; ++i) {
        EXPECT_GE(explained_variance(i), 0.0);
        if (i > 0) {
            EXPECT_LE(explained_variance(i), explained_variance(i-1));
        }
    }
}

TEST_F(PCATest, PCAExplainedVarianceRatio) {
    decomposition::PCA pca(n_components);
    pca.fit(X, y);
    
    VectorXd explained_variance_ratio = pca.explained_variance_ratio();
    EXPECT_EQ(explained_variance_ratio.size(), n_components);
    
    // Check that explained variance ratios are in [0, 1] and sum to <= 1
    double total_ratio = 0.0;
    for (int i = 0; i < n_components; ++i) {
        EXPECT_GE(explained_variance_ratio(i), 0.0);
        EXPECT_LE(explained_variance_ratio(i), 1.0);
        total_ratio += explained_variance_ratio(i);
    }
    EXPECT_LE(total_ratio, 1.0);
}

TEST_F(PCATest, PCAMean) {
    decomposition::PCA pca(n_components);
    pca.fit(X, y);
    
    VectorXd mean = pca.mean();
    EXPECT_EQ(mean.size(), n_features);
    
    // Check that mean values are reasonable
    for (int j = 0; j < n_features; ++j) {
        EXPECT_FALSE(std::isnan(mean(j)));
        EXPECT_FALSE(std::isinf(mean(j)));
    }
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
