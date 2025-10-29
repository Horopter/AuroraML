#include <gtest/gtest.h>
#include "auroraml/pca.hpp"
#include "auroraml/truncated_svd.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class DecompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 5;
        
        X = MatrixXd::Random(n_samples, n_features);
        
        // Create data with clear principal components
        X_pca = MatrixXd::Zero(n_samples, n_features);
        for (int i = 0; i < n_samples; ++i) {
            X_pca(i, 0) = 2.0 * (rand() % 100) / 100.0 - 1.0;  // First PC
            X_pca(i, 1) = 0.5 * (rand() % 100) / 100.0 - 0.25; // Second PC
            X_pca(i, 2) = 0.1 * (rand() % 100) / 100.0 - 0.05; // Third PC
            X_pca(i, 3) = 0.05 * (rand() % 100) / 100.0 - 0.025; // Fourth PC
            X_pca(i, 4) = 0.01 * (rand() % 100) / 100.0 - 0.005; // Fifth PC
        }
        
        // Create sparse data for TruncatedSVD
        X_sparse = MatrixXd::Zero(n_samples, n_features);
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features; ++j) {
                if (rand() % 3 == 0) {  // 1/3 sparsity
                    X_sparse(i, j) = (rand() % 100) / 100.0 - 0.5;
                }
            }
        }
    }
    
    int n_samples, n_features;
    MatrixXd X, X_pca, X_sparse;
};

// PCA Tests
TEST_F(DecompositionTest, PCAFit) {
    decomposition::PCA pca(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    pca.fit(X_pca, y_dummy);
    
    EXPECT_TRUE(pca.is_fitted());
}

TEST_F(DecompositionTest, PCATransform) {
    decomposition::PCA pca(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    pca.fit(X_pca, y_dummy);
    
    MatrixXd X_transformed = pca.transform(X_pca);
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), 3);
    
    // Check that transformed data is reasonable
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FALSE(std::isnan(X_transformed(i, j)));
            EXPECT_FALSE(std::isinf(X_transformed(i, j)));
        }
    }
}

TEST_F(DecompositionTest, PCAFitTransform) {
    decomposition::PCA pca(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    MatrixXd X_transformed = pca.fit_transform(X_pca, y_dummy);
    
    EXPECT_TRUE(pca.is_fitted());
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), 3);
}

TEST_F(DecompositionTest, PCAComponents) {
    decomposition::PCA pca(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    pca.fit(X_pca, y_dummy);
    
    MatrixXd components = pca.components();
    EXPECT_EQ(components.rows(), 3);
    EXPECT_EQ(components.cols(), n_features);
    
    // Check that components are reasonable
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < n_features; ++j) {
            EXPECT_FALSE(std::isnan(components(i, j)));
            EXPECT_FALSE(std::isinf(components(i, j)));
        }
    }
}

TEST_F(DecompositionTest, PCAExplainedVarianceRatio) {
    decomposition::PCA pca(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    pca.fit(X_pca, y_dummy);
    
    VectorXd explained_variance_ratio = pca.explained_variance_ratio();
    EXPECT_EQ(explained_variance_ratio.size(), 3);
    
    // Check that explained variance ratios are reasonable
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(explained_variance_ratio(i), 0.0);
        EXPECT_LE(explained_variance_ratio(i), 1.0);
        EXPECT_FALSE(std::isnan(explained_variance_ratio(i)));
        EXPECT_FALSE(std::isinf(explained_variance_ratio(i)));
    }
}

TEST_F(DecompositionTest, PCADifferentNComponents) {
    decomposition::PCA pca(2);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    pca.fit(X_pca, y_dummy);
    
    EXPECT_TRUE(pca.is_fitted());
    
    MatrixXd X_transformed = pca.transform(X_pca);
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), 2);
}

TEST_F(DecompositionTest, PCAGetSetParams) {
    decomposition::PCA pca(3);
    
    Params params = pca.get_params();
    EXPECT_EQ(params["n_components"], "3");
    
    // Test set_params
    Params new_params = {{"n_components", "2"}};
    pca.set_params(new_params);
    
    Params updated_params = pca.get_params();
    EXPECT_EQ(updated_params["n_components"], "2");
}

// TruncatedSVD Tests
TEST_F(DecompositionTest, TruncatedSVDFit) {
    decomposition::TruncatedSVD svd(3);
    svd.fit(X_sparse);
    
    EXPECT_TRUE(svd.is_fitted());
}

TEST_F(DecompositionTest, TruncatedSVDTransform) {
    decomposition::TruncatedSVD svd(3);
    svd.fit(X_sparse);
    
    MatrixXd X_transformed = svd.transform(X_sparse);
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), 3);
    
    // Check that transformed data is reasonable
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FALSE(std::isnan(X_transformed(i, j)));
            EXPECT_FALSE(std::isinf(X_transformed(i, j)));
        }
    }
}

TEST_F(DecompositionTest, TruncatedSVDFitTransform) {
    decomposition::TruncatedSVD svd(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    MatrixXd X_transformed = svd.fit_transform(X_sparse, y_dummy);
    
    EXPECT_TRUE(svd.is_fitted());
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), 3);
}

TEST_F(DecompositionTest, TruncatedSVDComponents) {
    decomposition::TruncatedSVD svd(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    svd.fit(X_sparse, y_dummy);
    
    MatrixXd components = svd.components();
    EXPECT_EQ(components.rows(), n_features);
    EXPECT_EQ(components.cols(), 3);
    
    // Check that components are reasonable
    for (int i = 0; i < n_features; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FALSE(std::isnan(components(i, j)));
            EXPECT_FALSE(std::isinf(components(i, j)));
        }
    }
}

TEST_F(DecompositionTest, TruncatedSVDExplainedVariance) {
    decomposition::TruncatedSVD svd(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    svd.fit(X_sparse, y_dummy);
    
    VectorXd explained_variance = svd.explained_variance();
    EXPECT_EQ(explained_variance.size(), 3);
    
    // Check that explained variance values are reasonable
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(explained_variance(i), 0.0);
        EXPECT_FALSE(std::isnan(explained_variance(i)));
        EXPECT_FALSE(std::isinf(explained_variance(i)));
    }
}

TEST_F(DecompositionTest, TruncatedSVDDifferentNComponents) {
    decomposition::TruncatedSVD svd(2);
    svd.fit(X_sparse);
    
    EXPECT_TRUE(svd.is_fitted());
    
    MatrixXd X_transformed = svd.transform(X_sparse);
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), 2);
}

TEST_F(DecompositionTest, TruncatedSVDGetSetParams) {
    decomposition::TruncatedSVD svd(3);
    
    Params params = svd.get_params();
    EXPECT_EQ(params["n_components"], "3");
    
    // Test set_params
    Params new_params = {{"n_components", "2"}};
    svd.set_params(new_params);
    
    Params updated_params = svd.get_params();
    EXPECT_EQ(updated_params["n_components"], "2");
}

// Edge Cases and Error Handling
TEST_F(DecompositionTest, PCAEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    decomposition::PCA pca(3);
    EXPECT_THROW(pca.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(DecompositionTest, PCASingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    VectorXd y_dummy = VectorXd::Zero(1);
    
    decomposition::PCA pca(1);
    pca.fit(X_single, y_dummy);
    
    EXPECT_TRUE(pca.is_fitted());
    
    MatrixXd X_transformed = pca.transform(X_single);
    EXPECT_EQ(X_transformed.rows(), 1);
    EXPECT_EQ(X_transformed.cols(), 1);
}

TEST_F(DecompositionTest, PCASingleFeature) {
    MatrixXd X_single_feature = MatrixXd::Random(n_samples, 1);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    
    decomposition::PCA pca(1);
    pca.fit(X_single_feature, y_dummy);
    
    EXPECT_TRUE(pca.is_fitted());
    
    MatrixXd X_transformed = pca.transform(X_single_feature);
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), 1);
}

TEST_F(DecompositionTest, PCAZeroComponents) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    decomposition::PCA pca(0);
    EXPECT_THROW(pca.fit(X_pca, y_dummy), std::invalid_argument);
}

TEST_F(DecompositionTest, PCANegativeComponents) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    decomposition::PCA pca(-1);
    EXPECT_THROW(pca.fit(X_pca, y_dummy), std::invalid_argument);
}

TEST_F(DecompositionTest, PCAMoreComponentsThanFeatures) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    decomposition::PCA pca(n_features + 1);
    EXPECT_THROW(pca.fit(X_pca, y_dummy), std::invalid_argument);
}

TEST_F(DecompositionTest, PCAMoreComponentsThanSamples) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    decomposition::PCA pca(n_samples + 1);
    EXPECT_THROW(pca.fit(X_pca, y_dummy), std::invalid_argument);
}

TEST_F(DecompositionTest, PCANotFitted) {
    decomposition::PCA pca(3);
    
    EXPECT_FALSE(pca.is_fitted());
    EXPECT_THROW(pca.transform(X_pca), std::runtime_error);
    EXPECT_THROW(pca.components(), std::runtime_error);
    EXPECT_THROW(pca.explained_variance_ratio(), std::runtime_error);
}

TEST_F(DecompositionTest, PCAWrongFeatureCount) {
    decomposition::PCA pca(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    pca.fit(X_pca, y_dummy);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(pca.transform(X_wrong), std::invalid_argument);
}

TEST_F(DecompositionTest, TruncatedSVDEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    decomposition::TruncatedSVD svd(3);
    EXPECT_THROW(svd.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(DecompositionTest, TruncatedSVDSingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    
    decomposition::TruncatedSVD svd(1);
    svd.fit(X_single);
    
    EXPECT_TRUE(svd.is_fitted());
    
    MatrixXd X_transformed = svd.transform(X_single);
    EXPECT_EQ(X_transformed.rows(), 1);
    EXPECT_EQ(X_transformed.cols(), 1);
}

TEST_F(DecompositionTest, TruncatedSVDSingleFeature) {
    MatrixXd X_single_feature = MatrixXd::Random(n_samples, 1);
    
    decomposition::TruncatedSVD svd(1);
    svd.fit(X_single_feature);
    
    EXPECT_TRUE(svd.is_fitted());
    
    MatrixXd X_transformed = svd.transform(X_single_feature);
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), 1);
}

TEST_F(DecompositionTest, TruncatedSVDZeroComponents) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    decomposition::TruncatedSVD svd(0);
    EXPECT_THROW(svd.fit(X_sparse, y_dummy), std::invalid_argument);
}

TEST_F(DecompositionTest, TruncatedSVDNegativeComponents) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    decomposition::TruncatedSVD svd(-1);
    EXPECT_THROW(svd.fit(X_sparse, y_dummy), std::invalid_argument);
}

TEST_F(DecompositionTest, TruncatedSVDMoreComponentsThanFeatures) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    decomposition::TruncatedSVD svd(n_features + 1);
    EXPECT_THROW(svd.fit(X_sparse, y_dummy), std::invalid_argument);
}

TEST_F(DecompositionTest, TruncatedSVDMoreComponentsThanSamples) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    decomposition::TruncatedSVD svd(n_samples + 1);
    EXPECT_THROW(svd.fit(X_sparse, y_dummy), std::invalid_argument);
}

TEST_F(DecompositionTest, TruncatedSVDNotFitted) {
    decomposition::TruncatedSVD svd(3);
    
    EXPECT_FALSE(svd.is_fitted());
    EXPECT_THROW(svd.transform(X_sparse), std::runtime_error);
    EXPECT_THROW(svd.components(), std::runtime_error);
    EXPECT_THROW(svd.explained_variance(), std::runtime_error);
}

TEST_F(DecompositionTest, TruncatedSVDWrongFeatureCount) {
    decomposition::TruncatedSVD svd(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    svd.fit(X_sparse, y_dummy);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(svd.transform(X_wrong), std::invalid_argument);
}

// Consistency Tests
TEST_F(DecompositionTest, PCAConsistency) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    decomposition::PCA pca1(3);
    decomposition::PCA pca2(3);
    
    pca1.fit(X_pca, y_dummy);
    pca2.fit(X_pca, y_dummy);
    
    MatrixXd X_transformed1 = pca1.transform(X_pca);
    MatrixXd X_transformed2 = pca2.transform(X_pca);
    
    // Results should be consistent (though not necessarily identical)
    EXPECT_EQ(X_transformed1.rows(), X_transformed2.rows());
    EXPECT_EQ(X_transformed1.cols(), X_transformed2.cols());
    EXPECT_EQ(X_transformed1.rows(), n_samples);
    EXPECT_EQ(X_transformed1.cols(), 3);
}

TEST_F(DecompositionTest, TruncatedSVDConsistency) {
    decomposition::TruncatedSVD svd1(3);
    decomposition::TruncatedSVD svd2(3);
    
    svd1.fit(X_sparse);
    svd2.fit(X_sparse);
    
    MatrixXd X_transformed1 = svd1.transform(X_sparse);
    MatrixXd X_transformed2 = svd2.transform(X_sparse);
    
    // Results should be consistent (though not necessarily identical)
    EXPECT_EQ(X_transformed1.rows(), X_transformed2.rows());
    EXPECT_EQ(X_transformed1.cols(), X_transformed2.cols());
    EXPECT_EQ(X_transformed1.rows(), n_samples);
    EXPECT_EQ(X_transformed1.cols(), 3);
}

// Dimensionality Reduction Properties
TEST_F(DecompositionTest, PCADimensionalityReduction) {
    decomposition::PCA pca(2);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    pca.fit(X_pca, y_dummy);
    
    MatrixXd X_transformed = pca.transform(X_pca);
    
    // Should reduce dimensionality from n_features to 2
    EXPECT_EQ(X_transformed.cols(), 2);
    EXPECT_LT(X_transformed.cols(), n_features);
}

TEST_F(DecompositionTest, TruncatedSVDDimensionalityReduction) {
    decomposition::TruncatedSVD svd(2);
    svd.fit(X_sparse);
    
    MatrixXd X_transformed = svd.transform(X_sparse);
    
    // Should reduce dimensionality from n_features to 2
    EXPECT_EQ(X_transformed.cols(), 2);
    EXPECT_LT(X_transformed.cols(), n_features);
}

TEST_F(DecompositionTest, PCAExplainedVarianceOrdering) {
    decomposition::PCA pca(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    pca.fit(X_pca, y_dummy);
    
    VectorXd explained_variance_ratio = pca.explained_variance_ratio();
    
    // Explained variance ratios should be in descending order
    for (int i = 1; i < 3; ++i) {
        EXPECT_GE(explained_variance_ratio(i-1), explained_variance_ratio(i));
    }
}

TEST_F(DecompositionTest, TruncatedSVDExplainedVarianceOrdering) {
    decomposition::TruncatedSVD svd(3);
    svd.fit(X_sparse);
    
    VectorXd explained_variance = svd.explained_variance();
    
    // Explained variance values should be in descending order
    for (int i = 1; i < 3; ++i) {
        EXPECT_GE(explained_variance(i-1), explained_variance(i));
    }
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
