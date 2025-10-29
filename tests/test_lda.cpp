#include <gtest/gtest.h>
#include "auroraml/lda.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class LDATest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data with clear class separation
        n_samples = 100;
        n_features = 4;
        n_classes = 3;
        
        // Create data with clear class separation
        X = MatrixXd::Zero(n_samples, n_features);
        y = VectorXd::Zero(n_samples);
        
        // Class 0: centered around (0, 0, 0, 0)
        for (int i = 0; i < 30; ++i) {
            X.row(i) = VectorXd::Random(n_features) * 0.5;
            y(i) = 0;
        }
        
        // Class 1: centered around (2, 2, 0, 0)
        for (int i = 30; i < 60; ++i) {
            X.row(i) = VectorXd::Random(n_features) * 0.5;
            X(i, 0) += 2.0;
            X(i, 1) += 2.0;
            y(i) = 1;
        }
        
        // Class 2: centered around (0, 0, 2, 2)
        for (int i = 60; i < 100; ++i) {
            X.row(i) = VectorXd::Random(n_features) * 0.5;
            X(i, 2) += 2.0;
            X(i, 3) += 2.0;
            y(i) = 2;
        }
        
        // Create test data
        X_test = MatrixXd::Random(20, n_features);
        y_test = VectorXd::Zero(20);
        for (int i = 0; i < 20; ++i) {
            y_test(i) = i % n_classes;
        }
    }
    
    int n_samples, n_features, n_classes;
    MatrixXd X, X_test;
    VectorXd y, y_test;
};

// Basic functionality tests
TEST_F(LDATest, LDAFit) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    EXPECT_TRUE(lda.is_fitted());
    EXPECT_EQ(lda.components().rows(), 2);
    EXPECT_EQ(lda.components().cols(), n_features);
    EXPECT_EQ(lda.explained_variance().size(), 2);
    EXPECT_EQ(lda.mean().size(), n_features);
    EXPECT_EQ(lda.class_means().size(), n_classes);
    EXPECT_EQ(lda.classes().size(), n_classes);
}

TEST_F(LDATest, LDATransform) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    MatrixXd X_transformed = lda.transform(X);
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), 2);
    
    // Check that transformed data has reasonable values
    EXPECT_TRUE(X_transformed.allFinite());
}

TEST_F(LDATest, LDAInverseTransform) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    MatrixXd X_transformed = lda.transform(X);
    MatrixXd X_reconstructed = lda.inverse_transform(X_transformed);
    
    EXPECT_EQ(X_reconstructed.rows(), n_samples);
    EXPECT_EQ(X_reconstructed.cols(), n_features);
    EXPECT_TRUE(X_reconstructed.allFinite());
}

TEST_F(LDATest, LDAFitTransform) {
    decomposition::LDA lda(2);
    MatrixXd X_transformed = lda.fit_transform(X, y);
    
    EXPECT_TRUE(lda.is_fitted());
    EXPECT_EQ(X_transformed.rows(), n_samples);
    EXPECT_EQ(X_transformed.cols(), 2);
}

// Parameter tests
TEST_F(LDATest, LDAGetSetParams) {
    decomposition::LDA lda(3);
    
    Params params = lda.get_params();
    EXPECT_EQ(params["n_components"], "3");
    
    Params new_params = {{"n_components", "2"}};
    lda.set_params(new_params);
    
    Params updated_params = lda.get_params();
    EXPECT_EQ(updated_params["n_components"], "2");
}

TEST_F(LDATest, LDAAutoComponents) {
    decomposition::LDA lda(-1);  // Auto-determine components
    lda.fit(X, y);
    
    EXPECT_TRUE(lda.is_fitted());
    EXPECT_EQ(lda.components().rows(), n_classes - 1);  // Should be n_classes - 1
    EXPECT_EQ(lda.components().cols(), n_features);
}

// Accessor method tests
TEST_F(LDATest, LDAComponents) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    MatrixXd components = lda.components();
    EXPECT_EQ(components.rows(), 2);
    EXPECT_EQ(components.cols(), n_features);
    EXPECT_TRUE(components.allFinite());
}

TEST_F(LDATest, LDAExplainedVariance) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    VectorXd explained_var = lda.explained_variance();
    EXPECT_EQ(explained_var.size(), 2);
    EXPECT_TRUE(explained_var.allFinite());
    
    // Explained variance should be non-negative
    for (int i = 0; i < explained_var.size(); ++i) {
        EXPECT_GE(explained_var(i), 0.0);
    }
}

TEST_F(LDATest, LDAExplainedVarianceRatio) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    VectorXd explained_var_ratio = lda.explained_variance_ratio();
    EXPECT_EQ(explained_var_ratio.size(), 2);
    EXPECT_TRUE(explained_var_ratio.allFinite());
    
    // Ratios should sum to 1.0 (approximately)
    double sum = explained_var_ratio.sum();
    EXPECT_NEAR(sum, 1.0, 1e-10);
    
    // Each ratio should be between 0 and 1
    for (int i = 0; i < explained_var_ratio.size(); ++i) {
        EXPECT_GE(explained_var_ratio(i), 0.0);
        EXPECT_LE(explained_var_ratio(i), 1.0);
    }
}

TEST_F(LDATest, LDAMean) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    VectorXd mean = lda.mean();
    EXPECT_EQ(mean.size(), n_features);
    EXPECT_TRUE(mean.allFinite());
}

TEST_F(LDATest, LDAClassMeans) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    std::vector<VectorXd> class_means = lda.class_means();
    EXPECT_EQ(class_means.size(), n_classes);
    
    for (const auto& class_mean : class_means) {
        EXPECT_EQ(class_mean.size(), n_features);
        EXPECT_TRUE(class_mean.allFinite());
    }
}

TEST_F(LDATest, LDAClasses) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    std::vector<int> classes = lda.classes();
    EXPECT_EQ(classes.size(), n_classes);
    
    // Classes should be sorted
    for (int i = 1; i < classes.size(); ++i) {
        EXPECT_LE(classes[i-1], classes[i]);
    }
}

// Error handling tests
TEST_F(LDATest, LDANotFitted) {
    decomposition::LDA lda(2);
    
    EXPECT_FALSE(lda.is_fitted());
    EXPECT_THROW(lda.transform(X), std::runtime_error);
    EXPECT_THROW(lda.inverse_transform(X_test), std::runtime_error);
    EXPECT_THROW(lda.components(), std::runtime_error);
    EXPECT_THROW(lda.explained_variance(), std::runtime_error);
    EXPECT_THROW(lda.explained_variance_ratio(), std::runtime_error);
    EXPECT_THROW(lda.mean(), std::runtime_error);
    EXPECT_THROW(lda.class_means(), std::runtime_error);
    EXPECT_THROW(lda.classes(), std::runtime_error);
}

TEST_F(LDATest, LDAEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    decomposition::LDA lda(2);
    EXPECT_THROW(lda.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(LDATest, LDASingleClass) {
    VectorXd y_single_class = VectorXd::Zero(n_samples);  // All same class
    
    decomposition::LDA lda(2);
    EXPECT_THROW(lda.fit(X, y_single_class), std::invalid_argument);
}

TEST_F(LDATest, LDAMoreComponentsThanFeatures) {
    decomposition::LDA lda(n_features + 1);
    EXPECT_THROW(lda.fit(X, y), std::invalid_argument);
}

TEST_F(LDATest, LDAWrongFeatureCount) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features + 1);
    EXPECT_THROW(lda.transform(X_wrong), std::invalid_argument);
}

TEST_F(LDATest, LDADimensionMismatch) {
    MatrixXd X_wrong = MatrixXd::Random(n_samples, n_features);
    VectorXd y_wrong = VectorXd::Random(n_samples + 1);
    
    decomposition::LDA lda(2);
    EXPECT_THROW(lda.fit(X_wrong, y_wrong), std::invalid_argument);
}

// Consistency tests
TEST_F(LDATest, LDAConsistency) {
    decomposition::LDA lda1(2);
    decomposition::LDA lda2(2);
    
    lda1.fit(X, y);
    lda2.fit(X, y);
    
    MatrixXd X_transformed1 = lda1.transform(X);
    MatrixXd X_transformed2 = lda2.transform(X);
    
    // Results should be identical (within numerical precision)
    EXPECT_TRUE(X_transformed1.isApprox(X_transformed2, 1e-10));
}

TEST_F(LDATest, LDAClassSeparation) {
    decomposition::LDA lda(2);
    lda.fit(X, y);
    
    MatrixXd X_transformed = lda.transform(X);
    
    // Check that classes are well separated in transformed space
    std::vector<VectorXd> class_centers(n_classes);
    std::vector<int> class_counts(n_classes, 0);
    
    for (int i = 0; i < n_classes; ++i) {
        class_centers[i] = VectorXd::Zero(2);
    }
    
    for (int i = 0; i < n_samples; ++i) {
        int class_label = static_cast<int>(y(i));
        class_centers[class_label] += X_transformed.row(i).transpose();
        class_counts[class_label]++;
    }
    
    for (int i = 0; i < n_classes; ++i) {
        class_centers[i] /= class_counts[i];
    }
    
    // Check that class centers are reasonably separated
    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            double distance = (class_centers[i] - class_centers[j]).norm();
            EXPECT_GT(distance, 0.1);  // Classes should be separated
        }
    }
}

} // namespace test
} // namespace cxml
