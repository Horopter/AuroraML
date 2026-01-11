#include <gtest/gtest.h>
#include "ingenuityml/kmeans.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class KMeansTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data with clear clusters
        n_samples = 100;
        n_features = 2;
        n_clusters = 3;
        
        X = MatrixXd::Zero(n_samples, n_features);
        
        // Create 3 distinct clusters
        for (int i = 0; i < n_samples / 3; ++i) {
            X(i, 0) = 0.0 + MatrixXd::Random(1, 1)(0) * 0.5;  // Cluster 1
            X(i, 1) = 0.0 + MatrixXd::Random(1, 1)(0) * 0.5;
        }
        for (int i = n_samples / 3; i < 2 * n_samples / 3; ++i) {
            X(i, 0) = 3.0 + MatrixXd::Random(1, 1)(0) * 0.5;  // Cluster 2
            X(i, 1) = 3.0 + MatrixXd::Random(1, 1)(0) * 0.5;
        }
        for (int i = 2 * n_samples / 3; i < n_samples; ++i) {
            X(i, 0) = 6.0 + MatrixXd::Random(1, 1)(0) * 0.5;  // Cluster 3
            X(i, 1) = 6.0 + MatrixXd::Random(1, 1)(0) * 0.5;
        }
        
        y = VectorXd::Zero(n_samples);  // Dummy y for fit
    }
    
    int n_samples, n_features, n_clusters;
    MatrixXd X;
    VectorXd y;
};

TEST_F(KMeansTest, KMeansFit) {
    cluster::KMeans kmeans(n_clusters);
    kmeans.fit(X, y);
    
    EXPECT_TRUE(kmeans.is_fitted());
    EXPECT_EQ(kmeans.cluster_centers().rows(), n_clusters);
    EXPECT_EQ(kmeans.cluster_centers().cols(), n_features);
}

TEST_F(KMeansTest, KMeansPredictLabels) {
    cluster::KMeans kmeans(n_clusters);
    kmeans.fit(X, y);
    
    VectorXi labels = kmeans.predict_labels(X);
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid cluster indices
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_GE(labels(i), 0);
        EXPECT_LT(labels(i), n_clusters);
    }
}

TEST_F(KMeansTest, KMeansTransform) {
    cluster::KMeans kmeans(n_clusters);
    kmeans.fit(X, y);
    
    MatrixXd distances = kmeans.transform(X);
    EXPECT_EQ(distances.rows(), n_samples);
    EXPECT_EQ(distances.cols(), n_clusters);
    
    // Check that distances are non-negative
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_clusters; ++j) {
            EXPECT_GE(distances(i, j), 0.0);
        }
    }
}

TEST_F(KMeansTest, KMeansInertia) {
    cluster::KMeans kmeans(n_clusters);
    kmeans.fit(X, y);
    
    double inertia = kmeans.inertia();
    EXPECT_GE(inertia, 0.0);
    EXPECT_FALSE(std::isnan(inertia));
    EXPECT_FALSE(std::isinf(inertia));
}

TEST_F(KMeansTest, KMeansLabels) {
    cluster::KMeans kmeans(n_clusters);
    kmeans.fit(X, y);
    
    VectorXi labels = kmeans.labels();
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid cluster indices
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_GE(labels(i), 0);
        EXPECT_LT(labels(i), n_clusters);
    }
}

} // namespace test
} // namespace ingenuityml

int main(int argc, char **argv) {
    // Enable test shuffling within this file
    ::testing::FLAGS_gtest_shuffle = true;
    ::testing::FLAGS_gtest_random_seed = 42;  // Reproducible shuffle
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
