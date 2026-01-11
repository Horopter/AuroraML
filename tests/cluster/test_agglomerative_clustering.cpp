#include <gtest/gtest.h>
#include "ingenuityml/agglomerative.hpp"
#include <Eigen/Dense>

namespace ingenuityml {
namespace test {

class AgglomerativeClusteringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data with well-separated clusters
        n_samples = 100;
        n_features = 2;
        
        X_clustered = MatrixXd::Zero(n_samples, n_features);
        for (int i = 0; i < n_samples/3; ++i) {
            X_clustered(i, 0) = 1.0 + 0.1 * (rand() % 100) / 100.0;
            X_clustered(i, 1) = 1.0 + 0.1 * (rand() % 100) / 100.0;
        }
        for (int i = n_samples/3; i < 2*n_samples/3; ++i) {
            X_clustered(i, 0) = -1.0 + 0.1 * (rand() % 100) / 100.0;
            X_clustered(i, 1) = 1.0 + 0.1 * (rand() % 100) / 100.0;
        }
        for (int i = 2*n_samples/3; i < n_samples; ++i) {
            X_clustered(i, 0) = 0.0 + 0.1 * (rand() % 100) / 100.0;
            X_clustered(i, 1) = -1.0 + 0.1 * (rand() % 100) / 100.0;
        }
        
        y_dummy = VectorXd::Zero(n_samples);
    }
    
    int n_samples, n_features;
    MatrixXd X_clustered;
    VectorXd y_dummy;
};

// Positive test cases
TEST_F(AgglomerativeClusteringTest, AgglomerativeClusteringFitPredict) {
    cluster::AgglomerativeClustering agg(3);
    agg.fit(X_clustered, y_dummy);
    VectorXi labels = agg.labels();
    
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid cluster assignments
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_GE(labels(i), 0);
        EXPECT_LT(labels(i), 3);
    }
}

TEST_F(AgglomerativeClusteringTest, AgglomerativeClusteringLabels) {
    cluster::AgglomerativeClustering agg(3);
    agg.fit(X_clustered, y_dummy);
    
    VectorXi labels = agg.labels();
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_GE(labels(i), 0);
        EXPECT_LT(labels(i), 3);
    }
}

TEST_F(AgglomerativeClusteringTest, AgglomerativeClusteringDifferentNClusters) {
    std::vector<int> n_clusters_values = {2, 3, 4};
    
    for (int n_clusters : n_clusters_values) {
        cluster::AgglomerativeClustering agg(n_clusters);
        agg.fit(X_clustered, y_dummy);
        VectorXi labels = agg.labels();
        EXPECT_EQ(labels.size(), n_samples);
        
        // Check that labels are valid
        for (int i = 0; i < n_samples; ++i) {
            EXPECT_GE(labels(i), 0);
            EXPECT_LT(labels(i), n_clusters);
        }
    }
}

// Negative test cases
TEST_F(AgglomerativeClusteringTest, AgglomerativeClusteringEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    cluster::AgglomerativeClustering agg(3);
    EXPECT_THROW(agg.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(AgglomerativeClusteringTest, AgglomerativeClusteringZeroClusters) {
    cluster::AgglomerativeClustering agg(0);
    EXPECT_THROW(agg.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(AgglomerativeClusteringTest, AgglomerativeClusteringNegativeClusters) {
    cluster::AgglomerativeClustering agg(-1);
    EXPECT_THROW(agg.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(AgglomerativeClusteringTest, AgglomerativeClusteringMoreClustersThanSamples) {
    MatrixXd X_small = X_clustered.topRows(5);
    VectorXd y_small = VectorXd::Zero(5);
    
    cluster::AgglomerativeClustering agg(10);
    EXPECT_THROW(agg.fit(X_small, y_small), std::invalid_argument);
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

