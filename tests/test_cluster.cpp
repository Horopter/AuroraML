#include <gtest/gtest.h>
#include "auroraml/kmeans.hpp"
#include "auroraml/dbscan.hpp"
#include "auroraml/agglomerative.hpp"
#include <Eigen/Dense>

namespace auroraml {
namespace test {

class ClusterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic data
        n_samples = 100;
        n_features = 2;
        
        X = MatrixXd::Random(n_samples, n_features);
        
        // Create well-separated clusters for testing
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
    }
    
    int n_samples, n_features;
    MatrixXd X, X_clustered;
};

// KMeans Tests
TEST_F(ClusterTest, KMeansFit) {
    cluster::KMeans kmeans(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples); // Dummy y for unsupervised learning
    kmeans.fit(X_clustered, y_dummy);
    
    EXPECT_TRUE(kmeans.is_fitted());
}

TEST_F(ClusterTest, KMeansPredict) {
    cluster::KMeans kmeans(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    kmeans.fit(X_clustered, y_dummy);
    
    VectorXi labels = kmeans.labels();
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid cluster assignments
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(labels(i) >= 0 && labels(i) < 3);
    }
}

TEST_F(ClusterTest, KMeansClusterCenters) {
    cluster::KMeans kmeans(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    kmeans.fit(X_clustered, y_dummy);
    
    MatrixXd centers = kmeans.cluster_centers();
    EXPECT_EQ(centers.rows(), 3);
    EXPECT_EQ(centers.cols(), n_features);
    
    // Check that centers are reasonable
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < n_features; ++j) {
            EXPECT_FALSE(std::isnan(centers(i, j)));
            EXPECT_FALSE(std::isinf(centers(i, j)));
        }
    }
}

TEST_F(ClusterTest, KMeansInertia) {
    cluster::KMeans kmeans(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    kmeans.fit(X_clustered, y_dummy);
    
    double inertia = kmeans.inertia();
    EXPECT_GT(inertia, 0.0);
    EXPECT_FALSE(std::isnan(inertia));
    EXPECT_FALSE(std::isinf(inertia));
}

TEST_F(ClusterTest, KMeansDifferentK) {
    cluster::KMeans kmeans(2);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    kmeans.fit(X_clustered, y_dummy);
    
    EXPECT_TRUE(kmeans.is_fitted());
    
    VectorXi labels = kmeans.labels();
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid cluster assignments
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(labels(i) >= 0 && labels(i) < 2);
    }
}

TEST_F(ClusterTest, KMeansGetSetParams) {
    cluster::KMeans kmeans(3);
    
    Params params = kmeans.get_params();
    EXPECT_EQ(params["n_clusters"], "3");
    
    // Test set_params
    Params new_params = {{"n_clusters", "5"}};
    kmeans.set_params(new_params);
    
    Params updated_params = kmeans.get_params();
    EXPECT_EQ(updated_params["n_clusters"], "5");
}

// DBSCAN Tests
TEST_F(ClusterTest, DBSCANFit) {
    cluster::DBSCAN dbscan(0.5, 5);
    dbscan.fit(X_clustered);
    
    EXPECT_TRUE(dbscan.is_fitted());
}

TEST_F(ClusterTest, DBSCANLabels) {
    cluster::DBSCAN dbscan(0.5, 5);
    dbscan.fit(X_clustered);
    
    VectorXi labels = dbscan.labels();
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid (including noise points with label -1)
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(labels(i) >= -1);
    }
}

TEST_F(ClusterTest, DBSCANDifferentEps) {
    cluster::DBSCAN dbscan(0.1, 5);
    dbscan.fit(X_clustered);
    
    EXPECT_TRUE(dbscan.is_fitted());
    
    VectorXi labels = dbscan.labels();
    EXPECT_EQ(labels.size(), n_samples);
}

TEST_F(ClusterTest, DBSCANDifferentMinSamples) {
    cluster::DBSCAN dbscan(0.5, 3);
    dbscan.fit(X_clustered);
    
    EXPECT_TRUE(dbscan.is_fitted());
    
    VectorXi labels = dbscan.labels();
    EXPECT_EQ(labels.size(), n_samples);
}

TEST_F(ClusterTest, DBSCANGetSetParams) {
    cluster::DBSCAN dbscan(0.5, 5);
    
    Params params = dbscan.get_params();
    EXPECT_EQ(params["eps"], "0.500000");
    EXPECT_EQ(params["min_samples"], "5");
    
    // Test set_params
    Params new_params = {{"eps", "0.3"}, {"min_samples", "3"}};
    dbscan.set_params(new_params);
    
    Params updated_params = dbscan.get_params();
    EXPECT_EQ(updated_params["eps"], "0.300000");
    EXPECT_EQ(updated_params["min_samples"], "3");
}

// AgglomerativeClustering Tests
TEST_F(ClusterTest, AgglomerativeClusteringFit) {
    cluster::AgglomerativeClustering agg(3);
    agg.fit(X_clustered);
    
    EXPECT_TRUE(agg.is_fitted());
}

TEST_F(ClusterTest, AgglomerativeClusteringLabels) {
    cluster::AgglomerativeClustering agg(3);
    agg.fit(X_clustered);
    
    VectorXi labels = agg.labels();
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid cluster assignments
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(labels(i) >= 0 && labels(i) < 3);
    }
}

TEST_F(ClusterTest, AgglomerativeClusteringDifferentNClusters) {
    cluster::AgglomerativeClustering agg(2);
    agg.fit(X_clustered);
    
    EXPECT_TRUE(agg.is_fitted());
    
    VectorXi labels = agg.labels();
    EXPECT_EQ(labels.size(), n_samples);
    
    // Check that labels are valid cluster assignments
    for (int i = 0; i < n_samples; ++i) {
        EXPECT_TRUE(labels(i) >= 0 && labels(i) < 2);
    }
}

TEST_F(ClusterTest, AgglomerativeClusteringGetSetParams) {
    cluster::AgglomerativeClustering agg(3);
    
    Params params = agg.get_params();
    EXPECT_EQ(params["n_clusters"], "3");
    
    // Test set_params
    Params new_params = {{"n_clusters", "5"}};
    agg.set_params(new_params);
    
    Params updated_params = agg.get_params();
    EXPECT_EQ(updated_params["n_clusters"], "5");
}

// Edge Cases and Error Handling
TEST_F(ClusterTest, KMeansEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    cluster::KMeans kmeans(3);
    EXPECT_THROW(kmeans.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(ClusterTest, KMeansSingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    VectorXd y_dummy = VectorXd::Zero(1);
    
    cluster::KMeans kmeans(1);
    kmeans.fit(X_single, y_dummy);
    
    EXPECT_TRUE(kmeans.is_fitted());
    
    VectorXi labels = kmeans.labels();
    EXPECT_EQ(labels.size(), 1);
    EXPECT_EQ(labels(0), 0);
}

TEST_F(ClusterTest, KMeansSingleFeature) {
    MatrixXd X_single_feature = MatrixXd::Random(n_samples, 1);
    
    cluster::KMeans kmeans(3);
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    kmeans.fit(X_single_feature, y_dummy);
    
    EXPECT_TRUE(kmeans.is_fitted());
    
    VectorXi labels = kmeans.labels();
    EXPECT_EQ(labels.size(), n_samples);
}

TEST_F(ClusterTest, KMeansZeroClusters) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::KMeans kmeans(0);
    EXPECT_THROW(kmeans.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(ClusterTest, KMeansNegativeClusters) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::KMeans kmeans(-1);
    EXPECT_THROW(kmeans.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(ClusterTest, KMeansMoreClustersThanSamples) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::KMeans kmeans(n_samples + 1);
    EXPECT_THROW(kmeans.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(ClusterTest, KMeansNotFitted) {
    cluster::KMeans kmeans(3);
    
    EXPECT_FALSE(kmeans.is_fitted());
    EXPECT_THROW(kmeans.labels(), std::runtime_error);
    EXPECT_THROW(kmeans.cluster_centers(), std::runtime_error);
    EXPECT_THROW(kmeans.inertia(), std::runtime_error);
}

TEST_F(ClusterTest, DBSCANEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    cluster::DBSCAN dbscan(0.5, 5);
    EXPECT_THROW(dbscan.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(ClusterTest, DBSCANSingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    VectorXd y_dummy = VectorXd::Zero(1);
    
    cluster::DBSCAN dbscan(0.5, 1);
    dbscan.fit(X_single, y_dummy);
    
    EXPECT_TRUE(dbscan.is_fitted());
    
    VectorXi labels = dbscan.labels();
    EXPECT_EQ(labels.size(), 1);
    EXPECT_EQ(labels(0), 0); // Single sample should be cluster 0
}

TEST_F(ClusterTest, DBSCANSingleFeature) {
    MatrixXd X_single_feature = MatrixXd::Random(n_samples, 1);
    
    cluster::DBSCAN dbscan(0.5, 5);
    dbscan.fit(X_single_feature);
    
    EXPECT_TRUE(dbscan.is_fitted());
    
    VectorXi labels = dbscan.labels();
    EXPECT_EQ(labels.size(), n_samples);
}

TEST_F(ClusterTest, DBSCANZeroEps) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::DBSCAN dbscan(0.0, 5);
    EXPECT_THROW(dbscan.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(ClusterTest, DBSCANNegativeEps) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::DBSCAN dbscan(-0.5, 5);
    EXPECT_THROW(dbscan.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(ClusterTest, DBSCANZeroMinSamples) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::DBSCAN dbscan(0.5, 0);
    EXPECT_THROW(dbscan.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(ClusterTest, DBSCANNegativeMinSamples) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::DBSCAN dbscan(0.5, -1);
    EXPECT_THROW(dbscan.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(ClusterTest, DBSCANNotFitted) {
    cluster::DBSCAN dbscan(0.5, 5);
    
    EXPECT_FALSE(dbscan.is_fitted());
    EXPECT_THROW(dbscan.labels(), std::runtime_error);
}

TEST_F(ClusterTest, AgglomerativeClusteringEmptyData) {
    MatrixXd X_empty = MatrixXd::Zero(0, n_features);
    VectorXd y_empty = VectorXd::Zero(0);
    
    cluster::AgglomerativeClustering agg(3);
    EXPECT_THROW(agg.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(ClusterTest, AgglomerativeClusteringSingleSample) {
    MatrixXd X_single = MatrixXd::Random(1, n_features);
    
    cluster::AgglomerativeClustering agg(1);
    agg.fit(X_single);
    
    EXPECT_TRUE(agg.is_fitted());
    
    VectorXi labels = agg.labels();
    EXPECT_EQ(labels.size(), 1);
    EXPECT_EQ(labels(0), 0);
}

TEST_F(ClusterTest, AgglomerativeClusteringSingleFeature) {
    MatrixXd X_single_feature = MatrixXd::Random(n_samples, 1);
    
    cluster::AgglomerativeClustering agg(3);
    agg.fit(X_single_feature);
    
    EXPECT_TRUE(agg.is_fitted());
    
    VectorXi labels = agg.labels();
    EXPECT_EQ(labels.size(), n_samples);
}

TEST_F(ClusterTest, AgglomerativeClusteringZeroClusters) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::AgglomerativeClustering agg(0);
    EXPECT_THROW(agg.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(ClusterTest, AgglomerativeClusteringNegativeClusters) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::AgglomerativeClustering agg(-1);
    EXPECT_THROW(agg.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(ClusterTest, AgglomerativeClusteringMoreClustersThanSamples) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::AgglomerativeClustering agg(n_samples + 1);
    EXPECT_THROW(agg.fit(X_clustered, y_dummy), std::invalid_argument);
}

TEST_F(ClusterTest, AgglomerativeClusteringNotFitted) {
    cluster::AgglomerativeClustering agg(3);
    
    EXPECT_FALSE(agg.is_fitted());
    EXPECT_THROW(agg.labels(), std::runtime_error);
}

// Consistency Tests
TEST_F(ClusterTest, KMeansConsistency) {
    VectorXd y_dummy = VectorXd::Zero(n_samples);
    cluster::KMeans kmeans1(3);
    cluster::KMeans kmeans2(3);
    
    kmeans1.fit(X_clustered, y_dummy);
    kmeans2.fit(X_clustered, y_dummy);
    
    VectorXi labels1 = kmeans1.labels();
    VectorXi labels2 = kmeans2.labels();
    
    // Results should be consistent (though not necessarily identical)
    EXPECT_EQ(labels1.size(), labels2.size());
    EXPECT_EQ(labels1.size(), n_samples);
}

TEST_F(ClusterTest, DBSCANConsistency) {
    cluster::DBSCAN dbscan1(0.5, 5);
    cluster::DBSCAN dbscan2(0.5, 5);
    
    dbscan1.fit(X_clustered);
    dbscan2.fit(X_clustered);
    
    VectorXi labels1 = dbscan1.labels();
    VectorXi labels2 = dbscan2.labels();
    
    // Results should be consistent (though not necessarily identical)
    EXPECT_EQ(labels1.size(), labels2.size());
    EXPECT_EQ(labels1.size(), n_samples);
}

TEST_F(ClusterTest, AgglomerativeClusteringConsistency) {
    cluster::AgglomerativeClustering agg1(3);
    cluster::AgglomerativeClustering agg2(3);
    
    agg1.fit(X_clustered);
    agg2.fit(X_clustered);
    
    VectorXi labels1 = agg1.labels();
    VectorXi labels2 = agg2.labels();
    
    // Results should be consistent (though not necessarily identical)
    EXPECT_EQ(labels1.size(), labels2.size());
    EXPECT_EQ(labels1.size(), n_samples);
}

} // namespace test
} // namespace cxml

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
